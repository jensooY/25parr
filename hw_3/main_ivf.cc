#include <vector>      
#include <string>       
#include <fstream>       
#include <iostream>      
#include <set>            // 用于 std::set 存储标准答案ID，便于快速查找
#include <chrono>         // 用于高精度计时 (std::chrono)
#include <iomanip>        // 用于输出格式控制，如设置精度 
#include <queue>        
#include <utility>        // 用于 std::pair
#include <cstdint>        // 用于固定宽度的整数类型，如 uint32_t, int64_t
#include <algorithm>     
// --- 自定义头文件 ---
#include "simd_distance.h" 
#include "kmeans.h"      
#include "ivf_index.h"   
#include "pq_utils.h" 

using namespace std;

// --- 1. 数据加载函数 ---
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
      std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    return data;
}
// --- 2. 查询结果结构体 ---
struct SearchResult {
    float recall;       // 召回率
    int64_t latency;    // 延迟 (单位：微秒 us)
};

// --- main 函数入口 ---
int main(int argc, char *argv[]) 
{
    // --- A. 初始化和数据加载阶段 ---
    cout << "--- [main2.cc] IVF 测试 ---" << endl;
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // --- B. 实验参数设置 ---
    test_number = 2000; 
    const size_t k = 10;
    const size_t nk_for_reranking = 1500; // 【新增可调参数】IVF-PQ ADC阶段召回的候选数量，用于后续精确重排

    // --- 【新增】全局PQ码本训练和数据编码 (与SIMD实验中逻辑类似) ---
    PQCodebook global_pq_codebook;        // 全局PQ码本对象
    uint8_t* global_base_pq_codes = nullptr; // 指向PQ编码后基础数据的指针

    // 初始化 global_pq_codebook.params (M, Ks, D, D_sub, n_train, kmeans_iters)
    global_pq_codebook.params.M = 4;     // 示例：子空间数量
    global_pq_codebook.params.Ks = 256;  // 示例：每个子空间的码字数量
    global_pq_codebook.params.D = vecdim;
    global_pq_codebook.params.D_sub = vecdim / global_pq_codebook.params.M;
    global_pq_codebook.params.n_train = base_number; // 使用全部base数据进行训练
    global_pq_codebook.params.kmeans_iters = 20;     // K-Means迭代次数

    cout << "\n--- 全局PQ码本训练与数据编码 ---" << endl;
    train_pq(base, global_pq_codebook); 
    cout << "PQ码本训练完成。" << endl;

    global_base_pq_codes = new uint8_t[base_number * global_pq_codebook.params.M];
    cout << "开始对基础数据进行PQ编码..." << endl;
    for (size_t i = 0; i < base_number; ++i) // 编码整个数据集
        encode_pq(base + i * vecdim, global_pq_codebook, global_base_pq_codes + i * global_pq_codebook.params.M);
    cout << "基础数据PQ编码完成。" << endl;


    // IVF 索引构建的关键参数 (需要调优的)
    size_t num_ivf_clusters = 2048; // KMeans聚类的目标簇数量
    int kmeans_max_iterations = 20; // KMeans算法的最大迭代次数 

    // --- 【重要开关】控制是否在IVF索引构建时启用内存重排 ---
    bool enable_memory_reordering = true; // 或 false，根据你的测试计划

    cout << "\n--- IVF 索引构建阶段 ---" << endl;
    // 创建IVFIndex对象实例，传入内存重排开关
    // --- 【修改】创建IVFIndex对象时传入PQ相关指针 ---
    IVFIndex ivf_search_index(base,                    // 原始浮点base数据 (用于KMeans和精确重排)
                              global_base_pq_codes,   // 【传入】PQ编码后的基础数据
                              &global_pq_codebook,    // 【传入】PQ码本指针
                              base_number, vecdim,
                              enable_memory_reordering);
    cout << "  IVFIndex对象创建时，内存重排已 "
         << (enable_memory_reordering ? "启用" : "禁用") << endl;
    cout << "开始构建IVF索引 (KMeans聚类)..." << endl;
    cout << "  目标簇数量: " << num_ivf_clusters << endl;
    cout << "  KMeans最大迭代次数: " << kmeans_max_iterations << endl;

    // 记录索引构建开始时间
    auto build_start_time = chrono::high_resolution_clock::now();

    // 调用IVFIndex对象的build方法来执行KMeans并构建内部索引结构
    if (!ivf_search_index.build(num_ivf_clusters, kmeans_max_iterations))
     {
        cerr << "错误: 构建IVF索引失败。程序退出。" << endl;
        delete[] test_query;
        delete[] test_gt;
        delete[] base;
        return 1; // 返回错误码1，表示程序异常结束
    }
    // 记录索引构建结束时间并计算耗时
    auto build_end_time = chrono::high_resolution_clock::now();
    auto build_duration_ms = chrono::duration_cast<chrono::milliseconds>(build_end_time - build_start_time);
    cout << "IVF索引构建完成。耗时: " << build_duration_ms.count() << " ms" << endl;

    // --- C. 在线查询与评估阶段 ---
    bool run_single_thread_test = false;  // 设置为 true 来运行单线程测试
    bool run_pthread_test = false;       // 设置为 true 来运行【单阶段】Pthread测试 
    bool run_pthread_two_stage_test = false; // <<< 新增：设置为 true 来运行两阶段Pthread测试
    bool run_ivfpq_pthread_with_rerank_test = false; // 新增，用于测试Pthread IVF-PQ + Rerank
    bool run_openmp_test = true; //测试OpenMP

    if (run_single_thread_test) 
    {
        vector<SearchResult> results_ivf_single(test_number);
        // IVF查询的关键参数 (调优参数)
        size_t nprobe_param = 20; // 查询时探查（搜索）的最近簇的数量

        cout << "\n--- IVF 单线程查询阶段 ---" << endl;
        cout << "  探查簇数量 (nprobe): " << nprobe_param << endl;

        for (size_t i = 0; i < test_number; ++i) 
        {
            const float* current_query_vector = test_query + i * vecdim;
            // 记录单次查询开始时间
            auto query_start_time = chrono::high_resolution_clock::now();
            // 调用IVFIndex对象的search_single_thread方法进行搜索
            priority_queue<pair<float, uint32_t>> current_results_pq =
                ivf_search_index.search_single_thread(current_query_vector, k, nprobe_param);
            // 记录单次查询结束时间并计算延迟 (单位：微秒)
            auto query_end_time = chrono::high_resolution_clock::now();
            long long query_latency_us = chrono::duration_cast<chrono::microseconds>(query_end_time - query_start_time).count();

            // --- D. 评估当前查询的召回率
            set<uint32_t> ground_truth_ids; 
            for (size_t j = 0; j < k; ++j) 
                ground_truth_ids.insert(test_gt[j + i * test_gt_d]);

            size_t correct_hits = 0; // 记录搜索结果中命中标准答案的数量
            while (!current_results_pq.empty()) 
            {
                uint32_t result_id = current_results_pq.top().second; // 获取结果中的数据点ID
                if (ground_truth_ids.count(result_id)) 
                    correct_hits++; // 如果命中，计数加1
                current_results_pq.pop(); // 移除已处理的元素
            }

            // 计算召回率
            float recall_value = static_cast<float>(correct_hits) / k;

            // 存储本次查询的召回率和延迟
            results_ivf_single[i] = {recall_value, query_latency_us};
        }
        cout << "所有查询执行完毕。" << endl;

        // --- E. 计算并输出平均性能指标 ---
        float total_recall = 0.0f;
        double total_latency_us_sum = 0.0;

        for (size_t i = 0; i < test_number; ++i) 
        {
            total_recall += results_ivf_single[i].recall;
            total_latency_us_sum += results_ivf_single[i].latency;
        }

        cout << "\n--- IVF 单线程最终测试结果 ---" << endl;
        cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param << ", k=" << k << endl;
        cout << "  平均召回率 Recall@" << k << ": " << total_recall / test_number << endl;
        cout << "  平均查询延迟 (us): " << total_latency_us_sum / test_number << endl;

        cout << "------------------------------------" << endl;
    }

    if (run_pthread_test) 
    {
        // --- Pthread 并行在线查询与评估阶段 ---
        vector<SearchResult> results_ivf_pthread(test_number);
        size_t nprobe_param = 20;      
        int num_worker_threads = 1; // 可调参数: Pthread工作线程数量 (例如 1, 2, 4, 8)

        cout << "\n--- [执行] IVF Pthread (单阶段并行) 查询阶段 ("
             << (ivf_search_index.use_memory_reordering_ ? "使用内存重排数据" : "使用原始数据") << ") ---" << endl;
        cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param
             << ", k=" << k << ", 工作线程数=" << num_worker_threads << endl;

        for (size_t i = 0; i < test_number; ++i) 
        {
            const float* current_query_vector = test_query + i * vecdim;
            auto query_start_time_pthread = chrono::high_resolution_clock::now();
            // 调用IVFIndex对象的search_pthread方法进行搜索
            priority_queue<pair<float, uint32_t>> current_results_pq_pthread =
                ivf_search_index.search_pthread(current_query_vector, k, nprobe_param, num_worker_threads);
            auto query_end_time_pthread = chrono::high_resolution_clock::now();
            long long query_latency_us_pthread = chrono::duration_cast<chrono::microseconds>(query_end_time_pthread - query_start_time_pthread).count();

            // --- 评估当前查询的召回率
            set<uint32_t> ground_truth_ids_pthread;
            for (size_t j = 0; j < k; ++j) 
                ground_truth_ids_pthread.insert(test_gt[j + i * test_gt_d]);
            size_t correct_hits_pthread = 0;
            while (!current_results_pq_pthread.empty()) 
            {
                if (ground_truth_ids_pthread.count(current_results_pq_pthread.top().second)) 
                    correct_hits_pthread++;
                current_results_pq_pthread.pop();
            }
            float recall_value_pthread = (k == 0) ? 1.0f : (ground_truth_ids_pthread.empty() ? ((correct_hits_pthread == 0) ? 1.0f : 0.0f) : static_cast<float>(correct_hits_pthread) / k);
            results_ivf_pthread[i] = {recall_value_pthread, query_latency_us_pthread};
        }

        float total_recall_pthread_avg = 0.0f;
        double total_latency_us_sum_pthread_avg = 0.0;
        for (size_t i = 0; i < test_number; ++i) 
        {
            total_recall_pthread_avg += results_ivf_pthread[i].recall;
            total_latency_us_sum_pthread_avg += results_ivf_pthread[i].latency;
        }
        cout << "\n--- IVF Pthread (单阶段并行) 最终测试结果 ("<< (ivf_search_index.use_memory_reordering_ ? "已内存重排" : "未内存重排") << ") ---" << endl;
        if (test_number > 0) {
            cout << "  平均召回率 Recall@" << k << ": " << total_recall_pthread_avg / test_number << endl;
            cout << "  平均查询延迟 (us): " << total_latency_us_sum_pthread_avg / test_number << endl;
        }
        cout << "------------------------------------" << endl;
    }

     // --- 【新增】两阶段 Pthread 并行在线查询与评估阶段 ---
     if (run_pthread_two_stage_test)
    {
        vector<SearchResult> results_ivf_pthread_two_stage(test_number);
        // --- 使用与之前测试一致的 nprobe_param ---
        size_t nprobe_param = 20; // 基于你单线程找到的最优值
        // --- 【新增可调参数】为两阶段并行分别设置线程数 ---
        int num_threads_for_centroids = 8; // <<< 可调参数：用于并行计算质心距离的线程数
        int num_threads_for_clusters = 3;  // <<< 可调参数：用于并行在候选簇内搜索的线程数

        cout << "\n--- [执行] IVF Pthread (两阶段并行) 查询阶段 ---" << endl;
        cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param
             << ", k=" << k << endl;
        cout << "  质心探查线程数=" << num_threads_for_centroids
             << ", 簇内搜索线程数=" << num_threads_for_clusters << endl;

        for (size_t i = 0; i < test_number; ++i)
        {
            const float* current_query_vector = test_query + i * vecdim;
            auto query_start_time_pthread_ts = chrono::high_resolution_clock::now(); // _ts 后缀表示 TwoStage

            // --- 调用【新增】的 search_pthread_two_stage 方法 ---
            priority_queue<pair<float, uint32_t>> current_results_pq_pthread_ts =
                ivf_search_index.search_pthread_two_stage(current_query_vector, k, nprobe_param,
                                                          num_threads_for_centroids,
                                                          num_threads_for_clusters);

            auto query_end_time_pthread_ts = chrono::high_resolution_clock::now();
            long long query_latency_us_pthread_ts = chrono::duration_cast<chrono::microseconds>(query_end_time_pthread_ts - query_start_time_pthread_ts).count();

            // --- 评估当前查询的召回率 (与之前逻辑相同) ---
            set<uint32_t> ground_truth_ids_pthread_ts;
            for (size_t j = 0; j < k; ++j)
                ground_truth_ids_pthread_ts.insert(test_gt[j + i * test_gt_d]);

            size_t correct_hits_pthread_ts = 0;
            while (!current_results_pq_pthread_ts.empty())
            {
                if (ground_truth_ids_pthread_ts.count(current_results_pq_pthread_ts.top().second))
                    correct_hits_pthread_ts++;
                current_results_pq_pthread_ts.pop();
            }
            float recall_value_pthread_ts = (k == 0) ? 1.0f : (ground_truth_ids_pthread_ts.empty() ? ((correct_hits_pthread_ts == 0) ? 1.0f : 0.0f) : static_cast<float>(correct_hits_pthread_ts) / k);
            results_ivf_pthread_two_stage[i] = {recall_value_pthread_ts, query_latency_us_pthread_ts};
        }
        cout << "所有两阶段Pthread查询执行完毕。" << endl;

        // --- 计算并输出平均性能指标 ---
        float total_recall_pthread_ts_avg = 0.0f;
        double total_latency_us_sum_pthread_ts_avg = 0.0;
        for (size_t i = 0; i < test_number; ++i)
        {
            total_recall_pthread_ts_avg += results_ivf_pthread_two_stage[i].recall;
            total_latency_us_sum_pthread_ts_avg += results_ivf_pthread_two_stage[i].latency;
        }
        cout << "\n--- IVF Pthread (两阶段并行) 最终测试结果 ---" << endl;
        if (test_number > 0) {
            cout << "  平均召回率 Recall@" << k << ": " << total_recall_pthread_ts_avg / test_number << endl;
            cout << "  平均查询延迟 (us): " << total_latency_us_sum_pthread_ts_avg / test_number << endl;
        }
        cout << "------------------------------------" << endl;
    }



    // --- 【新增】用于存储各阶段平均耗时的变量 ---
    double avg_latency_adc_recall_total = 0;
    double avg_latency_extract_ids_total = 0; // 从PQ结果提取ID到vector的时间
    double avg_latency_rerank_parallel_total = 0;
    double avg_latency_rerank_sort_total = 0;
    double avg_latency_overall_query_total = 0; // 这个会和你现有的总延迟类似
    double avg_latency_lut_computation_total = 0; // <<< 新增：用于累加LUT计算耗时


    // --- 【新增】Pthread IVF-PQ (ADC召回 + 【并行】精确重排) 测试块 ---
    if (run_ivfpq_pthread_with_rerank_test)
    {
        vector<SearchResult> results_ivfpq_pthread_rerank(test_number);
        // IVF-PQ 查询参数
        size_t nprobe_param = 20;      // 基于你之前的最优参数
        // Pthread 工作线程数
        int num_worker_threads_for_pq_adc = 3; // 用于ADC召回阶段的线程数
        int num_threads_for_exact_rerank = 3;  // 【可调参数】用于并行精确重排的线程数


        cout << "\n--- [执行] IVF-PQ Pthread (ADC召回 + 并行精确重排) 查询阶段 (" // 更新打印信息
             << (ivf_search_index.use_memory_reordering_ ? "使用内存重排数据" : "使用原始数据") << ") ---" << endl;
        cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param
             << ", 最终k=" << k << ", ADC召回Nk=" << nk_for_reranking << endl; // 使用 k 而不是 k_final_results
        cout << "  ADC召回线程数=" << num_worker_threads_for_pq_adc
             << ", 精确重排线程数=" << num_threads_for_exact_rerank << endl;


        // 【新增】累加各阶段耗时
        long long total_lut_computation_us_pr = 0;    // _pr 后缀表示 Parallel Rerank
        long long total_adc_recall_us_pr = 0;
        long long total_extract_ids_us_pr = 0;
        long long total_rerank_parallel_us_pr = 0;   // 保持这个名字，表示并行重排阶段
        long long total_rerank_sort_us_pr = 0;
        long long total_overall_query_us_pr = 0;


        for (size_t i = 0; i < test_number; ++i)
        {
            const float* current_query_vector = test_query + i * vecdim;

            // --- 计时点 L0_pr: 整体查询开始 / LUT计算开始 ---
            auto t_l0_pr_overall_query_start = chrono::high_resolution_clock::now();

            // 1. 为当前查询预计算 LUT (在主线程中)
            std::vector<float> lut_main_pr(global_pq_codebook.params.M * global_pq_codebook.params.Ks); // _pr 后缀
            compute_inner_product_lut_neon(current_query_vector, global_pq_codebook, lut_main_pr.data());

            // --- 计时点 L1_pr: LUT计算结束 / ADC召回开始 ---
            auto t_l1_pr_lut_end_adc_recall_start = chrono::high_resolution_clock::now();


            // --- 阶段 1: 调用 Pthread IVF-PQ ADC 搜索方法，召回 top-Nk_for_reranking 个候选 ---
            priority_queue<pair<float, uint32_t>> adc_candidates_pq =
                ivf_search_index.search_pthread_ivfpq_adc(current_query_vector,
                                                          nk_for_reranking,
                                                          nprobe_param,
                                                          num_worker_threads_for_pq_adc,
                                                          lut_main_pr.data()); // <<< 传递预计算的LUT
            // --- 计时点 T2_pr: ADC召回结束 / 提取ID开始 ---
            auto t2_pr_adc_recall_end_extract_start = chrono::high_resolution_clock::now();


            vector<uint32_t> adc_candidate_original_ids;
            adc_candidate_original_ids.reserve(adc_candidates_pq.size());
            while(!adc_candidates_pq.empty()){
                adc_candidate_original_ids.push_back(adc_candidates_pq.top().second);
                adc_candidates_pq.pop();
            }
            // std::reverse(adc_candidate_original_ids.begin(), adc_candidate_original_ids.end()); // 可选

            // --- 计时点 T2.5_pr: 提取ID结束 / 并行重排开始 ---
            auto t2_5_pr_extract_end_rerank_parallel_start = chrono::high_resolution_clock::now();


            vector<pair<float, uint32_t>> rerank_candidates_exact_all;
            rerank_candidates_exact_all.reserve(adc_candidate_original_ids.size());

            if (!adc_candidate_original_ids.empty()) {
                // --- 定义用于精确重排的Pthread数据和工作函数 ---
                struct RerankThreadData {
                    const float* query_vec_ptr;
                    const float* base_data_ptr;
                    size_t dim_val;
                    const vector<uint32_t>* ids_to_rerank_ptr;
                    size_t start_idx_in_ids;
                    size_t end_idx_in_ids;
                    vector<pair<float, uint32_t>> local_exact_results;
                    int thread_label_for_print;
                };

                auto exact_rerank_worker = [](void* arg) -> void* { // Lambda作为工作函数
                    RerankThreadData* data = static_cast<RerankThreadData*>(arg);
                    // std::cout << "[Rerank Worker] Label: " << data->thread_label_for_print << " TID: " << pthread_self()
                    //           << " Range: [" << data->start_idx_in_ids << ", " << data->end_idx_in_ids -1 << "]" << std::endl;
                    data->local_exact_results.reserve(data->end_idx_in_ids - data->start_idx_in_ids);
                    for (size_t idx = data->start_idx_in_ids; idx < data->end_idx_in_ids; ++idx) {
                        uint32_t original_id = (*data->ids_to_rerank_ptr)[idx];
                        const float* candidate_vec_float = data->base_data_ptr + original_id * data->dim_val;
                        float exact_dist = compute_inner_product_distance_neon_optimized(data->query_vec_ptr, candidate_vec_float, data->dim_val);
                        data->local_exact_results.push_back({exact_dist, original_id});
                    }
                    pthread_exit(nullptr);
                    return nullptr;
                };

                // --- 【修正】使用 num_threads_for_exact_rerank 计算 actual_threads_for_rerank ---
                int actual_threads_for_rerank = num_threads_for_exact_rerank; // 用户设定的目标线程数
                if (adc_candidate_original_ids.size() < (size_t)actual_threads_for_rerank) { // 注意类型转换
                     actual_threads_for_rerank = adc_candidate_original_ids.size();
                }
                if (actual_threads_for_rerank <= 0) actual_threads_for_rerank = 1;


                vector<pthread_t> rerank_threads(actual_threads_for_rerank); // 使用 actual_threads_for_rerank
                vector<RerankThreadData> rerank_thread_data_array(actual_threads_for_rerank); // 使用 actual_threads_for_rerank
                size_t ids_per_rerank_thread = (adc_candidate_original_ids.size() + actual_threads_for_rerank - 1) / actual_threads_for_rerank; // 使用 actual_threads_for_rerank

                for (int t_idx = 0; t_idx < actual_threads_for_rerank; ++t_idx) { // 使用 actual_threads_for_rerank
                    rerank_thread_data_array[t_idx].query_vec_ptr = current_query_vector;
                    rerank_thread_data_array[t_idx].base_data_ptr = base;
                    rerank_thread_data_array[t_idx].dim_val = vecdim;
                    rerank_thread_data_array[t_idx].ids_to_rerank_ptr = &adc_candidate_original_ids;
                    rerank_thread_data_array[t_idx].start_idx_in_ids = t_idx * ids_per_rerank_thread;
                    rerank_thread_data_array[t_idx].end_idx_in_ids = std::min(((size_t)t_idx + 1) * ids_per_rerank_thread, adc_candidate_original_ids.size()); // 确保类型一致
                    rerank_thread_data_array[t_idx].thread_label_for_print = t_idx;

                    if (rerank_thread_data_array[t_idx].start_idx_in_ids < rerank_thread_data_array[t_idx].end_idx_in_ids) {
                        pthread_create(&rerank_threads[t_idx], nullptr, exact_rerank_worker, &rerank_thread_data_array[t_idx]);
                    } else {
                        rerank_threads[t_idx] = 0;
                    }
                }

                for (int t_idx = 0; t_idx < actual_threads_for_rerank; ++t_idx) { // 【修正】使用 actual_threads_for_rerank
                    if (rerank_threads[t_idx] != 0) {
                        pthread_join(rerank_threads[t_idx], nullptr);
                        // 确保 t_idx 在 rerank_thread_data_array 的有效范围内
                        if (t_idx < rerank_thread_data_array.size()) {
                            rerank_candidates_exact_all.insert(rerank_candidates_exact_all.end(),
                                                            rerank_thread_data_array[t_idx].local_exact_results.begin(),
                                                            rerank_thread_data_array[t_idx].local_exact_results.end());
                        }
                    }
                }
            } // end if (!adc_candidate_original_ids.empty())

            // --- 计时点 T3_pr: 并行重排结束 / 最终排序开始 ---
            auto t3_pr_rerank_parallel_end_sort_start = chrono::high_resolution_clock::now();


            sort(rerank_candidates_exact_all.begin(), rerank_candidates_exact_all.end());

            // --- 计时点 T4_pr: 最终排序结束 / 整个查询结束 ---
            auto t4_pr_sort_end_query_end = chrono::high_resolution_clock::now();


            // --- 累加各阶段耗时 ---
            total_lut_computation_us_pr += chrono::duration_cast<chrono::microseconds>(t_l1_pr_lut_end_adc_recall_start - t_l0_pr_overall_query_start).count();
            total_adc_recall_us_pr += chrono::duration_cast<chrono::microseconds>(t2_pr_adc_recall_end_extract_start - t_l1_pr_lut_end_adc_recall_start).count();
            total_extract_ids_us_pr += chrono::duration_cast<chrono::microseconds>(t2_5_pr_extract_end_rerank_parallel_start - t2_pr_adc_recall_end_extract_start).count();
            total_rerank_parallel_us_pr += chrono::duration_cast<chrono::microseconds>(t3_pr_rerank_parallel_end_sort_start - t2_5_pr_extract_end_rerank_parallel_start).count();
            total_rerank_sort_us_pr += chrono::duration_cast<chrono::microseconds>(t4_pr_sort_end_query_end - t3_pr_rerank_parallel_end_sort_start).count();
            total_overall_query_us_pr += chrono::duration_cast<chrono::microseconds>(t4_pr_sort_end_query_end - t_l0_pr_overall_query_start).count();


            // --- D. 评估当前查询的召回率 (基于最终的 k 个精确重排结果) ---
            set<uint32_t> ground_truth_ids_ivfpq;
            for (size_t j = 0; j < k; ++j)
            {
                if ( ( (size_t)i * test_gt_d + j ) < ( test_number * test_gt_d ) ) {
                     ground_truth_ids_ivfpq.insert(test_gt[j + i * test_gt_d]);
                }
            }
            size_t correct_hits_ivfpq = 0;
            for (size_t r_idx = 0; r_idx < k && r_idx < rerank_candidates_exact_all.size(); ++r_idx)
            {
                if (ground_truth_ids_ivfpq.count(rerank_candidates_exact_all[r_idx].second))
                    correct_hits_ivfpq++;
            }
            float recall_value_ivfpq;
            if (k == 0) recall_value_ivfpq = 1.0f;
            else if (ground_truth_ids_ivfpq.empty() && k > 0) recall_value_ivfpq = (correct_hits_ivfpq == 0) ? 1.0f : 0.0f;
            else recall_value_ivfpq = static_cast<float>(correct_hits_ivfpq) / k;

            results_ivfpq_pthread_rerank[i] = {recall_value_ivfpq, chrono::duration_cast<chrono::microseconds>(t4_pr_sort_end_query_end - t_l0_pr_overall_query_start).count()};
        }
        cout << "所有IVF-PQ Pthread (ADC+并行Rerank) 查询执行完毕。" << endl;

        // --- E. 计算并输出平均性能指标 ---
        float total_recall_ivfpq_pthread_avg = 0.0f;
        for (size_t i = 0; i < test_number; ++i)
        {
            total_recall_ivfpq_pthread_avg += results_ivfpq_pthread_rerank[i].recall;
        }
        cout << "\n--- IVF-PQ Pthread (ADC+并行Rerank) 最终测试结果 ("
             << (ivf_search_index.use_memory_reordering_ ? "已内存重排" : "未内存重排") << ") ---" << endl;
        if (test_number > 0) {
            cout << "  平均召回率 Recall@" << k << ": " << total_recall_ivfpq_pthread_avg / test_number << endl;
            cout << "  平均查询总延迟 (us):               " << (double)total_overall_query_us_pr / test_number << endl;
            cout << "  平均LUT预计算阶段延迟 (us):      " << (double)total_lut_computation_us_pr / test_number << endl;
            cout << "  平均ADC召回阶段延迟 (us):          " << (double)total_adc_recall_us_pr / test_number << endl;
            cout << "  平均提取ID阶段延迟 (us):           " << (double)total_extract_ids_us_pr / test_number << endl;
            cout << "  平均并行重排计算阶段延迟 (us):   " << (double)total_rerank_parallel_us_pr / test_number << endl;
            cout << "  平均最终排序阶段延迟 (us):       " << (double)total_rerank_sort_us_pr / test_number << endl;
        }
        cout << "------------------------------------" << endl;
    }

    //     // --- Pthread IVF-PQ (ADC召回 + 【串行】精确重排) 测试块 ---
    // if (run_ivfpq_pthread_with_rerank_test) // 假设这个开关控制的是 “ADC召回并行，重排串行” 的版本
    // {
    //     vector<SearchResult> results_ivfpq_pthread_rerank(test_number);
    //     // IVF-PQ 查询参数
    //     size_t nprobe_param = 20;      // 基于你之前的最优参数
    //     // Pthread 工作线程数 (用于簇内ADC搜索)
    //     int num_worker_threads_for_pq_adc = 3;

    //     cout << "\n--- [执行] IVF-PQ Pthread (ADC召回 + 【串行】精确重排) 查询阶段 (" // 更新打印信息
    //          << (ivf_search_index.use_memory_reordering_ ? "使用内存重排数据" : "使用原始数据") << ") ---" << endl;
    //     cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param
    //          << ", 最终k=" << k << ", ADC召回Nk=" << nk_for_reranking // 使用你之前定义的 nk_for_reranking
    //          << ", ADC召回工作线程数=" << num_worker_threads_for_pq_adc << endl; // 修改打印信息

    //     // --- 用于累加各阶段耗时的变量 (针对这个测试块) ---
    //     long long total_lut_computation_us_s = 0;    // 【新增】LUT计算耗时 (后缀_s表示此测试块)
    //     long long total_adc_recall_us_s = 0;         // ADC召回 (不含LUT)
    //     long long total_extract_ids_us_s = 0;        // 提取ID
    //     long long total_rerank_serial_us_s = 0;      // 串行重排计算
    //     long long total_rerank_sort_us_s = 0;        // 最终排序
    //     long long total_overall_query_us_s = 0;      // 总查询 (从LUT开始)

    //     for (size_t i = 0; i < test_number; ++i)
    //     {
    //         const float* current_query_vector = test_query + i * vecdim;

    //         // --- 计时点 L0_s: 整体查询开始 / LUT计算开始 ---
    //         auto t_l0_s_overall_query_start = chrono::high_resolution_clock::now();

    //         // 1. 为当前查询预计算 LUT (在主线程中)
    //         std::vector<float> lut_main_s(global_pq_codebook.params.M * global_pq_codebook.params.Ks); // _s 后缀
    //         compute_inner_product_lut_neon(current_query_vector, global_pq_codebook, lut_main_s.data());

    //         // --- 计时点 L1_s: LUT计算结束 / ADC召回开始 ---
    //         auto t_l1_s_lut_end_adc_recall_start = chrono::high_resolution_clock::now();

    //         // 阶段 1: 调用 Pthread IVF-PQ ADC 搜索方法，召回 top-Nk_for_reranking 个候选
    //         priority_queue<pair<float, uint32_t>> adc_candidates_pq =
    //             ivf_search_index.search_pthread_ivfpq_adc(current_query_vector,
    //                                                       nk_for_reranking,
    //                                                       nprobe_param,
    //                                                       num_worker_threads_for_pq_adc,
    //                                                       lut_main_s.data()); // 将预计算的LUT传递进去

    //         // --- 计时点 T2_s: ADC召回结束 / 提取ID开始 ---
    //         auto t2_s_adc_recall_end_extract_start = chrono::high_resolution_clock::now();

    //         vector<uint32_t> adc_candidate_original_ids_s;
    //         adc_candidate_original_ids_s.reserve(adc_candidates_pq.size());
    //         while(!adc_candidates_pq.empty())
    //         {
    //             adc_candidate_original_ids_s.push_back(adc_candidates_pq.top().second);
    //             adc_candidates_pq.pop();
    //         }
    //         // std::reverse(adc_candidate_original_ids_s.begin(), adc_candidate_original_ids_s.end()); // 可选

    //         // --- 计时点 T2.5_s: 提取ID结束 / 串行重排计算开始 ---
    //         auto t2_5_s_extract_end_rerank_serial_start = chrono::high_resolution_clock::now();

    //         vector<pair<float, uint32_t>> rerank_candidates_exact; // 存储 (精确距离, 原始ID)
    //         rerank_candidates_exact.reserve(nk_for_reranking); // 使用 nk_for_reranking

    //         for (uint32_t original_candidate_id : adc_candidate_original_ids_s) // 遍历提取出的ID
    //         {
    //             if (original_candidate_id < base_number) // 安全检查
    //             {
    //                 const float* candidate_float_vector = base + original_candidate_id * vecdim;
    //                 float exact_distance = compute_inner_product_distance_neon_optimized(current_query_vector, candidate_float_vector, vecdim);
    //                 rerank_candidates_exact.push_back({exact_distance, original_candidate_id});
    //             }
    //         }

    //         // --- 计时点 T3_s: 串行重排计算结束 / 最终排序开始 ---
    //         auto t3_s_rerank_serial_end_sort_start = chrono::high_resolution_clock::now();


    //         sort(rerank_candidates_exact.begin(), rerank_candidates_exact.end());

    //         // --- 计时点 T4_s: 最终排序结束 / 整个查询结束 ---
    //         auto t4_s_sort_end_query_end = chrono::high_resolution_clock::now();


    //         // --- 累加各阶段耗时 ---
    //         total_lut_computation_us_s += chrono::duration_cast<chrono::microseconds>(t_l1_s_lut_end_adc_recall_start - t_l0_s_overall_query_start).count(); // LUT计算时间
    //         total_adc_recall_us_s += chrono::duration_cast<chrono::microseconds>(t2_s_adc_recall_end_extract_start - t_l1_s_lut_end_adc_recall_start).count(); // ADC召回（不含LUT）
    //         total_extract_ids_us_s += chrono::duration_cast<chrono::microseconds>(t2_5_s_extract_end_rerank_serial_start - t2_s_adc_recall_end_extract_start).count();
    //         total_rerank_serial_us_s += chrono::duration_cast<chrono::microseconds>(t3_s_rerank_serial_end_sort_start - t2_5_s_extract_end_rerank_serial_start).count();
    //         total_rerank_sort_us_s += chrono::duration_cast<chrono::microseconds>(t4_s_sort_end_query_end - t3_s_rerank_serial_end_sort_start).count();
    //         total_overall_query_us_s += chrono::duration_cast<chrono::microseconds>(t4_s_sort_end_query_end - t_l0_s_overall_query_start).count(); // 总时间从LUT计算开始


    //         // --- D. 评估当前查询的召回率 (与之前逻辑相同) ---
    //         set<uint32_t> ground_truth_ids_ivfpq;
    //         for (size_t j = 0; j < k; ++j) // 使用 k (最终返回的top-k)
    //         {
    //             if ( ( (size_t)i * test_gt_d + j ) < ( test_number * test_gt_d ) ) { // 确保不越界
    //                  ground_truth_ids_ivfpq.insert(test_gt[j + i * test_gt_d]);
    //             }
    //         }
    //         size_t correct_hits_ivfpq = 0;
    //         for (size_t r_idx = 0; r_idx < k && r_idx < rerank_candidates_exact.size(); ++r_idx) // 使用 k
    //         {
    //             if (ground_truth_ids_ivfpq.count(rerank_candidates_exact[r_idx].second))
    //                 correct_hits_ivfpq++;
    //         }
    //         float recall_value_ivfpq;
    //         if (k == 0) recall_value_ivfpq = 1.0f;
    //         else if (ground_truth_ids_ivfpq.empty() && k > 0) recall_value_ivfpq = (correct_hits_ivfpq == 0) ? 1.0f : 0.0f;
    //         else recall_value_ivfpq = static_cast<float>(correct_hits_ivfpq) / k;


    //         results_ivfpq_pthread_rerank[i] = {recall_value_ivfpq, chrono::duration_cast<chrono::microseconds>(t4_s_sort_end_query_end - t_l0_s_overall_query_start).count()}; // 存储总延迟
    //     }
    //     cout << "所有IVF-PQ Pthread (ADC+串行Rerank) 查询执行完毕。" << endl;

    //     // --- E. 计算并输出平均性能指标 ---
    //     float total_recall_ivfpq_pthread_avg = 0.0f;
    //     for (size_t i = 0; i < test_number; ++i)
    //     {
    //         total_recall_ivfpq_pthread_avg += results_ivfpq_pthread_rerank[i].recall;
    //     }
    //     cout << "\n--- IVF-PQ Pthread (ADC+串行Rerank) 最终测试结果 ("
    //          << (ivf_search_index.use_memory_reordering_ ? "已内存重排" : "未内存重排") << ") ---" << endl;
    //     if (test_number > 0) {
    //         cout << "  平均召回率 Recall@" << k << ": " << total_recall_ivfpq_pthread_avg / test_number << endl;
    //         cout << "  平均查询总延迟 (us):               " << (double)total_overall_query_us_s / test_number << endl;
    //         cout << "  平均LUT预计算阶段延迟 (us):      " << (double)total_lut_computation_us_s / test_number << endl; // 新增
    //         cout << "  平均ADC召回阶段延迟 (us):          " << (double)total_adc_recall_us_s / test_number << endl;
    //         cout << "  平均提取ID阶段延迟 (us):           " << (double)total_extract_ids_us_s / test_number << endl;
    //         cout << "  平均串行重排计算阶段延迟 (us):   " << (double)total_rerank_serial_us_s / test_number << endl; // 修改名称
    //         cout << "  平均最终排序阶段延迟 (us):       " << (double)total_rerank_sort_us_s / test_number << endl;
    //     }
    //     cout << "------------------------------------" << endl;
    // }

    // --- 【新增】OpenMP 并行在线查询与评估阶段 ---
    if (run_openmp_test) // 确保这个布尔开关在 main 函数开始处已定义并按需设置
    {
        vector<SearchResult> results_ivf_openmp(test_number); // 存储OpenMP测试的结果
        size_t nprobe_param = 20;    
        int num_openmp_threads_to_set = 2;

        cout << "\n--- [执行] IVF OpenMP (簇内并行) 查询阶段 ("
             << (ivf_search_index.use_memory_reordering_ ? "使用内存重排数据" : "使用原始数据") << ") ---" << endl;
        cout << "  参数: 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param
             << ", k=" << k << ", OpenMP线程数=" << num_openmp_threads_to_set << endl;

        // 用于累加总延迟，以便计算平均值
        long long total_latency_openmp_us = 1;

        // 循环处理每条测试查询
        for (size_t i = 0; i < test_number; ++i)
        {
            const float* current_query_vector = test_query + i * vecdim; // 当前查询向量的指针

            // 记录单次查询开始时间
            auto query_start_time_openmp = chrono::high_resolution_clock::now();

            // --- 调用 IVFIndex 类的 OpenMP 版本的搜索方法 ---
            priority_queue<pair<float, uint32_t>> current_results_pq_openmp =
                ivf_search_index.search_openmp(current_query_vector,
                                               k,                      // 最终返回的top-k数量
                                               nprobe_param,
                                               num_openmp_threads_to_set);

            // 记录单次查询结束时间并计算延迟
            auto query_end_time_openmp = chrono::high_resolution_clock::now();
            long long query_latency_us_openmp = chrono::duration_cast<chrono::microseconds>(query_end_time_openmp - query_start_time_openmp).count();
            total_latency_openmp_us += query_latency_us_openmp; // 累加单次查询延迟

            // --- D. 评估当前查询的召回率 (与之前Pthread版本的评估逻辑相同) ---
            set<uint32_t> ground_truth_ids_openmp;
            for (size_t j = 0; j < k; ++j)
            {
                // 安全检查，确保不会越界访问 test_gt
                if (((size_t)i * test_gt_d + j) < (test_number * test_gt_d)) {
                     ground_truth_ids_openmp.insert(test_gt[j + i * test_gt_d]);
                }
            }

            size_t correct_hits_openmp = 0; // 记录命中标准答案的数量
            // 从优先队列中取出结果进行比较
            while (!current_results_pq_openmp.empty())
            {
                uint32_t result_id_openmp = current_results_pq_openmp.top().second; // 获取结果中的数据点ID
                if (ground_truth_ids_openmp.count(result_id_openmp)) // 如果找到的结果在标准答案中
                    correct_hits_openmp++; // 命中数量加1
                current_results_pq_openmp.pop(); // 移除已比较的元素
            }

            // 计算召回率
            float recall_value_openmp;
            if (k == 0) recall_value_openmp = 1.0f;
            else if (ground_truth_ids_openmp.empty() && k > 0) recall_value_openmp = (correct_hits_openmp == 0) ? 1.0f : 0.0f;
            else recall_value_openmp = static_cast<float>(correct_hits_openmp) / k;

            results_ivf_openmp[i] = {recall_value_openmp, query_latency_us_openmp}; // 存储召回率和延迟
        }
        cout << "所有IVF OpenMP查询执行完毕。" << endl;

        // --- E. 计算并输出平均性能指标 ---
        float total_recall_openmp_avg = 0.0f;
        // total_latency_openmp_us 已经在循环中累加了
        for (size_t i = 0; i < test_number; ++i) 
            total_recall_openmp_avg += results_ivf_openmp[i].recall;

        cout << "\n--- IVF OpenMP (簇内并行) 最终测试结果 ("
             << (ivf_search_index.use_memory_reordering_ ? "已内存重排" : "未内存重排") << ") ---" << endl;
        if (test_number > 0)
        {
            cout << "  平均召回率 Recall@" << k << ": " << total_recall_openmp_avg / test_number << endl;
            cout << "  平均查询延迟 (us): " << (double)total_latency_openmp_us / test_number << endl;
        } 
        else 
            cout << "  没有执行任何查询。" << endl;
        cout << "------------------------------------" << endl;
    }


    // --- F. 清理动态分配的内存 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    if (global_base_pq_codes) delete[] global_base_pq_codes; // <<< 新增：清理PQ编码数据

    return 0;
}