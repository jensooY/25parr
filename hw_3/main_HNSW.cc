#include <vector>
#include <cstring> // For memcpy in HNSW (if not included by hnswlib.h)
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <queue>     // For std::priority_queue used by HNSW searchKnn
#include <utility>   // For std::pair
#include <cstdint>   // For uint32_t, int64_t, labeltype
#include <algorithm> // For std::min, std::sort, std::reverse
#include <stdexcept> // For std::runtime_error

// --- HNSW 核心头文件 ---
#include "hnswlib/hnswlib/hnswlib.h"
using namespace std;

// --- 距离计算 (HNSW内部会使用其space对象中定义的距离) ---
// #include "simd_distance.h" // 这个文件主要是你自己实现的距离函数，HNSW会用它自己的

// 使用hnswlib命名空间，方便调用其类和函数
using namespace hnswlib;
// 也可以在需要的地方显式使用 hnswlib::

// --- 1. 数据加载函数 (保持不变) ---
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) { // 增加文件打开失败的检查
        std::cerr << "错误：无法打开数据文件 " << data_path << std::endl;
        exit(1);
    }
    // 假设n和d在文件中是4字节int (根据DEEP100K常见格式)
    int temp_n, temp_d;
    fin.read(reinterpret_cast<char*>(&temp_n), 4);
    fin.read(reinterpret_cast<char*>(&temp_d), 4);
    n = static_cast<size_t>(temp_n);
    d = static_cast<size_t>(temp_d);

    T* data = new T[n * d];
    size_t sz = sizeof(T); // 使用 size_t
    for(size_t i = 0; i < n; ++i){ // 使用 size_t
        fin.read(reinterpret_cast<char*>(data + i * d), (long long)d * sz); // read的第二个参数通常是 std::streamsize
    }
    fin.close();

    std::cerr<<"加载数据 "<<data_path<<"\n";
    std::cerr<<"维度: "<<d<<"  数量:"<<n<<"  每元素大小:"<<sizeof(T)<<"\n";
    return data;
}

//--- 2. 用于存储单次查询的结果：recall (召回率) 和 latency (延迟，单位微秒) ---
struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

//--- 3. HNSW 索引构建函数 (如果索引不存在或需要重建时调用) ---
// base: 原始浮点基础数据
// base_number: 基础数据点的数量
// vecdim: 数据的维度
// index_path_output: 构建完成后索引保存的路径
// space_ptr: 指向 SpaceInterface 的指针 (例如 InnerProductSpace 或 L2Space)
// M_construction: HNSW构建参数M
// efConstruction_param: HNSW构建参数efConstruction
void build_hnsw_index(const float* base, size_t base_number, size_t vecdim,
                      const std::string& index_path_output,
                      SpaceInterface<float>* space_ptr,
                      size_t M_construction = 16, size_t efConstruction_param = 150)
{
    std::cout << "\n--- 开始构建HNSW索引 ---" << std::endl;
    std::cout << "  M: " << M_construction << ", efConstruction: " << efConstruction_param << std::endl;
    std::cout << "  保存路径: " << index_path_output << std::endl;

    // 创建HNSW算法对象
    // HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
    HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float>(space_ptr, base_number, M_construction, efConstruction_param);

    // 逐个添加数据点到索引中
    // HNSW库内部可能会使用OpenMP并行化addPoint的部分过程，如果库本身支持且编译时开启了OpenMP
    // 这里我们也可以在外部对addPoint的调用进行并行化，如果addPoint本身是线程安全的
    // 根据你提供的代码，addPoint的调用是在一个 #pragma omp parallel for 循环中进行的，说明它应该是（或者期望是）线程安全的
    // 不过，为了简化初始运行，我们可以先串行添加，或者保留你之前的并行添加逻辑。
    // 为了安全和逐步调试，先尝试串行添加第一个点，然后并行添加剩余的点。
    std::cout << "  开始添加数据点到HNSW索引..." << std::endl;
    if (base_number > 0) {
        appr_alg->addPoint(base, (labeltype)0); // 添加第一个点，labeltype通常是size_t或unsigned
    }
    #pragma omp parallel for schedule(dynamic) // 使用动态调度，因为addPoint耗时可能不均
    for (size_t i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + i * vecdim, (labeltype)i);
    }
    std::cout << "  所有数据点添加完毕。" << std::endl;

    // 保存构建好的索引到文件
    appr_alg->saveIndex(index_path_output);
    std::cout << "HNSW索引已保存到: " << index_path_output << std::endl;

    delete appr_alg; // 清理HNSW对象
    std::cout << "--- HNSW索引构建完成 ---" << std::endl;
}


// --- main 函数入口 ---
int main(int argc, char *argv[])
{
    cout << "--- [main_hnsw.cc] HNSW 测试 ---" << endl;

    // --- A. 初始化和数据加载阶段 ---
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; // 你的数据集路径
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // --- B. 实验参数设置 ---
    test_number = 2000;     // 限制测试查询的数量
    const size_t k = 10;    // 每条查询返回top-k个结果

    std::vector<SearchResult> results_hnsw(test_number); // 存储HNSW测试的结果

    // --- C. HNSW 索引准备阶段 ---
    // 定义HNSW使用的距离空间 (确保与索引构建时一致)
    // 我们这里使用内积距离 InnerProductSpace
    SpaceInterface<float> *space = new InnerProductSpace(vecdim);
    // 或者L2距离: L2Space* space = new L2Space(vecdim);

    HierarchicalNSW<float> *appr_alg = nullptr; // HNSW算法对象指针
    std::string hnsw_index_path = "files/hnsw_ip_index.bin"; // 索引文件保存/加载路径 (建议根据距离类型命名)

    // 尝试加载已存在的索引，如果不存在或加载失败，则构建新的索引
    std::ifstream index_file_checker(hnsw_index_path, std::ios::binary);
    if (index_file_checker.good()) {
        index_file_checker.close();
        cout << "\n--- 尝试加载已存在的HNSW索引 ---" << endl;
        try {
            // HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements = 0)
            appr_alg = new HierarchicalNSW<float>(space, hnsw_index_path, false, base_number);
            cout << "HNSW索引加载成功: " << hnsw_index_path << endl;
        } catch (const std::runtime_error& e) {
            cerr << "错误: 加载HNSW索引失败: " << e.what() << ". 将尝试重新构建。" << endl;
            if (appr_alg) { delete appr_alg; appr_alg = nullptr; }
            // 如果加载失败，删除可能损坏的索引文件，然后构建
            std::remove(hnsw_index_path.c_str());
            build_hnsw_index(base, base_number, vecdim, hnsw_index_path, space);
            // 重新加载刚刚构建的索引
            try {
                appr_alg = new HierarchicalNSW<float>(space, hnsw_index_path, false, base_number);
                cout << "重新构建并加载HNSW索引成功。" << endl;
            } catch (const std::runtime_error& e2) {
                 cerr << "错误: 重新构建后加载HNSW索引仍然失败: " << e2.what() << endl;
                 delete space; delete[] test_query; delete[] test_gt; delete[] base; return 1;
            }
        }
    } else {
        index_file_checker.close();
        cout << "\n--- HNSW索引文件不存在，开始构建新索引 ---" << endl;
        build_hnsw_index(base, base_number, vecdim, hnsw_index_path, space);
        // 加载刚刚构建的索引
        try {
            appr_alg = new HierarchicalNSW<float>(space, hnsw_index_path, false, base_number);
            cout << "新构建并加载HNSW索引成功。" << endl;
        } catch (const std::runtime_error& e) {
            cerr << "错误: 新构建后加载HNSW索引失败: " << e.what() << endl;
            delete space; delete[] test_query; delete[] test_gt; delete[] base; return 1;
        }
    }

    // 检查appr_alg是否成功初始化
    if (!appr_alg) {
        cerr << "错误: HNSW算法对象未能成功初始化。" << endl;
        delete space; delete[] test_query; delete[] test_gt; delete[] base; return 1;
    }

    // 设置查询时的 ef 参数 (efSearch)
    // 这个参数对召回率和速度影响很大，是HNSW查询时的主要调优参数
    size_t efSearch = 100; // 【可调参数】例如，可以从 k (10) 开始，逐步增加到几百
    appr_alg->setEf(efSearch);
    cout << "HNSW efSearch (查询时beam宽度) 设置为: " << efSearch << endl;


    // --- D. 在线查询与评估阶段 ---
    cout << "\n--- HNSW 查询阶段 ---" << endl;
    cout << "  将执行查询数量: " << test_number << endl;
    cout << "  返回 Top-K: " << k << endl;

    long long total_latency_hnsw_us = 0; // 用于累加总延迟

    for (size_t i = 0; i < test_number; ++i)
    {
        const float* current_query_vector = test_query + i * vecdim;

        // 记录单次查询开始时间
        auto query_start_time = chrono::high_resolution_clock::now();

        // --- 调用 HNSW 的 searchKnn 方法进行搜索 ---
        // searchKnn(const void *query_data, size_t k_neighbors, BaseFilterFunctor* isIdAllowed = nullptr)
        // 返回的是 std::priority_queue<std::pair<dist_t, labeltype>>
        // labeltype 通常是 uint32_t 或 size_t，代表原始ID
        // priority_queue 默认是最大堆，hnswlib内部的CompareByFirst使其行为像最小堆（距离小的优先级高，top是距离最大的）
        // 但从hnswlib的searchKnn取出时，通常是按距离从小到大（如果用vector接收并反转）
        std::priority_queue<std::pair<float, labeltype>> result_pq_hnsw;
        try {
            result_pq_hnsw = appr_alg->searchKnn(current_query_vector, k);
        } catch (const std::runtime_error& e) {
            std::cerr << "查询 " << i << " 时HNSW搜索出错: " << e.what() << std::endl;
            results_hnsw[i] = {0.0f, 0}; // 记录错误或跳过
            continue;
        }


        // 记录单次查询结束时间并计算延迟
        auto query_end_time = chrono::high_resolution_clock::now();
        long long query_latency_us = chrono::duration_cast<chrono::microseconds>(query_end_time - query_start_time).count();
        total_latency_hnsw_us += query_latency_us;

        // --- 处理HNSW返回的结果并评估召回率 ---
        std::vector<uint32_t> found_ids; // 只存储ID用于评估
        found_ids.reserve(k);
        // 从优先队列中取出结果。std::priority_queue的top()是最大元素（基于CompareByFirst，距离大的在顶部）
        // 我们需要的是距离最小的k个。HNSW的searchKnn通常已经筛选好了。
        // 取出时是按距离从大到小，所以如果需要从小到大，需要反转。
        std::vector<std::pair<float, uint32_t>> temp_results;
        temp_results.reserve(result_pq_hnsw.size());
        while (!result_pq_hnsw.empty()) {
            temp_results.push_back({result_pq_hnsw.top().first, (uint32_t)result_pq_hnsw.top().second});
            result_pq_hnsw.pop();
        }
        std::reverse(temp_results.begin(), temp_results.end()); // 现在是按距离从小到大

        for (size_t r_idx = 0; r_idx < k && r_idx < temp_results.size(); ++r_idx) {
            found_ids.push_back(temp_results[r_idx].second);
        }


        set<uint32_t> ground_truth_ids;
        for (size_t j = 0; j < k; ++j)
        {
            if (((size_t)i * test_gt_d + j) < (test_number * test_gt_d)) {
                 ground_truth_ids.insert(test_gt[j + i * test_gt_d]);
            }
        }
        size_t correct_hits = 0;
        for (uint32_t found_id : found_ids)
        {
            if (ground_truth_ids.count(found_id))
            {
                correct_hits++;
            }
        }
        float recall_value;
        if (k == 0) recall_value = 1.0f;
        else if (ground_truth_ids.empty() && k > 0) recall_value = (correct_hits == 0) ? 1.0f : 0.0f;
        else recall_value = static_cast<float>(correct_hits) / k;

        results_hnsw[i] = {recall_value, query_latency_us};
    }
    cout << "所有HNSW查询执行完毕。" << endl;

    // --- E. 计算并输出平均性能指标 ---
    float total_recall_hnsw_avg = 0.0f;
    for (size_t i = 0; i < test_number; ++i)
    {
        total_recall_hnsw_avg += results_hnsw[i].recall;
    }
    cout << "\n--- HNSW 最终测试结果 ---" << endl;
    if (test_number > 0) {
        cout << "  参数: efSearch=" << efSearch << ", k=" << k << endl;
        cout << "  平均召回率 Recall@" << k << ": " << total_recall_hnsw_avg / test_number << endl;
        cout << "  平均查询延迟 (us): " << (double)total_latency_hnsw_us / test_number << endl;
    } else {
        cout << "  没有执行任何查询。" << endl;
    }
    cout << "  HNSW内部距离计算次数: " << (appr_alg ? appr_alg->metric_distance_computations.load() : 0) << endl;
    cout << "  HNSW内部跳数: " << (appr_alg ? appr_alg->metric_hops.load() : 0) << endl;
    cout << "------------------------------------" << endl;

    // --- F. 清理动态分配的内存 ---
    cout << "\n开始清理内存..." << endl;
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    if (appr_alg) delete appr_alg; // 清理HNSW对象
    if (space) delete space;       // 清理距离空间对象
    cout << "内存清理完毕。" << endl;

    return 0;
}