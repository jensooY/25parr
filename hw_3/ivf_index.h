#ifndef IVF_INDEX_H // 包含守卫
#define IVF_INDEX_H

#include <vector>
#include <queue>
#include <utility>       // std::pair
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <limits>        // 数值极限
#include "kmeans.h"      
#include "simd_distance.h" // 包含SIMD内积距离计算头文件
#include <pthread.h>
#include "pq_utils.h"
#include <omp.h>

using namespace std;
class IVFIndex; // 前向声明IVFIndex类，因为IVFThreadData中会用到它

// --- Pthread 工作线程所需的数据结构 (用于【单阶段】并行化簇内搜索) ---
// 这个结构体用于向每个Pthread工作线程传递其任务所需的信息 (用于 search_pthread)
struct IVFThreadData 
{ 
    const IVFIndex* ivf_index_ptr;       // 指向IVFIndex对象的指针，用于访问索引数据 (质心、倒排列表、基础数据)
    const float* query_vector_ptr;     // 指向当前查询向量的指针
    size_t k_neighbors;                // 需要查找的最近邻数量 (top-k)
    vector<uint32_t> clusters_to_search; // 该线程负责搜索的簇的ID列表
    priority_queue<pair<float, uint32_t>> local_top_k_results; // 该线程的局部top-k结果 (距离, 点ID)

    // 构造函数
    IVFThreadData(const IVFIndex* index, const float* query, size_t k)
        : ivf_index_ptr(index), query_vector_ptr(query), k_neighbors(k) {}
};

// --- 【新增】Pthread 工作线程所需的数据结构 (用于【两阶段并行】中的质心探查阶段) ---
// 这个结构体用于向每个Pthread工作线程传递其任务所需的信息 (用于 search_pthread_two_stage 的第一阶段)
struct IVFThreadData_CentroidProbe_TS 
{ // _TS 后缀表示 TwoStage，以区分
    const IVFIndex* ivf_index_ptr_ts;       // 指向IVFIndex对象的指针
    const float* query_vector_ptr_ts;    // 指向当前查询向量的指针
    size_t start_centroid_idx_ts;       // 该线程负责的起始质心索引
    size_t end_centroid_idx_ts;         // 该线程负责的结束质心索引 (不包含)
    // priority_queue或者vector都可以用来存储局部的(距离, 质心ID)对
    vector<pair<float, uint32_t>> local_centroid_distances_ts; // 该线程计算出的 (距离, 质心ID) 对

    // 构造函数
    IVFThreadData_CentroidProbe_TS(const IVFIndex* index, const float* query) // k_neighbors 和 clusters_to_search 在此阶段不需要
        : ivf_index_ptr_ts(index), query_vector_ptr_ts(query), start_centroid_idx_ts(0), end_centroid_idx_ts(0) {}
};

// --- 【新增】Pthread 工作线程所需的数据结构 (用于【两阶段并行】中的簇内搜索阶段) ---
// 这个结构体与原有的IVFThreadData结构非常相似，但为了清晰区分，独立定义
// 这里我们创建一个新的，以明确其用于两阶段并行中的第二阶段
struct IVFThreadData_ClusterSearch_TS { // _TS 后缀表示 TwoStage
    const IVFIndex* ivf_index_ptr_ts;
    const float* query_vector_ptr_ts;
    size_t k_neighbors_ts;
    vector<uint32_t> clusters_to_search_ts;
    priority_queue<pair<float, uint32_t>> local_top_k_results_ts;

    // 构造函数
    IVFThreadData_ClusterSearch_TS(const IVFIndex* index, const float* query, size_t k)
        : ivf_index_ptr_ts(index), query_vector_ptr_ts(query), k_neighbors_ts(k) {}
};

// --- Pthread 工作线程所需的数据结构 (用于 search_pthread_ivfpq_adc) ---
struct IVFThreadData_PQ {
    const IVFIndex* ivf_index_ptr;
    const float* query_vector_ptr;
    size_t nk_pq_candidates; // 目标是召回Nk个
    vector<uint32_t> clusters_to_search;
    priority_queue<pair<float, uint32_t>> local_top_nk_results_adc; // 存储ADC距离和原始ID
    float* lut_for_thread; // 每个线程可以有自己的LUT副本，或者共享一个只读的

    IVFThreadData_PQ(const IVFIndex* index, const float* query, size_t nk_pq, float* lut)
        : ivf_index_ptr(index), query_vector_ptr(query), nk_pq_candidates(nk_pq), lut_for_thread(lut) {}
};


// IVF索引类定义
class IVFIndex 
{
public:
    // --- 构造函数 (增加一个参数控制是否启用内存重排) ---
    // --- 构造函数 (增加PQ相关指针) ---
    IVFIndex(const float* base_data_ptr,         // 原始浮点基础数据
             const uint8_t* base_pq_codes_ptr, // 【新增】PQ编码后的基础数据 (可选, nullptr表示不使用PQ)
             const PQCodebook* pq_codebook_ptr, // 【新增】PQ码本指针 (可选, nullptr表示不使用PQ)
             size_t num_base_points,
             size_t dimension,
             bool enable_reordering = false)
    : base_data_original_(base_data_ptr),// 【修改】存储原始数据指针
      base_pq_codes_global_(base_pq_codes_ptr), // 【新增】存储全局PQ编码指针
      pq_codebook_global_(pq_codebook_ptr),   // 【新增】存储全局PQ码本指针
      num_base_(num_base_points),
      dim_(dimension),
      index_built_(false),
      num_clusters_in_index_(0),
      use_memory_reordering_(enable_reordering),  // 【新增】初始化内存重排标志
      base_data_reordered_(nullptr) {}           // 【新增】初始化重排数据指针为空


    // --- 析构函数 (需要处理重排数据的释放) ---
    ~IVFIndex()
    { // 【修改】
        if (use_memory_reordering_ && base_data_reordered_) { // 【修改】仅当启用重排且已分配时才释放
            delete[] base_data_reordered_; // 释放它
            base_data_reordered_ = nullptr;
        }
    }

    // 构建IVF索引
    inline bool build(size_t num_clusters, int kmeans_max_iterations);

    // 执行单线程的IVF搜索 
    inline priority_queue<pair<float, uint32_t>> search_single_thread
        (const float* query_vector,size_t k,size_t nprobe);


    // --- Pthread 并行化IVF搜索 (【单阶段】并行版本) ---
    inline priority_queue<pair<float, uint32_t>> search_pthread
        (const float* query_vector,
            size_t k,
            size_t nprobe,
            int num_threads_to_use); 

    // 【新增】Pthread 单阶段并行IVF-PQ搜索 (簇内用ADC，可选重排)
    inline priority_queue<pair<float, uint32_t>> search_pthread_ivfpq_adc( // <<< 【修改这里】
        const float* query_vector,
        size_t nk_pq,
        size_t nprobe,
        int num_threads_to_use,
        const float* precomputed_lut // <<< 【添加这个参数】
    );


    // --- 【新增】Pthread 两阶段并行化IVF搜索 ---
    // query_vector: 查询向量
    // k: top-k
    // nprobe: 探查的簇数量
    // num_threads_for_centroids: 【新增】用于并行计算与质心距离的工作线程数量
    // num_threads_for_clusters: 【新增】用于并行在候选簇内搜索数据点的工作线程数量
    // 返回值: 全局的top-k结果优先队列
    inline priority_queue<pair<float, uint32_t>> search_pthread_two_stage
        (const float* query_vector,
            size_t k,
            size_t nprobe,
            int num_threads_for_centroids, // 新参数
            int num_threads_for_clusters);  // 新参数

    
    // --- 【新增】OpenMP 并行化IVF搜索 (仅并行化簇内搜索) ---
    // query_vector: 查询向量
    // k: top-k
    // nprobe: 探查的簇数量
    // num_threads: OpenMP使用的线程数量
    // 返回值: 全局的top-k结果优先队列
    inline priority_queue<pair<float, uint32_t>> search_openmp(
        const float* query_vector,
        size_t k,
        size_t nprobe,
        int num_threads // OpenMP将使用的线程数
    );

public:
    const float* base_data_original_; // 【修改】指向【原始】N*D基础数据的指针 (不拥有所有权)
    float* base_data_reordered_;      // 【新增】指向【重排后】N*D基础数据的指针 (如果启用重排，此类拥有此内存)
    bool use_memory_reordering_;      // 【新增】标记是否启用了内存重排

    // 【新增】辅助函数，用于获取当前应使用的数据源指针 (原始或重排后)
    inline const float* getCurrentBaseDataPtr() const {
        return use_memory_reordering_ ? base_data_reordered_ : base_data_original_;
    }

    // 【新增】用于在评估时，将重排后的索引映射回原始ID的查找表
    // 只有在 use_memory_reordering_ 为 true 时才会被填充和使用
    vector<uint32_t> reordered_idx_to_original_id_;

        // --- 新增PQ相关成员 ---
    const uint8_t* base_pq_codes_global_; // 指向全局PQ编码数据 (不拥有所有权)
    const PQCodebook* pq_codebook_global_; // 指向全局PQ码本 (不拥有所有权)

    // --- 以下成员变量与之前一致 ---
    size_t num_base_;
    size_t dim_;
    bool index_built_;
    size_t num_clusters_in_index_;
    vector<float> centroids_;
    // 【重要修改】inverted_lists_ 的含义：
    // 如果 use_memory_reordering_ == false, 它存储的是【原始数据点的ID】。
    // 如果 use_memory_reordering_ == true, 它存储的是数据点在【base_data_reordered_ 中的索引 (0 to num_base-1)】。
    vector<vector<uint32_t>> inverted_lists_;
};

// 构建IVF索引的函数实现(离线) 
inline bool IVFIndex::build(size_t num_clusters, int kmeans_max_iterations)
{
    num_clusters_in_index_ = num_clusters;

    // 1. 执行KMeans聚类，得到初始的质心和包含【原始ID】的倒排列表
    vector<vector<uint32_t>> initial_inverted_lists(num_clusters_in_index_); // 临时存储KMeans结果的倒排列表（含原始ID）
    // KMeans总是在原始数据上进行，以得到最准确的簇划分
    bool kmeans_success = run_kmeans(base_data_original_, num_base_, dim_,
                                     num_clusters_in_index_, kmeans_max_iterations,
                                     centroids_, initial_inverted_lists); // 输出到临时的倒排列表

    if (!kmeans_success)
    {
        cerr << "IVFIndex::build: KMeans聚类失败。" << endl;
        index_built_ = false;
        return false;
    }

    // 2. 【新增】根据 use_memory_reordering_ 标志决定是否执行内存重排
    if (use_memory_reordering_)
    {
        cout << "  IVFIndex::build: 正在执行内存重排..." << endl;
        // 如果之前分配过重排内存（例如build被多次调用），先释放
        if (base_data_reordered_ != nullptr)
        {
            delete[] base_data_reordered_;
            base_data_reordered_ = nullptr;
        }
        base_data_reordered_ = new float[num_base_ * dim_];   // 为重排后的数据分配内存
        reordered_idx_to_original_id_.resize(num_base_);      // 调整映射表大小
        inverted_lists_.assign(num_clusters_in_index_, vector<uint32_t>()); // 清空并重置主倒排列表，准备存【新索引】

        uint32_t current_reordered_idx = 0; // 当前在 base_data_reordered_ 中的写入索引 (0 to num_base_-1)

        // 遍历KMeans聚类后的每个簇 (initial_inverted_lists 包含的是原始ID)
        for (size_t c = 0; c < num_clusters_in_index_; ++c)
        {
            // 遍历该簇中的每一个【原始数据点ID】
            for (uint32_t original_point_id : initial_inverted_lists[c])
            {
                if (original_point_id < num_base_) // 安全检查
                {
                    // 从原始数据中复制向量到重排后的数据区域
                    const float* src_vector_ptr = base_data_original_ + original_point_id * dim_;
                    float* dest_vector_ptr = base_data_reordered_ + current_reordered_idx * dim_;
                    std::copy(src_vector_ptr, src_vector_ptr + dim_, dest_vector_ptr);

                    // 更新映射表：记录当前【重排后索引】对应的【原始ID】
                    reordered_idx_to_original_id_[current_reordered_idx] = original_point_id;

                    // 更新主倒排列表 inverted_lists_：现在它存储的是点在 base_data_reordered_ 中的【新索引】
                    inverted_lists_[c].push_back(current_reordered_idx);

                    current_reordered_idx++; // 移动到下一个重排数据的写入位置
                }
            }
        }
        cout << "  IVFIndex::build: 内存重排完成。总共重排点数: " << current_reordered_idx << endl; // 调试信息
    }
    else // 如果不使用内存重排
    {
        // 直接使用KMeans基于原始ID生成的倒排列表
        inverted_lists_ = initial_inverted_lists;
        // base_data_reordered_ 保持为 nullptr
        // reordered_idx_to_original_id_ 保持为空或未初始化大小
    }

    index_built_ = true; // 标记索引（可能包含重排）构建完成
    return index_built_;
}


// 单线程IVF搜索函数的实现（在线查）
inline priority_queue<pair<float, uint32_t>> IVFIndex::search_single_thread
    (const float* query_vector,
        size_t k,
        size_t nprobe)
{
    priority_queue<pair<float, uint32_t>> top_k_results;

    // --- 【修改】获取当前应使用的数据源指针 ---
    const float* current_base_to_search = getCurrentBaseDataPtr();

    size_t actual_nprobe = nprobe; 
    // 计算与质心的距离
    vector<pair<float, uint32_t>> cluster_distances;
    cluster_distances.reserve(num_clusters_in_index_);

    for (size_t i = 0; i < num_clusters_in_index_; ++i)
    {
        const float* centroid_vec = centroids_.data() + i * dim_;
        float dist_to_centroid = compute_inner_product_distance_neon_optimized(query_vector, centroid_vec, dim_);
        cluster_distances.push_back({dist_to_centroid, (uint32_t)i});
    }
    // 排序选出候选簇
    if (actual_nprobe < cluster_distances.size())
    {
        partial_sort(cluster_distances.begin(),
                     cluster_distances.begin() + actual_nprobe,
                     cluster_distances.end());
    }
    else
        sort(cluster_distances.begin(), cluster_distances.end());

    // 在选定的nprobe个候选簇内进行搜索
    for (size_t i = 0; i < actual_nprobe && i < cluster_distances.size(); ++i)
    {
        uint32_t cluster_id = cluster_distances[i].second;
        if (cluster_id >= inverted_lists_.size()) continue;
        // 【注意】这里的 point_indices_in_cluster 现在包含的是【新索引】(如果重排) 或 【原始ID】(如果未重排)
        const vector<uint32_t>& point_indices_in_cluster = inverted_lists_[cluster_id];

        for (uint32_t point_index_in_current_base : point_indices_in_cluster) // 变量名更清晰
        {
            // 使用当前索引从正确的数据源获取数据点向量
            if (point_index_in_current_base >= num_base_) continue; // 索引不应超过总点数
            const float* data_point_vec = current_base_to_search + point_index_in_current_base * dim_;

            float dist_to_point = compute_inner_product_distance_neon_optimized(query_vector, data_point_vec, dim_);

            // --- 【修改】在返回结果时，需要确保返回的是【原始ID】---
            uint32_t original_id_to_return = point_index_in_current_base; // 默认情况下，如果未重排，它就是原始ID
            if (use_memory_reordering_) 
            { // 如果使用了内存重排
                // point_index_in_current_base 是在 reordered_base 中的索引
                if (point_index_in_current_base < reordered_idx_to_original_id_.size())// 安全检查
                    original_id_to_return = reordered_idx_to_original_id_[point_index_in_current_base]; // 通过映射表找回原始ID
                else 
                    // cerr << "错误: search_single_thread - 重排索引越界 " << point_index_in_current_base << endl; // 调试
                    continue; // 跳过这个无效的索引
            }

            // 更新top-k结果队列 (与之前一致，只是存入的是 original_id_to_return)
            if (top_k_results.size() < k)
                top_k_results.push({dist_to_point, original_id_to_return});
            else if (dist_to_point < top_k_results.top().first)
            {
                top_k_results.pop();
                top_k_results.push({dist_to_point, original_id_to_return});
            }
        }
    }
    return top_k_results;
}

// --- Pthread 工作线程的执行函数 (用于【单阶段】并行 search_pthread) ---
inline void* ivf_pthread_worker_function(void* arg)
{
    // 1. 将void*参数转换回 IVFThreadData* 类型
    IVFThreadData* thread_data = static_cast<IVFThreadData*>(arg);

    // 从线程数据中获取必要信息
    const IVFIndex* index = thread_data->ivf_index_ptr;
    const float* query_vec = thread_data->query_vector_ptr;
    size_t k_val = thread_data->k_neighbors; // 使用新变量名以示区分

    // --- 【修改】获取当前应使用的数据源指针 ---
    const float* current_base_for_thread = index->getCurrentBaseDataPtr();

    // 2. 遍历分配给该线程的簇 (clusters_to_search)
    for (uint32_t cluster_id : thread_data->clusters_to_search)
    {
        // 【注意】这里的 point_indices_in_cluster 现在包含的是【新索引】(如果重排) 或 【原始ID】(如果未重排)
        const vector<uint32_t>& point_indices_in_cluster = index->inverted_lists_[cluster_id];

        // 3. 遍历簇内的每个数据点
        for (uint32_t point_index_in_current_base : point_indices_in_cluster) // 变量名更清晰
        {
            // --- 【修改】从正确的数据源获取数据点向量 ---
            const float* data_point_vec = current_base_for_thread + point_index_in_current_base * index->dim_;

            // 4. 计算SIMD精确内积距离 (与之前一致)
            float dist_to_point = compute_inner_product_distance_neon_optimized(query_vec, data_point_vec, index->dim_);

            // 5. 更新该线程的局部top-k优先队列
            // --- 【修正】在存入结果时，需要确保存入的是【原始ID】---
            uint32_t original_id_to_return_thread = point_index_in_current_base; // 默认是当前索引/ID
            if (index->use_memory_reordering_) { // 【新增判断和映射】如果使用了内存重排
                 if (point_index_in_current_base < index->reordered_idx_to_original_id_.size()) { // 安全检查
                    original_id_to_return_thread = index->reordered_idx_to_original_id_[point_index_in_current_base]; // 通过映射表找回原始ID
                } else {
                    // cerr << "错误: ivf_pthread_worker_function - 重排索引越界 " << point_index_in_current_base << endl; // 调试
                    continue; // 跳过这个无效的索引
                }
            }

            if (thread_data->local_top_k_results.size() < k_val)
                thread_data->local_top_k_results.push({dist_to_point, original_id_to_return_thread}); // 使用 original_id_to_return_thread
            else if (dist_to_point < thread_data->local_top_k_results.top().first)
            {
                thread_data->local_top_k_results.pop();
                thread_data->local_top_k_results.push({dist_to_point, original_id_to_return_thread}); // 使用 original_id_to_return_thread
            }
        }
    }


    // 6. 线程执行完毕
    pthread_exit(nullptr);
}

// --- Pthread 并行搜索方法的实现 (【单阶段】并行版本) ---
inline priority_queue<pair<float, uint32_t>> IVFIndex::search_pthread(
    const float* query_vector,
    size_t k,
    size_t nprobe,
    int num_threads_to_use // 使用的工作线程数量
) 
{
    priority_queue<pair<float, uint32_t>> final_top_k_results; // 用于存储最终合并的结果

    // 1. 定位候选簇 (与单线程版本一致，由主线程完成)
    size_t actual_nprobe = nprobe;

    vector<pair<float, uint32_t>> cluster_distances_to_query;
    cluster_distances_to_query.reserve(num_clusters_in_index_);
    for (size_t c = 0; c < num_clusters_in_index_; ++c)
    {
        const float* centroid_vec = centroids_.data() + c * dim_;
        float dist = compute_inner_product_distance_neon_optimized(query_vector, centroid_vec, dim_);
        cluster_distances_to_query.push_back({dist, (uint32_t)c});
    }

    if (actual_nprobe < cluster_distances_to_query.size())
    {
        partial_sort(cluster_distances_to_query.begin(),
                     cluster_distances_to_query.begin() + actual_nprobe,
                     cluster_distances_to_query.end());
    }
    else
        sort(cluster_distances_to_query.begin(), cluster_distances_to_query.end());

    vector<uint32_t> probed_cluster_ids;
    probed_cluster_ids.reserve(actual_nprobe);
    for (size_t i = 0; i < actual_nprobe && i < cluster_distances_to_query.size(); ++i)
        probed_cluster_ids.push_back(cluster_distances_to_query[i].second);

    if (probed_cluster_ids.empty())
        return final_top_k_results;

    // 2. 准备Pthread相关变量
    int actual_num_threads = std::min((int)probed_cluster_ids.size(), num_threads_to_use);
    if (actual_num_threads <= 0) actual_num_threads = 1;

    vector<pthread_t> threads(actual_num_threads);
    vector<IVFThreadData> thread_data_array; 
    thread_data_array.reserve(actual_num_threads);

    // 3. 分配任务给线程
    for (int i = 0; i < actual_num_threads; ++i)
        thread_data_array.emplace_back(this, query_vector, k);

    for (size_t i = 0; i < probed_cluster_ids.size(); ++i)
        thread_data_array[i % actual_num_threads].clusters_to_search.push_back(probed_cluster_ids[i]);

    // 4. 创建并启动工作线程
    for (int i = 0; i < actual_num_threads; ++i)
    {  
        if (!thread_data_array[i].clusters_to_search.empty()) 
        { 
            int rc = pthread_create(&threads[i], nullptr, ivf_pthread_worker_function, &thread_data_array[i]);
            if (rc) { /* cerr << "错误: search_pthread无法创建线程 " << i << endl; */ } 
        }
    }


    // 5. 等待所有工作线程执行完毕
    for (int i = 0; i < actual_num_threads; ++i)
    {  
        if (!thread_data_array[i].clusters_to_search.empty()) 
            pthread_join(threads[i], nullptr);
    }

    // 6. 合并所有线程的局部top-k结果
    for (int i = 0; i < actual_num_threads; ++i)
    {
        if (!thread_data_array[i].clusters_to_search.empty())
         { 
            priority_queue<pair<float, uint32_t>>& local_pq = thread_data_array[i].local_top_k_results;
            while (!local_pq.empty())
            {
                pair<float, uint32_t> candidate = local_pq.top();
                local_pq.pop();

                if (final_top_k_results.size() < k)
                    final_top_k_results.push(candidate);
                else if (candidate.first < final_top_k_results.top().first)
                {
                    final_top_k_results.pop();
                    final_top_k_results.push(candidate);
                }
            }
        }
    }
    return final_top_k_results;
}


// --- 【新增】Pthread 工作线程的执行函数 (用于【两阶段并行】的【质心探查阶段】) ---
// arg: 指向 IVFThreadData_CentroidProbe_TS 结构体的指针
inline void* ivf_pthread_worker_centroid_probe_ts(void* arg) 
{ // _ts 后缀
    // 1. 将void*参数转换回 IVFThreadData_CentroidProbe_TS* 类型
    IVFThreadData_CentroidProbe_TS* thread_data = static_cast<IVFThreadData_CentroidProbe_TS*>(arg);

    // 从线程数据中获取必要信息
    const IVFIndex* index = thread_data->ivf_index_ptr_ts;
    const float* query_vec = thread_data->query_vector_ptr_ts;

    // 预分配空间以存储该线程计算的(距离, 质心ID)对
    thread_data->local_centroid_distances_ts.reserve(thread_data->end_centroid_idx_ts - thread_data->start_centroid_idx_ts);

    // 2. 遍历分配给该线程的质心范围 (从 start_centroid_idx_ts 到 end_centroid_idx_ts-1)
    for (size_t c = thread_data->start_centroid_idx_ts; c < thread_data->end_centroid_idx_ts; ++c) {
        if (c >= index->num_clusters_in_index_) break; // 安全检查，防止越界
        const float* centroid_vec = index->centroids_.data() + c * index->dim_; // 获取当前质心向量指针
        // 3. 计算查询向量与当前质心的SIMD精确内积距离
        float dist = compute_inner_product_distance_neon_optimized(query_vec, centroid_vec, index->dim_);
        // 4. 将计算得到的 (距离, 质心ID) 对存入该线程的局部结果中
        thread_data->local_centroid_distances_ts.push_back({dist, (uint32_t)c});
    }

    // 5. 线程执行完毕，退出。局部结果存储在 thread_data->local_centroid_distances_ts 中。
    pthread_exit(nullptr);
}

// --- 【新增】Pthread 工作线程的执行函数 (用于【两阶段并行】的【簇内搜索阶段】) ---
// arg: 指向 IVFThreadData_ClusterSearch_TS 结构体的指针
inline void* ivf_pthread_worker_cluster_search_ts(void* arg) { // _ts 后缀
    // 1. 将void*参数转换回 IVFThreadData_ClusterSearch_TS* 类型
    IVFThreadData_ClusterSearch_TS* thread_data = static_cast<IVFThreadData_ClusterSearch_TS*>(arg);

    // 从线程数据中获取必要信息
    const IVFIndex* index = thread_data->ivf_index_ptr_ts;
    const float* query_vec = thread_data->query_vector_ptr_ts;
    size_t k = thread_data->k_neighbors_ts;

    // --- 【修改】获取当前应使用的数据源指针 ---
    const float* current_base_for_thread = index->getCurrentBaseDataPtr();

    // 2. 遍历分配给该线程的簇 (clusters_to_search_ts)
    for (uint32_t cluster_id : thread_data->clusters_to_search_ts) {
        if (cluster_id >= index->inverted_lists_.size()) continue; // 安全检查
        const vector<uint32_t>& point_ids_in_cluster = index->inverted_lists_[cluster_id];

        // 3. 遍历簇内的每个数据点
        for (uint32_t point_id : point_ids_in_cluster) {
            if (point_id >= index->num_base_) continue; // 安全检查
            const float* data_point_vec = current_base_for_thread + point_id * index->dim_;
            // 4. 计算SIMD精确内积距离
            float dist_to_point = compute_inner_product_distance_neon_optimized(query_vec, data_point_vec, index->dim_);
            // 5. 更新该线程的局部top-k优先队列
            if (thread_data->local_top_k_results_ts.size() < k) {
                thread_data->local_top_k_results_ts.push({dist_to_point, point_id});
            } else if (dist_to_point < thread_data->local_top_k_results_ts.top().first) {
                thread_data->local_top_k_results_ts.pop();
                thread_data->local_top_k_results_ts.push({dist_to_point, point_id});
            }
        }
    }
    // 6. 线程执行完毕
    pthread_exit(nullptr);
}


// --- 【新增】Pthread 两阶段并行搜索方法的实现 ---
inline priority_queue<pair<float, uint32_t>> IVFIndex::search_pthread_two_stage(
    const float* query_vector,
    size_t k,
    size_t nprobe,
    int num_threads_for_centroids, // 用于并行计算质心距离的线程数
    int num_threads_for_clusters   // 用于并行搜索簇内部的线程数
) {
    priority_queue<pair<float, uint32_t>> final_top_k_results; // 用于存储最终合并的结果

    // 检查索引是否已构建或是否有簇可供搜索 (与原search_pthread一致)
    if (!index_built_ || num_clusters_in_index_ == 0)
    {
        return final_top_k_results;
    }

    // --- 阶段 1: 并行计算查询向量与所有质心的距离 ---
    vector<pair<float, uint32_t>> all_centroid_distances_ts; // 用于存储所有(距离, 质心ID)对
    all_centroid_distances_ts.reserve(num_clusters_in_index_); // 预分配空间

    // 确定用于计算质心距离的实际线程数
    int actual_threads_for_centroids = std::min(num_threads_for_centroids, (int)num_clusters_in_index_);
    if (actual_threads_for_centroids <= 0) actual_threads_for_centroids = 1;

    vector<pthread_t> centroid_probe_threads_ts(actual_threads_for_centroids); // Pthread线程标识符数组
    vector<IVFThreadData_CentroidProbe_TS> centroid_thread_data_array_ts;  // 存储每个线程的数据
    centroid_thread_data_array_ts.reserve(actual_threads_for_centroids);

    // 计算每个质心探查线程大致需要处理的质心数量
    size_t centroids_per_probe_thread_ts = (num_clusters_in_index_ + actual_threads_for_centroids - 1) / actual_threads_for_centroids;

    // 为质心探查阶段的每个线程准备数据并创建线程
    for (int i = 0; i < actual_threads_for_centroids; ++i)
    {
        size_t start_idx_ts = i * centroids_per_probe_thread_ts;
        size_t end_idx_ts = std::min((i + 1) * centroids_per_probe_thread_ts, num_clusters_in_index_);

        if (start_idx_ts >= end_idx_ts) continue;

        // 创建线程数据对象 (IVFThreadData_CentroidProbe_TS)
        centroid_thread_data_array_ts.emplace_back(this, query_vector); // 构造函数只需要index和query
        IVFThreadData_CentroidProbe_TS& current_thread_data_ts = centroid_thread_data_array_ts.back();
        current_thread_data_ts.start_centroid_idx_ts = start_idx_ts;
        current_thread_data_ts.end_centroid_idx_ts = end_idx_ts;

        // 创建Pthread线程 (质心探查)
        int rc_centroid_ts = pthread_create(&centroid_probe_threads_ts[i], nullptr, ivf_pthread_worker_centroid_probe_ts, &current_thread_data_ts );
        if (rc_centroid_ts)
        {
            // cerr << "错误: search_pthread_two_stage 无法为质心探查创建线程 " << i << endl; // 调试信息
        }
    }

    // 等待所有质心探查线程执行完毕，并合并它们的局部结果
    for (int i = 0; i < actual_threads_for_centroids; ++i)
    {
        if (i * centroids_per_probe_thread_ts < num_clusters_in_index_ ) // 确保线程被创建
        {
            if (centroid_probe_threads_ts[i] != 0) { // 更可靠的检查是基于线程是否真的被创建
                 pthread_join(centroid_probe_threads_ts[i], nullptr);
                 // 合并该线程计算的质心距离
                 all_centroid_distances_ts.insert(all_centroid_distances_ts.end(),
                                              centroid_thread_data_array_ts[i].local_centroid_distances_ts.begin(),
                                              centroid_thread_data_array_ts[i].local_centroid_distances_ts.end());
            }
        }
    }

    // --- 从所有计算出的质心距离中，选出全局的 top-nprobe 个候选簇 ---
    size_t actual_nprobe_ts = nprobe; // 使用函数传入的nprobe作为目标
    if (actual_nprobe_ts == 0 || actual_nprobe_ts > all_centroid_distances_ts.size()) // 【与你原search_pthread一致的nprobe处理逻辑】
    {
        actual_nprobe_ts = std::min(all_centroid_distances_ts.size(), (size_t)1);
        if (actual_nprobe_ts == 0 && !all_centroid_distances_ts.empty()) actual_nprobe_ts = 1;
        else if (all_centroid_distances_ts.empty()) return final_top_k_results;
    }


    if (actual_nprobe_ts < all_centroid_distances_ts.size())
    {
        partial_sort(all_centroid_distances_ts.begin(),
                     all_centroid_distances_ts.begin() + actual_nprobe_ts,
                     all_centroid_distances_ts.end());
    }
    else
    {
        sort(all_centroid_distances_ts.begin(), all_centroid_distances_ts.end());
    }

    vector<uint32_t> probed_cluster_ids_ts; // 为两阶段并行版本使用独立变量名
    probed_cluster_ids_ts.reserve(actual_nprobe_ts);
    for (size_t i = 0; i < actual_nprobe_ts && i < all_centroid_distances_ts.size(); ++i)
    {
        probed_cluster_ids_ts.push_back(all_centroid_distances_ts[i].second);
    }

    if (probed_cluster_ids_ts.empty())
    {
        return final_top_k_results;
    }

    // --- 阶段 2: 并行在选定的 probed_cluster_ids_ts (候选簇) 内搜索数据点 ---
    int actual_threads_for_clusters_ts = std::min((int)probed_cluster_ids_ts.size(), num_threads_for_clusters); // 【与你原search_pthread一致的线程数确定逻辑】
    if (actual_threads_for_clusters_ts <= 0) actual_threads_for_clusters_ts = 1;

    vector<pthread_t> cluster_search_threads_ts(actual_threads_for_clusters_ts);
    vector<IVFThreadData_ClusterSearch_TS> cluster_search_thread_data_array_ts; // 【使用新的TS后缀的线程数据结构】
    cluster_search_thread_data_array_ts.reserve(actual_threads_for_clusters_ts);

    for (int i = 0; i < actual_threads_for_clusters_ts; ++i)
    {
        cluster_search_thread_data_array_ts.emplace_back(this, query_vector, k);
    }
    for (size_t i = 0; i < probed_cluster_ids_ts.size(); ++i)
    {
        cluster_search_thread_data_array_ts[i % actual_threads_for_clusters_ts].clusters_to_search_ts.push_back(probed_cluster_ids_ts[i]);
    }

    for (int i = 0; i < actual_threads_for_clusters_ts; ++i)
    {
        if (!cluster_search_thread_data_array_ts[i].clusters_to_search_ts.empty()) // 【与你原search_pthread一致的检查】
        {
            // 创建Pthread线程 (簇内搜索)
            int rc_cluster_ts = pthread_create(&cluster_search_threads_ts[i], nullptr, ivf_pthread_worker_cluster_search_ts, &cluster_search_thread_data_array_ts[i]);
            if (rc_cluster_ts)
            {
                // cerr << "错误: search_pthread_two_stage 无法为簇内搜索创建线程 " << i << endl; // 调试信息
            }
        }
    }

    for (int i = 0; i < actual_threads_for_clusters_ts; ++i)
    {
        if (!cluster_search_thread_data_array_ts[i].clusters_to_search_ts.empty()) // 【与你原search_pthread一致的检查】
        {
             if (cluster_search_threads_ts[i] != 0) { // 确保线程ID有效再join
                pthread_join(cluster_search_threads_ts[i], nullptr);
             }
        }
    }

    // 合并所有簇内搜索线程的局部top-k结果
    for (int i = 0; i < actual_threads_for_clusters_ts; ++i)
    {
        if (!cluster_search_thread_data_array_ts[i].clusters_to_search_ts.empty()) // 【与你原search_pthread一致的检查】
        {
            priority_queue<pair<float, uint32_t>>& local_pq_ts = cluster_search_thread_data_array_ts[i].local_top_k_results_ts;
            while (!local_pq_ts.empty())
            {
                pair<float, uint32_t> candidate_ts = local_pq_ts.top();
                local_pq_ts.pop();

                if (final_top_k_results.size() < k)
                {
                    final_top_k_results.push(candidate_ts);
                }
                else if (candidate_ts.first < final_top_k_results.top().first)
                {
                    final_top_k_results.pop();
                    final_top_k_results.push(candidate_ts);
                }
            }
        }
    }

    return final_top_k_results;
}


// --- 【新增】Pthread 工作线程的执行函数 (用于 search_pthread_ivfpq_adc) ---
// arg: 指向 IVFThreadData_PQ 结构体的指针
inline void* ivf_pthread_worker_ivfpq_adc(void* arg) {
    // 1. 将void*参数转换回 IVFThreadData_PQ* 类型
    IVFThreadData_PQ* thread_data = static_cast<IVFThreadData_PQ*>(arg);

    // 从线程数据中获取必要信息
    const IVFIndex* index = thread_data->ivf_index_ptr;
    // const float* query_vec = thread_data->query_vector_ptr; // 查询向量已用于计算LUT，这里不需要再用
    size_t nk_pq = thread_data->nk_pq_candidates;
    const float* lut_for_this_thread = thread_data->lut_for_thread; // 获取传递过来的LUT指针

    // 2. 遍历分配给该线程的簇 (clusters_to_search)
    for (uint32_t cluster_id : thread_data->clusters_to_search) {
        if (cluster_id >= index->inverted_lists_.size()) continue; // 安全检查
        // inverted_lists_ 包含的是【新索引】(如果启用了内存重排) 或 【原始ID】(如果未启用内存重排)
        const vector<uint32_t>& point_indices_in_cluster = index->inverted_lists_[cluster_id];

        // 3. 遍历簇内的每个数据点
        for (uint32_t point_idx_or_id : point_indices_in_cluster) 
        {
            uint32_t original_id_for_pq_lookup_thread; // 这个ID用于从全局 base_pq_codes_global_ 中查找PQ编码

            if (index->use_memory_reordering_) 
            { // 如果数据被内存重排了
                if (point_idx_or_id < index->reordered_idx_to_original_id_.size()) { // 安全检查
                    original_id_for_pq_lookup_thread = index->reordered_idx_to_original_id_[point_idx_or_id];
                } 
                else {
                    // cerr << "错误: ivf_pthread_worker_ivfpq_adc - 重排索引越界 " << point_idx_or_id << endl; // 调试
                    continue;
                }
            } 
            else { // 数据未重排
                original_id_for_pq_lookup_thread = point_idx_or_id;
            }

            if (original_id_for_pq_lookup_thread >= index->num_base_) continue; // 确保原始ID有效

            // 获取该【原始ID】对应的PQ编码
            const uint8_t* pq_code_thread = index->base_pq_codes_global_ + original_id_for_pq_lookup_thread * index->pq_codebook_global_->params.M;

            // 使用ADC计算近似内积 (使用共享的LUT)
            float approx_ip_thread = approximate_inner_product_adc(pq_code_thread, lut_for_this_thread, index->pq_codebook_global_->params);
            float approx_adc_distance_thread = 1.0f - approx_ip_thread; // 近似内积距离

            // 维护该线程的局部top-nk_pq候选
            if (thread_data->local_top_nk_results_adc.size() < nk_pq)
                thread_data->local_top_nk_results_adc.push({approx_adc_distance_thread, original_id_for_pq_lookup_thread}); // 存储【原始ID】
            else if (approx_adc_distance_thread < thread_data->local_top_nk_results_adc.top().first) 
            {
                thread_data->local_top_nk_results_adc.pop();
                thread_data->local_top_nk_results_adc.push({approx_adc_distance_thread, original_id_for_pq_lookup_thread}); // 存储【原始ID】
            }
        }
    }
    // 4. 线程执行完毕
    pthread_exit(nullptr);
}

// --- 【修改】Pthread 单阶段并行IVF-PQ搜索 (簇内用ADC) ---
// query_vector: D维查询向量 (float)
// nk_pq: 希望通过ADC召回的候选者数量
// nprobe: 需要探查的最近簇的数量
// num_threads_to_use: 用于簇内ADC搜索的工作线程数量
// precomputed_lut: 【新增参数】主线程预计算好的LUT指针
// 返回值: 一个优先队列，存储(【近似ADC距离】, 【原始数据点ID】)对
inline priority_queue<pair<float, uint32_t>> IVFIndex::search_pthread_ivfpq_adc( // <<< 确保有 IVFIndex::
    const float* query_vector,
    size_t nk_pq,
    size_t nprobe,
    int num_threads_to_use,
    const float* precomputed_lut // <<< 确保这个参数在这里
)
{
    priority_queue<pair<float, uint32_t>> final_top_nk_candidates_adc; // 用于存储最终合并的、基于ADC距离的top-Nk候选

    // --- 1. 【主线程】为当前查询向量预计算查找表 (LUT) ---
    // std::vector<float> lut_main(pq_codebook_global_->params.M * pq_codebook_global_->params.Ks);
    // compute_inner_product_lut_neon(query_vector, *pq_codebook_global_, lut_main.data());

    // --- 2. 【主线程】定位候选簇 (与 search_pthread 中的逻辑完全一致) ---
    size_t actual_nprobe = nprobe;

    vector<pair<float, uint32_t>> cluster_distances_to_query; // 存储 (到质心的【精确】距离, 簇ID)
    cluster_distances_to_query.reserve(num_clusters_in_index_);
    for (size_t c = 0; c < num_clusters_in_index_; ++c) 
    {
        const float* centroid_vec = centroids_.data() + c * dim_; // 获取质心向量
        float dist = compute_inner_product_distance_neon_optimized(query_vector, centroid_vec, dim_); // 精确距离计算
        cluster_distances_to_query.push_back({dist, (uint32_t)c});
    }

    // 对(距离, 簇ID)对进行排序，选出距离最小的 actual_nprobe 个
    if (actual_nprobe < cluster_distances_to_query.size()) 
    {
        partial_sort(cluster_distances_to_query.begin(),
                     cluster_distances_to_query.begin() + actual_nprobe,
                     cluster_distances_to_query.end());
    }
    else 
        sort(cluster_distances_to_query.begin(), cluster_distances_to_query.end());

    // 提取需要探查的簇的ID列表
    vector<uint32_t> probed_cluster_ids;
    probed_cluster_ids.reserve(actual_nprobe);
    for (size_t i = 0; i < actual_nprobe && i < cluster_distances_to_query.size(); ++i) 
        probed_cluster_ids.push_back(cluster_distances_to_query[i].second);

    // --- 3. 【主线程】准备Pthread相关变量 (与 search_pthread 类似，但使用 IVFThreadData_PQ) ---
    //    确定实际用于簇内ADC搜索的工作线程数量
    int actual_num_threads = std::min((int)probed_cluster_ids.size(), num_threads_to_use);
    if (actual_num_threads <= 0) actual_num_threads = 1; // 至少使用一个线程

    vector<pthread_t> threads(actual_num_threads);                // Pthread线程标识符数组
    vector<IVFThreadData_PQ> thread_data_array;              // 【使用新的线程数据结构 IVFThreadData_PQ】
    thread_data_array.reserve(actual_num_threads);              // 预分配空间

    // --- 4. 【主线程】分配任务给线程 ---
    //    为每个线程创建 IVFThreadData_PQ 对象，并传递主线程计算好的LUT的指针
    for (int i = 0; i < actual_num_threads; ++i) 
        thread_data_array.emplace_back(this, query_vector, nk_pq, const_cast<float*>(precomputed_lut)); 
    //    将候选簇ID (probed_cluster_ids) 轮询分配给工作线程
    for (size_t i = 0; i < probed_cluster_ids.size(); ++i) 
        thread_data_array[i % actual_num_threads].clusters_to_search.push_back(probed_cluster_ids[i]);

    // --- 5. 【主线程】创建并启动工作线程 ---
    //    每个工作线程将执行 ivf_pthread_worker_ivfpq_adc 函数
    for (int i = 0; i < actual_num_threads; ++i) 
    {
        if (!thread_data_array[i].clusters_to_search.empty()) // 只为分配到任务的线程创建
            int rc = pthread_create(&threads[i], nullptr, ivf_pthread_worker_ivfpq_adc, &thread_data_array[i]);
        else 
            threads[i] = 0; // 标记没有任务，因此未创建的线程
    }

    // --- 6. 【主线程】等待所有工作线程执行完毕 ---
    // pthread_join(pthread_t thread, void **retval);
    // 主线程会在此阻塞，直到指定的子线程终止。
    for (int i = 0; i < actual_num_threads; ++i) 
        if (threads[i] != 0) // 只 join 那些被成功创建的线程
            pthread_join(threads[i], nullptr);

    // --- 7. 【主线程】合并所有线程的局部top-nk_pq结果 (基于ADC距离) ---
    for (int i = 0; i < actual_num_threads; ++i) 
    {
        if (threads[i] != 0) 
        { // 只合并那些执行了任务的线程的结果
            priority_queue<pair<float, uint32_t>>& local_pq = thread_data_array[i].local_top_nk_results_adc;
            while (!local_pq.empty()) 
            {
                pair<float, uint32_t> candidate = local_pq.top(); // (adc_dist, original_id)
                local_pq.pop();
                // 维护全局的top-nk_pq候选 (基于ADC距离，距离越小越好)
                if (final_top_nk_candidates_adc.size() < nk_pq) {
                    final_top_nk_candidates_adc.push(candidate);
                } else if (candidate.first < final_top_nk_candidates_adc.top().first) {
                    final_top_nk_candidates_adc.pop();
                    final_top_nk_candidates_adc.push(candidate);
                }
            }
        }
    }
    return final_top_nk_candidates_adc; // 返回的是基于ADC距离的top-Nk候选及其【原始ID】
}


// --- OpenMP 并行化IVF搜索 (仅并行化簇内搜索) ---
// query_vector: D维查询向量
// k_final: 最终需要返回的top-k数量
// nprobe: 需要探查的最近簇的数量
// num_threads_to_set: 希望OpenMP在此并行区域使用的线程数量
// 返回值: 一个优先队列，存储(距离, 原始数据点ID)对
inline priority_queue<pair<float, uint32_t>> IVFIndex::search_openmp(
    const float* query_vector,
    size_t k_final,
    size_t nprobe,
    int num_threads_to_set
) 
{
    priority_queue<pair<float, uint32_t>> final_top_k_results; // 用于存储最终合并的结果


    // --- 获取当前应使用的数据源指针 (根据是否启用了内存重排) ---
    const float* current_base_to_search = getCurrentBaseDataPtr();

    // 1. 【主线程串行】定位候选簇 (与Pthread版本一致)
    size_t actual_nprobe = nprobe;
    vector<pair<float, uint32_t>> cluster_distances_to_query; // 存储(到质心的距离, 簇ID)
    cluster_distances_to_query.reserve(num_clusters_in_index_); // 预分配空间

    // 计算查询向量与所有质心的距离
    for (size_t c = 0; c < num_clusters_in_index_; ++c)
    {
        const float* centroid_vec = centroids_.data() + c * dim_; // 获取当前质心向量指针
        // 使用SIMD精确内积距离计算
        float dist = compute_inner_product_distance_neon_optimized(query_vector, centroid_vec, dim_);
        cluster_distances_to_query.push_back({dist, (uint32_t)c});
    }

    // 对(距离, 簇ID)对进行排序，选出距离最小的actual_nprobe个
    if (actual_nprobe < cluster_distances_to_query.size())
    {
        partial_sort(cluster_distances_to_query.begin(),
                     cluster_distances_to_query.begin() + actual_nprobe,
                     cluster_distances_to_query.end());
    }
    else // 如果actual_nprobe大于等于总簇数，则搜索所有簇
        sort(cluster_distances_to_query.begin(), cluster_distances_to_query.end());

    // 提取需要探查的簇的ID列表
    vector<uint32_t> probed_cluster_ids;
    probed_cluster_ids.reserve(actual_nprobe);
    for (size_t i = 0; i < actual_nprobe && i < cluster_distances_to_query.size(); ++i)
        probed_cluster_ids.push_back(cluster_distances_to_query[i].second);

    if (probed_cluster_ids.empty())
        return final_top_k_results; // 没有候选簇可供搜索

    // --- 2. 【OpenMP并行化】在选定的 probed_cluster_ids 内搜索数据点 ---

    // 创建一个vector来存储每个线程的局部top-k优先队列
    // 其大小应基于实际能创建的线程数，或者预设的线程数
    // 假设创建num_threads_to_set个
    vector<priority_queue<pair<float, uint32_t>>> per_thread_local_results(num_threads_to_set);

    // 设置OpenMP在此并行区域将要使用的线程数
    omp_set_num_threads(num_threads_to_set);

    // #pragma omp parallel: 开始一个并行区域，后续代码块由多个线程执行
    // #pragma omp for: 将紧随其后的for循环的迭代分配给并行区域中的线程
    //   - schedule(dynamic): 动态调度，线程完成当前块后请求下一块，适合迭代耗时不均的情况（如此处簇大小可能不一）
    //   - 循环变量 idx_cluster 默认是每个线程私有的。
    //   - 共享变量 (shared): query_vector, probed_cluster_ids, this (隐式共享，允许访问成员centroids_, inverted_lists_等),
    //                      current_base_to_search, k_final, dim_, use_memory_reordering_, reordered_idx_to_original_id_
    //   - 私有变量 (private): 在循环内部声明的变量，如 cluster_id, point_indices_in_cluster, point_index_in_current_base,
    //                        data_point_vec, dist_to_point, original_id_to_return 都是线程私有的。
    //   - per_thread_local_results 是共享的，但每个线程通过 thread_id 访问自己的那一份，不存在竞争。
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx_cluster = 0; idx_cluster < probed_cluster_ids.size(); ++idx_cluster)
    {
        int thread_id = omp_get_thread_num();
        uint32_t cluster_id = probed_cluster_ids[idx_cluster];

        if (cluster_id >= inverted_lists_.size())
        {
            continue;
        }
        const vector<uint32_t>& point_indices_in_cluster = inverted_lists_[cluster_id];

        for (uint32_t point_index_in_current_base : point_indices_in_cluster)
        {
            if (point_index_in_current_base >= num_base_)
            {
                continue;
            }
            const float* data_point_vec = current_base_to_search + point_index_in_current_base * dim_;
            float dist_to_point = compute_inner_product_distance_neon_optimized(query_vector, data_point_vec, dim_);

            // --- 【关键修正】获取用于返回的原始ID (适配内存重排) ---
            uint32_t original_id_to_return = point_index_in_current_base; // 默认是当前索引/ID
            if (use_memory_reordering_) { // 【与单线程和Pthread版本相同的逻辑】如果使用了内存重排
                 if (point_index_in_current_base < reordered_idx_to_original_id_.size()) { // 安全检查
                    original_id_to_return = reordered_idx_to_original_id_[point_index_in_current_base]; // 通过映射表找回原始ID
                } else {
                    // cerr << "错误: search_openmp - 重排索引越界 " << point_index_in_current_base << " in thread " << thread_id << endl; // 调试
                    continue; // 跳过这个无效的索引
                }
            }
            // ----------------------------------------------------------

            if (thread_id < per_thread_local_results.size()) 
            {
                if (per_thread_local_results[thread_id].size() < k_final)
                    per_thread_local_results[thread_id].push({dist_to_point, original_id_to_return}); // <<< 使用 original_id_to_return
                else if (dist_to_point < per_thread_local_results[thread_id].top().first)
                {
                    per_thread_local_results[thread_id].pop();
                    per_thread_local_results[thread_id].push({dist_to_point, original_id_to_return}); // <<< 使用 original_id_to_return
                }
            }
        }
    }  // omp parallel for 循环结束 (此处有一个隐式的屏障，所有线程会在此同步)

    // 3. 【主线程串行】合并所有线程的局部top-k结果
    // 在并行区域结束后，只有主线程会执行这里的代码
    for (int i = 0; i < num_threads_to_set; ++i) // 遍历每个线程的局部结果
    {
        if (i < per_thread_local_results.size()) // 安全检查
        {
            priority_queue<pair<float, uint32_t>>& local_pq = per_thread_local_results[i];
            while (!local_pq.empty())
            {
                pair<float, uint32_t> candidate = local_pq.top();
                local_pq.pop();

                // 将候选加入到最终的全局top-k优先队列中
                if (final_top_k_results.size() < k_final)
                    final_top_k_results.push(candidate);
                else if (candidate.first < final_top_k_results.top().first)
                {
                    final_top_k_results.pop();
                    final_top_k_results.push(candidate);
                }
            }
        }
    }

    return final_top_k_results; // 返回最终的top-k结果
}

#endif // IVF_INDEX_H