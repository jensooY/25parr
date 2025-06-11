#ifndef IVF_MPI_H // 包含守卫
#define IVF_MPI_H

#include <vector>
#include <queue>
#include <utility>       // std::pair
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <limits>        // 数值极限
#include <string>
#include <mpi.h>         // MPI核心头文件
#include <cstring>       // 【新增】为了 memcpy 和 memset (如果需要)

#include "kmeans.h"      // 假设 kmeans.h 是 header-only inline
#include "simd_distance.h" // SIMD内积距离计算头文件
#include "pq_utils.h"


// 使用std命名空间
using namespace std;

class IVFIndex_MPI {
public:
    // --- 构造函数 (增加一个参数控制是否启用内存重排) ---
    IVFIndex_MPI(int rank, int total_procs,
                 const float* original_base_for_rank0, // 用于KMeans或策略二Scatterv源
                 const uint8_t* global_pq_codes_ptr_rank0, // 【修改】这个参数在main中是 global_base_pq_codes_rank0_source
                 const PQCodebook* pq_codebook_ptr_all,   // 【修改】这个参数在main中是 global_pq_codebook_ptr_all_ranks
                 size_t num_base_total, size_t dimension,
                 bool enable_reordering = false) // 【新增 enable_reordering 参数】
    : rank_(rank),
      size_(total_procs),
      base_data_local_(nullptr),
      num_base_local_(0),
      local_start_idx_global_(-1),
      base_data_global_all_(nullptr),
      base_data_original_for_rank0_source_(original_base_for_rank0), // 【修改】用于rank0 Scatterv的源，在main中赋值
      base_pq_codes_global_rank0_source_for_bcast_(global_pq_codes_ptr_rank0), // 【新增】存储main传来的PQ编码源(仅rank0)
      pq_codebook_global_(pq_codebook_ptr_all),       // 【修改】存储main传来的PQ码本指针
      base_pq_codes_global_all_processes_(nullptr),// 【新增】所有进程的PQ编码数据副本 (策略一用)
      num_base_global_total_(num_base_total),
      dim_(dimension),
      use_memory_reordering_(enable_reordering),
      base_data_reordered_rank0_owner_(nullptr),
      index_fully_initialized_(false),
      num_clusters_in_index_(0)
    {
        if (num_base_global_total_ > 0 && dim_ > 0) {
            base_data_global_all_ = new float[num_base_global_total_ * dim_];
            if (base_data_global_all_ == nullptr && rank_ == 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

            // 为PQ编码的全局副本分配内存 (如果PQ码本有效)
            if (pq_codebook_global_ != nullptr && pq_codebook_global_->params.M > 0) {
                 base_pq_codes_global_all_processes_ = new uint8_t[num_base_global_total_ * pq_codebook_global_->params.M];
                 if (base_pq_codes_global_all_processes_ == nullptr && rank_ == 0) { MPI_Abort(MPI_COMM_WORLD, 1); }
            }
        }
    }
    //         if (base_data_global_all_ == nullptr && rank_ == 0) {
    //             cerr << "Rank " << rank_ << " [FATAL_CTOR]: new float for base_data_global_all_ FAILED!" << endl;
    //             MPI_Abort(MPI_COMM_WORLD, 1);
    //         }
    //     }
    // }

    // --- 析构函数 ---
    ~IVFIndex_MPI() {
        if (base_data_local_ != nullptr) { delete[] base_data_local_; }
        if (base_data_global_all_ != nullptr) { delete[] base_data_global_all_; }
        if (rank_ == 0 && base_data_reordered_rank0_owner_ != nullptr) { delete[] base_data_reordered_rank0_owner_; }
        if (base_pq_codes_global_all_processes_ != nullptr) { delete[] base_pq_codes_global_all_processes_; } // 【新增】释放
    }

    

    // --- 【修改】Rank 0 构建索引 (KMeans + 可选的内存重排) ---
    inline bool rank0_build_index_with_optional_reordering( // Rank 0 的原始数据
                                     size_t num_clusters, int kmeans_max_iterations);

    // --- 【修改】所有进程接收广播的索引核心数据 (质心、新倒排列表、新映射表) ---
    inline bool receive_bcast_index_data_after_reordering(MPI_Comm comm); // 名称变化

    // --- 【修改】策略一接收广播数据 (现在接收的是可能重排过的数据) ---
    inline bool receive_bcast_base_data_strategy1(MPI_Comm comm); // 内部逻辑可能需要调整广播源

    // --- receive_scattered_base_data_strategy2 (暂时不变，因为内存重排主要影响策略一的数据准备) ---
    // 如果策略二也要用内存重排，其分发逻辑会更复杂，我们先聚焦策略一的重排
    inline bool receive_scattered_base_data_strategy2(
                                                     const int* sendcounts_elements,
                                                     const int* displs_elements,
                                                     MPI_Comm comm);

    // --- 【修改】核心搜索逻辑 - 以适配内存重排 ---
    inline void search_clusters_locally_for_rank(const float* query_vector, size_t k_final, const vector<uint32_t>& cluster_ids_to_search, bool data_source_is_global, priority_queue<pair<float, uint32_t>>& local_top_k_results_queue, size_t current_query_idx_for_debug          ) const;
    // 【新增】PQ版本的本地搜索
    // --- 【新增MPI辅助】所有进程接收广播的PQ核心数据 (码本参数, 码本内容, 全局PQ编码) ---
    inline bool receive_bcast_pq_data(MPI_Comm comm);


    // --- 【新增】单进程执行的簇内搜索逻辑 (使用PQ ADC) ---
    inline void search_clusters_locally_pq_adc_for_rank(
        const float* query_lut, // 当前查询的预计算LUT
        size_t nk_to_recall,    // 要召回的候选数量
        const vector<uint32_t>& cluster_ids_to_search,
        bool data_source_is_global, // 对策略一，这里依然是true
        priority_queue<pair<float, uint32_t>>& local_top_nk_adc_results_queue, // (ADC距离, 原始ID)
        size_t current_query_idx_for_debug // 【增加这个参数】 
    ) const;

// 成员变量
public:
    int rank_; int size_;
    float* base_data_local_; size_t num_base_local_; long long local_start_idx_global_;
    const float* base_data_original_for_rank0_source_; // 【重要】由main.cc在rank0上赋值
    float* base_data_global_all_;
    bool use_memory_reordering_;
    float* base_data_reordered_rank0_owner_;
    vector<uint32_t> reordered_idx_to_original_id_;

    const uint8_t* base_pq_codes_global_rank0_source_for_bcast_; // 【新增】由main.cc在rank0上赋值
    const PQCodebook* pq_codebook_global_; // 【重要】由main.cc传入构造函数
    uint8_t* base_pq_codes_global_all_processes_; // 【新增】

    size_t num_base_global_total_; size_t dim_;
    bool index_fully_initialized_; size_t num_clusters_in_index_;
    vector<float> centroids_; vector<vector<uint32_t>> inverted_lists_;
};



// --- 方法实现 ---

// 【修改】Rank 0 构建索引 (KMeans + 可选的内存重排)
inline bool IVFIndex_MPI::rank0_build_index_with_optional_reordering(
    //const float* base_data_for_build,
    size_t num_clusters, int kmeans_max_iterations)
{
    if (rank_ != 0) return false;
    if (!base_data_original_for_rank0_source_ || num_base_global_total_ == 0 || dim_ == 0 || num_clusters == 0 || num_clusters > num_base_global_total_) {
        cerr << "Rank 0: 构建索引参数无效 (源数据或参数问题)。" << endl; return false;
    }
    num_clusters_in_index_ = num_clusters;

    // 1. 执行KMeans，得到初始的质心和包含【原始ID】的倒排列表
    vector<vector<uint32_t>> initial_inverted_lists(num_clusters_in_index_);
    bool kmeans_ok = run_kmeans(base_data_original_for_rank0_source_, num_base_global_total_, dim_,
                                num_clusters_in_index_, kmeans_max_iterations,
                                centroids_, initial_inverted_lists, /* stats file */ "files/rank0_kmeans_stats.txt");
    if (!kmeans_ok) { cerr << "Rank 0: KMeans失败。" << endl; return false; }
    cout << "Rank 0: KMeans完成。质心数: " << centroids_.size()/dim_ << endl;

    // 2. 执行内存重排（如果启用）
    if (use_memory_reordering_) {
        cout << "Rank 0: 开始内存重排..." << endl;
        if (base_data_reordered_rank0_owner_ != nullptr) delete[] base_data_reordered_rank0_owner_;
        base_data_reordered_rank0_owner_ = new float[num_base_global_total_ * dim_];
        reordered_idx_to_original_id_.assign(num_base_global_total_, 0); // 用0或其他无效值初始化
        vector<vector<uint32_t>> final_reordered_inverted_lists(num_clusters_in_index_);

        uint32_t current_reordered_idx = 0;
        for (size_t c = 0; c < num_clusters_in_index_; ++c) {
            for (uint32_t original_id : initial_inverted_lists[c]) {
                if (original_id < num_base_global_total_) {
                    memcpy(base_data_reordered_rank0_owner_ + current_reordered_idx * dim_,
                           base_data_original_for_rank0_source_  + original_id * dim_,
                           dim_ * sizeof(float));
                    reordered_idx_to_original_id_[current_reordered_idx] = original_id;
                    final_reordered_inverted_lists[c].push_back(current_reordered_idx);
                    current_reordered_idx++;
                }
            }
        }
        inverted_lists_ = final_reordered_inverted_lists; // 更新为基于新索引的倒排列表
        cout << "Rank 0: 内存重排完成。实际重排点数: " << current_reordered_idx << endl;
    } else {
        inverted_lists_ = initial_inverted_lists; // 使用原始ID的倒排列表
        //确保 reordered_idx_to_original_id_ 在不重排时为空或不被使用
        reordered_idx_to_original_id_.clear();
    }
    index_fully_initialized_ = true; // 仅对rank0标记，实际初始化需广播后
    return true;
}

// 【修改】所有进程接收广播的索引核心数据 (现在可能包含重排相关数据)
inline bool IVFIndex_MPI::receive_bcast_index_data_after_reordering(MPI_Comm comm)
{
    // 0. 广播 use_memory_reordering_ 标志，以便所有进程知道当前模式
    int reordering_flag_int = (rank_ == 0) ? (use_memory_reordering_ ? 1 : 0) : 0;
    MPI_Bcast(&reordering_flag_int, 1, MPI_INT, 0, comm);
    if (rank_ != 0) use_memory_reordering_ = (reordering_flag_int == 1);

    // 1. 广播质心和倒排列表 (与之前的 receive_bcast_index_data 逻辑相同)
    //    因为 inverted_lists_ 在rank 0上已经根据是否重排被正确设置了
    size_t centroids_flat_size_to_bcast  = (rank_ == 0) ? centroids_.size() : 0;
    MPI_Bcast(&centroids_flat_size_to_bcast, 1, MPI_UNSIGNED_LONG, 0, comm);
    if (dim_ > 0) num_clusters_in_index_ = centroids_flat_size_to_bcast  / dim_;
    else if (centroids_flat_size_to_bcast  > 0) return false; else num_clusters_in_index_ = 0;

    if (rank_ != 0) centroids_.resize(centroids_flat_size_to_bcast );
    if (centroids_flat_size_to_bcast  > 0) MPI_Bcast(centroids_.data(), centroids_flat_size_to_bcast , MPI_FLOAT, 0, comm);

    if (rank_ != 0) inverted_lists_.assign(num_clusters_in_index_, vector<uint32_t>());
    for (size_t c = 0; c < num_clusters_in_index_; ++c) {
        size_t inner_list_size_bcast = (rank_ == 0) ? inverted_lists_[c].size() : 0;
        MPI_Bcast(&inner_list_size_bcast, 1, MPI_UNSIGNED_LONG, 0, comm);
        if (rank_ != 0) inverted_lists_[c].resize(inner_list_size_bcast);
        if (inner_list_size_bcast > 0) MPI_Bcast(inverted_lists_[c].data(), inner_list_size_bcast, MPI_UNSIGNED, 0, comm);
    }

    // 2. 【新增】如果启用了内存重排，广播 reordered_idx_to_original_id_ 映射表
    if (use_memory_reordering_) {
        size_t map_size_bcast = (rank_ == 0) ? reordered_idx_to_original_id_.size() : 0;
        MPI_Bcast(&map_size_bcast, 1, MPI_UNSIGNED_LONG, 0, comm);
        if (rank_ != 0) {
            reordered_idx_to_original_id_.resize(map_size_bcast);
        }
        if (map_size_bcast > 0) { // 只有当映射表不为空时（即确实有数据被重排）才广播
            MPI_Bcast(reordered_idx_to_original_id_.data(), map_size_bcast, MPI_UNSIGNED, 0, comm);
        }
    } else { // 如果不使用内存重排，其他进程也应清空此表
        if (rank_ != 0) reordered_idx_to_original_id_.clear();
    }

    index_fully_initialized_ = true;
    return true;
}

// 【修改】策略一接收广播数据 (现在接收的是由rank0拥有的、可能重排过的数据)
inline bool IVFIndex_MPI::receive_bcast_base_data_strategy1(MPI_Comm comm)
{
    if (num_base_global_total_ > 0 && dim_ > 0) {
        // 所有进程的 base_data_global_all_ 应该在构造时已分配内存
        if (base_data_global_all_ == nullptr) { // 再次检查，理论上不应发生
            if (rank_ == 0) cerr << "Rank 0 [ERROR_BCAST_S1]: base_data_global_all_ is NULL before Bcast!" << endl;
            // 可以尝试在这里分配，但更应该在构造时完成
            base_data_global_all_ = new float[num_base_global_total_ * dim_];
             if (base_data_global_all_ == nullptr) { MPI_Abort(MPI_COMM_WORLD,1); return false;}
        }
        // Rank 0 的发送源现在是 base_data_reordered_rank0_owner_ (如果重排) 或 base_data_original_for_rank0_source_ (如果未重排)
        // 为了让 MPI_Bcast 的 buffer 参数对于根进程是发送缓冲区，对于其他进程是接收缓冲区，
        // Rank 0 需要将数据【先复制到它自己的 base_data_global_all_】中，然后再调用 MPI_Bcast。
        // 这个复制操作应该在 main.cc 中，在调用此函数之前完成。
        // 此函数假设 rank 0 的 base_data_global_all_ 已经包含了要广播的数据。
        MPI_Bcast(base_data_global_all_, num_base_global_total_ * dim_, MPI_FLOAT, 0, comm);
    }
    return true;
}

// // receive_scattered_base_data_strategy2 (暂时保持不变，因为我们先聚焦策略一的内存重排)
// inline bool IVFIndex_MPI::receive_scattered_base_data_strategy2(const float* send_buffer_rank0, const int* sendcounts_elements, const int* displs_elements, MPI_Comm comm) 
// { /* ... 与之前相同 ... */ return true;}


// 【修改】核心搜索逻辑 - 以适配内存重排
inline void IVFIndex_MPI::search_clusters_locally_for_rank(
    const float* query_vector,
    size_t k_final,
    const vector<uint32_t>& cluster_ids_to_search,
    bool data_source_is_global, // 对于策略一 + 内存重排，这个应为true
    priority_queue<pair<float, uint32_t>>& local_top_k_results_queue,
    size_t current_query_idx_for_debug          // 新增的参数
) const
{
    while(!local_top_k_results_queue.empty()) local_top_k_results_queue.pop();
    if (!index_fully_initialized_ ) return;

    // --- 获取当前应使用的数据源指针 (策略一总是用base_data_global_all_) ---
    const float* current_base_to_use_for_search = nullptr;
    if (data_source_is_global) { // 策略一
        current_base_to_use_for_search = base_data_global_all_; // 它现在可能指向重排后的数据
    } else { // 策略二
        current_base_to_use_for_search = base_data_local_;
    }

    if (current_base_to_use_for_search == nullptr && num_base_global_total_ > 0 && (data_source_is_global || num_base_local_ > 0) ) {
        // cerr << "Rank " << rank_ << ": search_clusters_locally - 使用的数据源指针为空!" << endl; // 调试
        return;
    }

    for (uint32_t cluster_id : cluster_ids_to_search)
    {
        if (cluster_id >= inverted_lists_.size()) continue;
        // inverted_lists_ 现在包含的是【新索引】(如果重排) 或 【原始ID】(如果未重排)
        const vector<uint32_t>& point_indices_in_this_cluster = inverted_lists_[cluster_id];

        for (uint32_t current_idx_in_base : point_indices_in_this_cluster) // 这个是重排后的索引或原始ID
        {
            // 安全检查，确保索引在当前使用的数据源的范围内
            size_t current_base_size_for_check = data_source_is_global ? num_base_global_total_ : num_base_local_;
            if (current_idx_in_base >= current_base_size_for_check) {
                // cerr << "Rank " << rank_ << ": search_clusters_locally - 索引 " << current_idx_in_base << " 越界 (max: " << current_base_size_for_check -1 << ")" << endl;
                continue;
            }

            // 从正确的数据源获取数据点向量
            const float* data_point_vec_to_compare = current_base_to_use_for_search + current_idx_in_base * dim_;

            float dist_to_point = compute_inner_product_distance_neon_optimized(query_vector, data_point_vec_to_compare, dim_);

            // --- 将结果存入优先队列时，需要使用【原始ID】进行评估和返回 ---
            uint32_t original_id_for_result = current_idx_in_base; // 默认
            if (use_memory_reordering_) { // 如果启用了内存重排，current_idx_in_base是重排后的索引
                if (current_idx_in_base < reordered_idx_to_original_id_.size()) {
                    original_id_for_result = reordered_idx_to_original_id_[current_idx_in_base];
                } else {
                    // cerr << "Rank " << rank_ << ": search_clusters_locally - 重排映射表越界，索引 " << current_idx_in_base << endl;
                    continue;
                }
            }
            // 对于策略二（如果未来也用内存重排），如果 current_base_to_use_for_search 是 local_base_reordered，
            // 那么 current_idx_in_base 是本地重排索引，需要先映射到全局原始ID。
            // 但当前 search_clusters_locally_for_rank 的设计是 inverted_lists_ 总是基于全局原始ID（如果未重排）
            // 或全局重排索引（如果重排）。策略二的本地化是在获取 data_point_vec_to_compare 时处理的。
            // 所以，只要 inverted_lists_ 的内容与 use_memory_reordering_ 状态一致，这里的 original_id_for_result 逻辑是对的。


            if (local_top_k_results_queue.size() < k_final) {
                local_top_k_results_queue.push({dist_to_point, original_id_for_result});
            } else if (dist_to_point < local_top_k_results_queue.top().first) {
                local_top_k_results_queue.pop();
                local_top_k_results_queue.push({dist_to_point, original_id_for_result});
            }
        }
    }
}


// 在 ivf_mpi.h 文件中，IVFIndex_MPI 类的定义之后

inline bool IVFIndex_MPI::receive_bcast_pq_data(MPI_Comm comm) {
    // 0. 检查Rank 0是否有有效的PQ码本指针
    //    并广播一个成功/失败标志，以便其他进程知道是否继续
    int pq_data_ready_flag_rank0 = 0;
    if (rank_ == 0) {
        if (pq_codebook_global_ != nullptr && base_pq_codes_global_rank0_source_for_bcast_ != nullptr) {
            pq_data_ready_flag_rank0 = 1; // Rank 0 准备好了PQ数据
        } else {
            cerr << "Rank 0: 错误！尝试广播PQ数据，但PQ码本指针或PQ编码源指针为空。" << endl;
        }
    }
    MPI_Bcast(&pq_data_ready_flag_rank0, 1, MPI_INT, 0, comm);

    if (pq_data_ready_flag_rank0 == 0) { // 如果Rank 0未能准备好数据
        if (rank_ != 0) cerr << "Rank " << rank_ << ": 收到Rank 0 PQ数据准备失败的信号。" << endl;
        // index_fully_initialized_ 应该为 false，或者有一个单独的 pq_initialized_ 标志
        return false; // 所有进程都不能继续
    }

    // 1. 广播 PQParams 结构体
    PQParams params_buffer_for_bcast; // 所有进程都需要这个缓冲区来接收/发送
    if (rank_ == 0) {
        // 再次确认，理论上 pq_codebook_global_ 此时不应为 nullptr
        if (pq_codebook_global_) {
            params_buffer_for_bcast = pq_codebook_global_->params;
        } else { /* 已在上面处理 */ }
    }
    MPI_Bcast(&params_buffer_for_bcast, sizeof(PQParams), MPI_BYTE, 0, comm);

    // 其他进程用接收到的参数填充它们的（通常是main中创建并传入指针的）PQCodebook对象的params成员
    if (rank_ != 0) {
        if (pq_codebook_global_ != nullptr) { // 确保指针有效
            // 我们假设 pq_codebook_global_ 指向的是一个所有进程都有的、可修改的 PQCodebook 对象
            // (例如 main.cc 中的 global_pq_codebook_obj_main)
            const_cast<PQCodebook*>(pq_codebook_global_)->params = params_buffer_for_bcast;
        } else {
            cerr << "Rank " << rank_ << ": 错误！接收PQ参数时，本地pq_codebook_global_指针为空。" << endl;
            MPI_Abort(comm, 1); return false; // 关键错误
        }
    }

    // 2. 广播 PQ 码本的质心数据 (pq_codebook_global_->centroids)
    size_t pq_centroids_flat_size = 0;
    if (pq_codebook_global_ != nullptr) { // 确保指针有效
        pq_centroids_flat_size = (rank_ == 0) ? pq_codebook_global_->centroids.size() : 0;
    }
    MPI_Bcast(&pq_centroids_flat_size, 1, MPI_UNSIGNED_LONG, 0, comm);

    if (pq_codebook_global_ != nullptr && rank_ != 0) {
        const_cast<PQCodebook*>(pq_codebook_global_)->centroids.resize(pq_centroids_flat_size);
    }
    if (pq_centroids_flat_size > 0 && pq_codebook_global_ != nullptr) {
        // Rank 0 从其 pq_codebook_global_->centroids.data() 发送
        // 其他 Rank 接收到其 pq_codebook_global_->centroids.data()
        MPI_Bcast(const_cast<float*>(pq_codebook_global_->centroids.data()), pq_centroids_flat_size, MPI_FLOAT, 0, comm);
    }

    // 3. 【策略一】广播完整的 PQ 编码数据 (到所有进程的 base_pq_codes_global_all_processes_)
    //    Rank 0 的发送源是它在IVFIndex构造时接收的 base_pq_codes_global_rank0_source_for_bcast_
    //    或者是在main.cc中，在调用此函数前，将全局编码的PQ数据【复制】到它自己的
    //    ivf_search_index.base_pq_codes_global_all_processes_ 成员中（如果采用这种模式）。
    //    我们采用后者，即假设main.cc中rank0已将其源PQ编码复制到它自己的
    //    IVFIndex实例的 base_pq_codes_global_all_processes_ 成员。
    if (num_base_global_total_ > 0 && pq_codebook_global_ != nullptr && pq_codebook_global_->params.M > 0) {
        size_t pq_codes_total_elements = num_base_global_total_ * pq_codebook_global_->params.M;
        // 确保所有进程的接收缓冲区已分配 (应该在构造函数中完成)
        if (base_pq_codes_global_all_processes_ == nullptr) {
             base_pq_codes_global_all_processes_ = new uint8_t[pq_codes_total_elements];
             if(base_pq_codes_global_all_processes_ == nullptr) {
                 if (rank_ == 0) cerr << "Rank 0: receive_bcast_pq_data 中 base_pq_codes_global_all_processes_ 分配失败！" << endl;
                 MPI_Abort(comm,1); return false;
             }
        }
        // MPI_Bcast 的第一个参数，对于根进程是发送缓冲区，对于其他进程是接收缓冲区。
        // 所有进程都使用它们自己的 base_pq_codes_global_all_processes_成员。
        MPI_Bcast(base_pq_codes_global_all_processes_, pq_codes_total_elements, MPI_UNSIGNED_CHAR, 0, comm);
    }
    // cout << "Rank " << rank_ << ": PQ核心数据接收完成。" << endl; // 调试
    return true;
}


// 【新增】单进程执行的簇内搜索逻辑 (使用PQ ADC)
inline void IVFIndex_MPI::search_clusters_locally_pq_adc_for_rank(
    const float* query_lut,
    size_t nk_to_recall,
    const vector<uint32_t>& cluster_ids_to_search,
    bool data_source_is_global_pq, // 对于策略一，这个是true
    priority_queue<pair<float, uint32_t>>& local_top_nk_adc_results_queue,
    size_t current_query_idx_for_debug // 新增的参数 // 用于调试打印
) const
{
    while(!local_top_nk_adc_results_queue.empty()) local_top_nk_adc_results_queue.pop();
    if (!index_fully_initialized_ || !pq_codebook_global_ || !base_pq_codes_global_all_processes_ ) { return; }

    const uint8_t* current_pq_codes_to_use = base_pq_codes_global_all_processes_; // 策略一用全局PQ码

    for (uint32_t cluster_id : cluster_ids_to_search) {
        if (cluster_id >= inverted_lists_.size()) continue;
        const vector<uint32_t>& point_indices_or_ids_in_cluster = inverted_lists_[cluster_id];
        for (uint32_t current_idx_or_id : point_indices_or_ids_in_cluster) {
            uint32_t original_id_for_pq_lookup;
            if (use_memory_reordering_) {
                if (current_idx_or_id < reordered_idx_to_original_id_.size()) {
                    original_id_for_pq_lookup = reordered_idx_to_original_id_[current_idx_or_id];
                } else continue;
            } else {
                original_id_for_pq_lookup = current_idx_or_id;
            }
            if (original_id_for_pq_lookup >= num_base_global_total_) continue;
            const uint8_t* pq_code_of_point = current_pq_codes_to_use + original_id_for_pq_lookup * pq_codebook_global_->params.M;
            float approx_ip = approximate_inner_product_adc(pq_code_of_point, query_lut, pq_codebook_global_->params); // 确保此函数可见
            float approx_adc_distance = 1.0f - approx_ip;
            if (local_top_nk_adc_results_queue.size() < nk_to_recall) {
                local_top_nk_adc_results_queue.push({approx_adc_distance, original_id_for_pq_lookup});
            } else if (approx_adc_distance < local_top_nk_adc_results_queue.top().first) {
                local_top_nk_adc_results_queue.pop();
                local_top_nk_adc_results_queue.push({approx_adc_distance, original_id_for_pq_lookup});
            }
        }
    }
}
#endif // IVF_MPI_H
