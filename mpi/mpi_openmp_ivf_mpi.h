#ifndef IVF_MPI_H
#define IVF_MPI_H

#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <mpi.h>
#include <omp.h>   // OpenMP头文件
#include <cstring> // For memcpy

#include "kmeans.h"      // 假设 kmeans.h 包含 run_kmeans
#include "simd_distance.h" // 包含 compute_inner_product_distance_neon_optimized

using namespace std;

class IVFIndex_MPI {
public:
    // --- 构造函数 ---
    IVFIndex_MPI(int rank, int total_procs,
                 const float* original_base_for_rank0, // Rank 0的原始浮点base，用于KMeans和重排源
                 size_t num_base_total, size_t dimension,
                 bool enable_reordering = false)
    : rank_(rank),
      size_(total_procs),
      base_data_local_(nullptr), // 用于策略二（如果以后实现）
      num_base_local_(0),
      local_start_idx_global_(-1),
      base_data_global_all_(nullptr), // 所有进程接收的（可能重排的）基础数据
      base_data_original_for_rank0_source_(original_base_for_rank0), // Rank 0持有的原始数据指针
      num_base_global_total_(num_base_total),
      dim_(dimension),
      use_memory_reordering_(enable_reordering),
      base_data_reordered_rank0_owner_(nullptr), // Rank 0拥有的重排后数据，用于广播
      index_fully_initialized_(false),
      num_clusters_in_index_(0)
    {
        if (num_base_global_total_ > 0 && dim_ > 0) {
            base_data_global_all_ = new float[num_base_global_total_ * dim_];
            if (base_data_global_all_ == nullptr && rank_ == 0) {
                cerr << "Rank " << rank_ << " [FATAL_CTOR]: new float for base_data_global_all_ FAILED!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    ~IVFIndex_MPI() {
        if (base_data_local_ != nullptr) { delete[] base_data_local_; }
        if (base_data_global_all_ != nullptr) { delete[] base_data_global_all_; }
        if (rank_ == 0 && base_data_reordered_rank0_owner_ != nullptr) { delete[] base_data_reordered_rank0_owner_; }
    }

    // Rank 0 构建索引 (KMeans + 可选内存重排)
    inline bool rank0_build_index_with_optional_reordering(size_t num_clusters, int kmeans_max_iterations);
    // 所有进程接收广播的IVF核心数据 (质心、重排后倒排表、映射表)
    inline bool receive_bcast_index_data_after_reordering(MPI_Comm comm);
    // 策略一：所有进程接收广播的（可能重排的）基础浮点数据
    inline bool receive_bcast_base_data_strategy1(MPI_Comm comm);

    // 核心搜索逻辑 - MPI进程内部使用OpenMP并行化簇内点精确距离搜索
    inline void search_clusters_locally_with_omp(
        const float* query_vector,
        size_t k_final,
        const vector<uint32_t>& cluster_ids_to_search_by_this_mpi_rank,
        priority_queue<pair<float, uint32_t>>& output_mpi_process_local_top_k,
        int num_omp_threads_to_use_inside,
        size_t current_query_idx_for_debug // 可选的调试参数
    ) const;

public:
    int rank_; int size_;
    float* base_data_local_; size_t num_base_local_; long long local_start_idx_global_;
    const float* base_data_original_for_rank0_source_;
    float* base_data_global_all_;
    bool use_memory_reordering_;
    float* base_data_reordered_rank0_owner_;
    vector<uint32_t> reordered_idx_to_original_id_;
    size_t num_base_global_total_; size_t dim_;
    bool index_fully_initialized_; size_t num_clusters_in_index_;
    vector<float> centroids_; vector<vector<uint32_t>> inverted_lists_;
};

// --- 方法实现 ---
inline bool IVFIndex_MPI::rank0_build_index_with_optional_reordering(size_t num_clusters, int kmeans_max_iterations) {
    if (rank_ != 0) return false;
    if (!base_data_original_for_rank0_source_ || num_base_global_total_ == 0 || dim_ == 0 || num_clusters == 0 || num_clusters > num_base_global_total_) {
        cerr << "Rank 0: 构建索引参数无效。" << endl; return false;
    }
    num_clusters_in_index_ = num_clusters;
    vector<vector<uint32_t>> initial_inverted_lists(num_clusters_in_index_);
    bool kmeans_ok = run_kmeans(base_data_original_for_rank0_source_, num_base_global_total_, dim_,
                                num_clusters_in_index_, kmeans_max_iterations,
                                centroids_, initial_inverted_lists, "files/rank0_kmeans_stats.txt");
    if (!kmeans_ok) { cerr << "Rank 0: KMeans失败。" << endl; return false; }
    // cout << "Rank 0: KMeans完成。质心数: " << centroids_.size()/dim_ << endl;

    if (use_memory_reordering_) {
        // cout << "Rank 0: 开始内存重排..." << endl;
        if (base_data_reordered_rank0_owner_ != nullptr) delete[] base_data_reordered_rank0_owner_;
        base_data_reordered_rank0_owner_ = new float[num_base_global_total_ * dim_];
        if(base_data_reordered_rank0_owner_ == nullptr) { cerr << "Rank 0: 重排内存分配失败!" << endl; return false; }
        reordered_idx_to_original_id_.assign(num_base_global_total_, 0);
        vector<vector<uint32_t>> final_reordered_inverted_lists(num_clusters_in_index_);
        uint32_t current_reordered_idx = 0;
        for (size_t c = 0; c < num_clusters_in_index_; ++c) {
            for (uint32_t original_id : initial_inverted_lists[c]) {
                if (original_id < num_base_global_total_) {
                    memcpy(base_data_reordered_rank0_owner_ + current_reordered_idx * dim_,
                           base_data_original_for_rank0_source_ + original_id * dim_,
                           dim_ * sizeof(float));
                    reordered_idx_to_original_id_[current_reordered_idx] = original_id;
                    final_reordered_inverted_lists[c].push_back(current_reordered_idx);
                    current_reordered_idx++;
                }
            }
        }
        inverted_lists_ = final_reordered_inverted_lists;
        // cout << "Rank 0: 内存重排完成。实际重排点数: " << current_reordered_idx << endl;
    } else {
        inverted_lists_ = initial_inverted_lists;
        reordered_idx_to_original_id_.clear();
    }
    // index_fully_initialized_ 标记由 receive_bcast_index_data_after_reordering 设置
    return true;
}

inline bool IVFIndex_MPI::receive_bcast_index_data_after_reordering(MPI_Comm comm) {
    int reordering_flag_int = (rank_ == 0) ? (use_memory_reordering_ ? 1 : 0) : 0;
    MPI_Bcast(&reordering_flag_int, 1, MPI_INT, 0, comm);
    if (rank_ != 0) use_memory_reordering_ = (reordering_flag_int == 1);

    size_t centroids_flat_size_bcast = (rank_ == 0) ? centroids_.size() : 0;
    MPI_Bcast(&centroids_flat_size_bcast, 1, MPI_UNSIGNED_LONG, 0, comm);
    if (dim_ > 0) num_clusters_in_index_ = centroids_flat_size_bcast / dim_;
    else if (centroids_flat_size_bcast > 0) { if(rank_==0) cerr<<"Dim is 0 error\n"; return false;}
    else num_clusters_in_index_ = 0;

    if (rank_ != 0) centroids_.resize(centroids_flat_size_bcast);
    if (centroids_flat_size_bcast > 0) MPI_Bcast(centroids_.data(), centroids_flat_size_bcast, MPI_FLOAT, 0, comm);

    if (rank_ != 0) inverted_lists_.assign(num_clusters_in_index_, vector<uint32_t>());
    for (size_t c = 0; c < num_clusters_in_index_; ++c) {
        size_t inner_list_size_bcast_local = (rank_ == 0) ? inverted_lists_[c].size() : 0;
        MPI_Bcast(&inner_list_size_bcast_local, 1, MPI_UNSIGNED_LONG, 0, comm);
        if (rank_ != 0) inverted_lists_[c].resize(inner_list_size_bcast_local);
        if (inner_list_size_bcast_local > 0) MPI_Bcast(inverted_lists_[c].data(), inner_list_size_bcast_local, MPI_UNSIGNED, 0, comm);
    }

    if (use_memory_reordering_) {
        size_t map_size_bcast = (rank_ == 0) ? reordered_idx_to_original_id_.size() : 0;
        MPI_Bcast(&map_size_bcast, 1, MPI_UNSIGNED_LONG, 0, comm);
        if (rank_ != 0) reordered_idx_to_original_id_.resize(map_size_bcast);
        if (map_size_bcast > 0) MPI_Bcast(reordered_idx_to_original_id_.data(), map_size_bcast, MPI_UNSIGNED, 0, comm);
    } else {
        if (rank_ != 0) reordered_idx_to_original_id_.clear();
    }
    index_fully_initialized_ = true;
    return true;
}

inline bool IVFIndex_MPI::receive_bcast_base_data_strategy1(MPI_Comm comm) {
    if (num_base_global_total_ > 0 && dim_ > 0) {
        if (base_data_global_all_ == nullptr) { // 应该在构造时已分配
             base_data_global_all_ = new float[num_base_global_total_ * dim_];
             if (base_data_global_all_ == nullptr) { MPI_Abort(MPI_COMM_WORLD,1); return false;}
        }
        MPI_Bcast(base_data_global_all_, num_base_global_total_ * dim_, MPI_FLOAT, 0, comm);
    }
    return true;
}

// search_clusters_locally_with_omp 实现 (与上一条回复中的最终版本一致)
inline void IVFIndex_MPI::search_clusters_locally_with_omp(
    const float* query_vector, size_t k_final,
    const vector<uint32_t>& cluster_ids_to_search_by_this_mpi_rank,
    priority_queue<pair<float, uint32_t>>& output_mpi_process_local_top_k,
    int num_omp_threads_to_set,
    size_t current_query_idx_for_debug
) const {
    while(!output_mpi_process_local_top_k.empty()) output_mpi_process_local_top_k.pop();
    if (!index_fully_initialized_ || cluster_ids_to_search_by_this_mpi_rank.empty() || base_data_global_all_ == nullptr) {
        return;
    }
    const float* current_base_to_search = base_data_global_all_;

    vector<priority_queue<pair<float, uint32_t>>> per_omp_thread_local_results(num_omp_threads_to_set);
    omp_set_num_threads(num_omp_threads_to_set);

    #pragma omp parallel
    {
        int omp_tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < cluster_ids_to_search_by_this_mpi_rank.size(); ++i) {
            uint32_t cluster_id = cluster_ids_to_search_by_this_mpi_rank[i];
            if (cluster_id >= inverted_lists_.size()) continue;
            const vector<uint32_t>& point_indices_in_this_cluster = inverted_lists_[cluster_id];
            for (uint32_t reordered_idx : point_indices_in_this_cluster) {
                if (reordered_idx >= num_base_global_total_) continue;
                const float* data_point_vec = current_base_to_search + reordered_idx * dim_;
                float dist_to_point = compute_inner_product_distance_neon_optimized(query_vector, data_point_vec, dim_);
                uint32_t original_id_for_result = reordered_idx;
                if (use_memory_reordering_) { // 启用了内存重排，所以这个条件为true
                    if (reordered_idx < reordered_idx_to_original_id_.size()) {
                        original_id_for_result = reordered_idx_to_original_id_[reordered_idx];
                    } else { continue; }
                }
                if (omp_tid < per_omp_thread_local_results.size()) {
                    if (per_omp_thread_local_results[omp_tid].size() < k_final) {
                        per_omp_thread_local_results[omp_tid].push({dist_to_point, original_id_for_result});
                    } else if (dist_to_point < per_omp_thread_local_results[omp_tid].top().first) {
                        per_omp_thread_local_results[omp_tid].pop();
                        per_omp_thread_local_results[omp_tid].push({dist_to_point, original_id_for_result});
                    }
                }
            }
        }
    }
    for (int i = 0; i < num_omp_threads_to_set; ++i) {
        if (i < per_omp_thread_local_results.size()) {
            priority_queue<pair<float, uint32_t>>& local_pq_from_omp = per_omp_thread_local_results[i];
            while (!local_pq_from_omp.empty()) {
                pair<float, uint32_t> candidate = local_pq_from_omp.top();
                local_pq_from_omp.pop();
                if (output_mpi_process_local_top_k.size() < k_final) {
                    output_mpi_process_local_top_k.push(candidate);
                } else if (candidate.first < output_mpi_process_local_top_k.top().first) {
                    output_mpi_process_local_top_k.pop();
                    output_mpi_process_local_top_k.push(candidate);
                }
            }
        }
    }
}

#endif // IVF_MPI_H
