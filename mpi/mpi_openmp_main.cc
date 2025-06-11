#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <set>
#include <chrono>
#include <iomanip>
#include <queue>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <omp.h>   // OpenMP头文件
#include <cstring> // For memcpy

// --- 自定义头文件 ---
#include "simd_distance.h"
#include "kmeans.h"
#include "ivf_mpi.h"

using namespace std;

// --- 数据加载函数 ---
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int current_rank_for_print) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        if (current_rank_for_print == 0) cerr << "错误：无法打开数据文件 " << data_path << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); return nullptr;
    }
    int temp_n, temp_d;
    fin.read(reinterpret_cast<char*>(&temp_n), 4); fin.read(reinterpret_cast<char*>(&temp_d), 4);
    n = static_cast<size_t>(temp_n); d = static_cast<size_t>(temp_d);
    T* data = new T[n * d]; size_t sz = sizeof(T);
    for(size_t i = 0; i < n; ++i) fin.read(reinterpret_cast<char*>(data + i * d), (long long)d * sz);
    fin.close();
    if (current_rank_for_print == 0) {
        std::cerr<<"加载数据 "<<data_path<<"\n";
        std::cerr<<"维度: "<<d<<"  数量:"<<n<<"  每元素大小:"<<sizeof(T)<<"\n";
    }
    return data;
}

// --- 查询结果结构体 ---
struct SearchResult { float recall; int64_t latency_us; };

// --- main 函数入口 ---
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        cout << "--- [main_mpi_omp.cc] IVF MPI + OpenMP (策略一+内存重排+簇内OMP) ---" << endl;
        cout << "MPI 总进程数: " << world_size << endl;
    }

    size_t test_number_global = 0, base_number_global = 0, test_gt_d_global = 0, vecdim_global = 0;
    float* test_query_global_rank0 = nullptr;
    int* test_gt_global_rank0 = nullptr;
    float* base_data_for_build_rank0_main = nullptr;

    if (world_rank == 0) {
        std::string data_path = "/anndata/";
        test_query_global_rank0 = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number_global, vecdim_global, world_rank);
        test_gt_global_rank0 = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number_global, test_gt_d_global, world_rank);
        base_data_for_build_rank0_main = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number_global, vecdim_global, world_rank);
    }
    MPI_Bcast(&test_number_global, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number_global, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim_global, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d_global, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    size_t test_number_to_run = std::min(test_number_global, (size_t)2000);
    const size_t k_final = 10;
    size_t num_ivf_clusters = 2048;
    int kmeans_max_iterations = 20;
    size_t nprobe_param = 20;
    bool enable_memory_reordering = true; // 固定启用内存重排
    int omp_threads_per_mpi_process = 1;  // 每个MPI进程内部的OpenMP线程数

    if (world_rank == 0) {
        cout << "配置: 内存重排=启用, 数据划分=策略一(广播)" << endl;
        cout << "IVF参数: 簇数=" << num_ivf_clusters << ", nprobe=" << nprobe_param << ", k=" << k_final << endl;
        cout << "OpenMP参数: 每个MPI进程内部OMP线程数=" << omp_threads_per_mpi_process << endl;
    }

    IVFIndex_MPI ivf_search_index(world_rank, world_size,
                                  base_data_for_build_rank0_main,
                                  base_number_global, vecdim_global,
                                  enable_memory_reordering);

    if (world_rank == 0) {
        if (!ivf_search_index.rank0_build_index_with_optional_reordering(num_ivf_clusters, kmeans_max_iterations)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (!ivf_search_index.receive_bcast_index_data_after_reordering(MPI_COMM_WORLD)) { MPI_Abort(MPI_COMM_WORLD, 1); }

    if (world_rank == 0) { // Rank 0 将重排后的数据复制到它的广播缓冲区
        const float* source_for_bcast_s1 = ivf_search_index.base_data_reordered_rank0_owner_;
        if (source_for_bcast_s1 && ivf_search_index.base_data_global_all_) {
            memcpy(ivf_search_index.base_data_global_all_, source_for_bcast_s1, base_number_global * vecdim_global * sizeof(float));
        } else if (base_number_global > 0) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (!ivf_search_index.receive_bcast_base_data_strategy1(MPI_COMM_WORLD)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    if (world_rank == 0) cout << "策略一：重排后基础浮点数据广播完成。" << endl;

    if (world_rank == 0 && base_data_for_build_rank0_main != nullptr) {
        delete[] base_data_for_build_rank0_main; base_data_for_build_rank0_main = nullptr;
    }

    vector<SearchResult> results_collector_rank0_final(test_number_to_run);
    vector<float> current_query_buffer_allranks(vecdim_global);
    vector<uint32_t> current_query_probed_ids_rank0_buffer;
    vector<uint32_t> probed_clusters_buffer_for_bcast_allranks;
    vector<float> local_top_k_distances_sendbuf(k_final, std::numeric_limits<float>::max());
    vector<uint32_t> local_top_k_ids_sendbuf(k_final, std::numeric_limits<uint32_t>::max());
    vector<float> gathered_all_distances_recvbuf_rank0;
    vector<uint32_t> gathered_all_ids_recvbuf_rank0;
    if (world_rank == 0) {
        gathered_all_distances_recvbuf_rank0.resize(k_final * world_size);
        gathered_all_ids_recvbuf_rank0.resize(k_final * world_size);
    }

    double total_query_processing_time_seconds_rank0_overall = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) { total_query_processing_time_seconds_rank0_overall = MPI_Wtime(); }

    for (size_t i = 0; i < test_number_to_run; ++i) {
        if (world_rank == 0) {
            std::copy(test_query_global_rank0 + i * vecdim_global, test_query_global_rank0 + (i + 1) * vecdim_global, current_query_buffer_allranks.data());
        }
        MPI_Bcast(current_query_buffer_allranks.data(), vecdim_global, MPI_FLOAT, 0, MPI_COMM_WORLD);

        current_query_probed_ids_rank0_buffer.clear();
        if (world_rank == 0) {
            vector<pair<float, uint32_t>> cluster_distances_q_main;
            cluster_distances_q_main.reserve(ivf_search_index.num_clusters_in_index_);
            for (size_t c = 0; c < ivf_search_index.num_clusters_in_index_; ++c) {
                const float* centroid_vec = ivf_search_index.centroids_.data() + c * ivf_search_index.dim_;
                float dist = compute_inner_product_distance_neon_optimized(current_query_buffer_allranks.data(), centroid_vec, ivf_search_index.dim_);
                cluster_distances_q_main.push_back({dist, (uint32_t)c});
            }
            if (nprobe_param < cluster_distances_q_main.size()) {
                partial_sort(cluster_distances_q_main.begin(), cluster_distances_q_main.begin() + nprobe_param, cluster_distances_q_main.end());
            } else {
                sort(cluster_distances_q_main.begin(), cluster_distances_q_main.end());
            }
            current_query_probed_ids_rank0_buffer.reserve(nprobe_param);
            for (size_t idx = 0; idx < nprobe_param && idx < cluster_distances_q_main.size(); ++idx) {
                current_query_probed_ids_rank0_buffer.push_back(cluster_distances_q_main[idx].second);
            }
        }
        size_t num_probed_to_bcast = (world_rank == 0) ? current_query_probed_ids_rank0_buffer.size() : 0;
        MPI_Bcast(&num_probed_to_bcast, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        probed_clusters_buffer_for_bcast_allranks.resize(num_probed_to_bcast);
        if (world_rank == 0 && num_probed_to_bcast > 0) { std::copy(current_query_probed_ids_rank0_buffer.begin(), current_query_probed_ids_rank0_buffer.end(), probed_clusters_buffer_for_bcast_allranks.data()); }
        if (num_probed_to_bcast > 0) MPI_Bcast(probed_clusters_buffer_for_bcast_allranks.data(), num_probed_to_bcast, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        vector<uint32_t> clusters_for_my_rank_to_search;
        if (num_probed_to_bcast > 0) {
            for (size_t cluster_idx_in_list = 0; cluster_idx_in_list < num_probed_to_bcast; ++cluster_idx_in_list) {
                if (cluster_idx_in_list % world_size == (size_t)world_rank) {
                    clusters_for_my_rank_to_search.push_back(probed_clusters_buffer_for_bcast_allranks[cluster_idx_in_list]);
                }
            }
        }

        priority_queue<pair<float, uint32_t>> local_pq_results_for_this_query;
        if (!clusters_for_my_rank_to_search.empty()) {
            ivf_search_index.search_clusters_locally_with_omp( // 调用OMP版本
                current_query_buffer_allranks.data(),
                k_final,
                clusters_for_my_rank_to_search,
                //true, // 对于策略一，data_source_is_global 总是 true
                local_pq_results_for_this_query,
                omp_threads_per_mpi_process,
                i); // 传递查询索引用于可能的调试
        }

        std::fill(local_top_k_distances_sendbuf.begin(), local_top_k_distances_sendbuf.end(), std::numeric_limits<float>::max());
        std::fill(local_top_k_ids_sendbuf.begin(), local_top_k_ids_sendbuf.end(), std::numeric_limits<uint32_t>::max());
        vector<pair<float, uint32_t>> temp_sorted_local_results_vec;
        while(!local_pq_results_for_this_query.empty()){ temp_sorted_local_results_vec.push_back(local_pq_results_for_this_query.top()); local_pq_results_for_this_query.pop(); }
        std::reverse(temp_sorted_local_results_vec.begin(), temp_sorted_local_results_vec.end());
        for(size_t j=0; j < temp_sorted_local_results_vec.size() && j < k_final; ++j){
            local_top_k_distances_sendbuf[j] = temp_sorted_local_results_vec[j].first;
            local_top_k_ids_sendbuf[j] = temp_sorted_local_results_vec[j].second;
        }

        MPI_Gather(local_top_k_distances_sendbuf.data(), k_final, MPI_FLOAT, (world_rank == 0) ? gathered_all_distances_recvbuf_rank0.data() : nullptr, k_final, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(local_top_k_ids_sendbuf.data(), k_final, MPI_UNSIGNED, (world_rank == 0) ? gathered_all_ids_recvbuf_rank0.data() : nullptr, k_final, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            priority_queue<pair<float, uint32_t>> final_global_pq_for_this_query;
            for (int r = 0; r < world_size; ++r) {
                for (size_t res_idx = 0; res_idx < k_final; ++res_idx) {
                    float dist = gathered_all_distances_recvbuf_rank0[r * k_final + res_idx];
                    uint32_t id = gathered_all_ids_recvbuf_rank0[r * k_final + res_idx];
                    if (id != std::numeric_limits<uint32_t>::max()) {
                        if (final_global_pq_for_this_query.size() < k_final) {
                            final_global_pq_for_this_query.push({dist, id});
                        } else if (dist < final_global_pq_for_this_query.top().first) {
                            final_global_pq_for_this_query.pop();
                            final_global_pq_for_this_query.push({dist, id});
                        }
                    }
                }
            }
            set<uint32_t> gt_set_for_this_query;
            for (size_t j = 0; j < k_final; ++j) {
                if ( (i * test_gt_d_global + j) < (test_number_global * test_gt_d_global) ) {
                     gt_set_for_this_query.insert(test_gt_global_rank0[j + i * test_gt_d_global]);
                }
            }
            size_t correct_hits_this_query = 0;
            vector<pair<float, uint32_t>> temp_final_sorted_results_vec;
            while(!final_global_pq_for_this_query.empty()) { temp_final_sorted_results_vec.push_back(final_global_pq_for_this_query.top()); final_global_pq_for_this_query.pop(); }
            std::reverse(temp_final_sorted_results_vec.begin(), temp_final_sorted_results_vec.end());
            for(size_t r_idx = 0; r_idx < k_final && r_idx < temp_final_sorted_results_vec.size(); ++r_idx) {
                if(gt_set_for_this_query.count(temp_final_sorted_results_vec[r_idx].second)){
                    correct_hits_this_query++;
                }
            }
            float recall_val_this_query;
            if (k_final == 0) recall_val_this_query = 1.0f;
            else if (gt_set_for_this_query.empty() && k_final > 0) recall_val_this_query = (correct_hits_this_query == 0) ? 1.0f : 0.0f;
            else recall_val_this_query = static_cast<float>(correct_hits_this_query) / k_final;
            results_collector_rank0_final[i].recall = recall_val_this_query;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        double total_processing_end_time_seconds_rank0 = MPI_Wtime();
        double overall_duration_seconds = total_processing_end_time_seconds_rank0 - total_query_processing_time_seconds_rank0_overall;
        double avg_latency_per_query_us = (test_number_to_run > 0) ? (overall_duration_seconds * 1e6) / test_number_to_run : 0;
        for(size_t i=0; i<test_number_to_run; ++i) results_collector_rank0_final[i].latency_us = static_cast<long long>(avg_latency_per_query_us);

        cout << "\n--- IVF MPI+OpenMP (策略一, 内存重排, 簇内OMP并行) 最终测试结果 ---" << endl;
        if (test_number_to_run > 0) {
            float avg_recall_final = 0; for(const auto& res : results_collector_rank0_final) avg_recall_final += res.recall;
            cout << "  参数: MPI进程数=" << world_size << ", OMP线程数/进程=" << omp_threads_per_mpi_process
                 << ", 簇数量=" << num_ivf_clusters << ", nprobe=" << nprobe_param << ", k=" << k_final << endl;
            cout << "  平均召回率 Recall@" << k_final << ": " << avg_recall_final / test_number_to_run << endl;
            cout << "  平均查询延迟 (us): " << avg_latency_per_query_us << endl;
        }
        cout << "------------------------------------" << endl;
    }

    if (world_rank == 0) {
        delete[] test_query_global_rank0; delete[] test_gt_global_rank0;
        if (base_data_for_build_rank0_main) delete[] base_data_for_build_rank0_main;
        // global_base_pq_codes_rank0_source 在这个版本中没有被使用和new，所以不需要delete
    }
    MPI_Finalize();
    return 0;
}
