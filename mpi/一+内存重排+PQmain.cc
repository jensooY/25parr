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
#include <cstring> // For memcpy

// --- 自定义头文件 ---
#include "simd_distance.h" // 包含 compute_inner_product_distance_neon_optimized
#include "kmeans.h"      // 包含 run_kmeans
#include "ivf_mpi.h"     // 包含 IVFIndex_MPI 类的最新定义
#include "pq_utils.h"   // 包含 PQCodebook, train_pq, encode_pq, compute_inner_product_lut_neon, approximate_inner_product_adc

using namespace std;

// --- 数据加载函数 ---
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d, int current_rank_for_print) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        if (current_rank_for_print == 0) cerr << "错误：无法打开数据文件 " << data_path << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return nullptr;
    }
    int temp_n, temp_d;
    fin.read(reinterpret_cast<char*>(&temp_n), 4);
    fin.read(reinterpret_cast<char*>(&temp_d), 4);
    n = static_cast<size_t>(temp_n);
    d = static_cast<size_t>(temp_d);

    T* data = new T[n * d];
    size_t sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(reinterpret_cast<char*>(data + i * d), (long long)d * sz);
    }
    fin.close();

    if (current_rank_for_print == 0) {
        std::cerr<<"加载数据 "<<data_path<<"\n";
        std::cerr<<"维度: "<<d<<"  数量:"<<n<<"  每元素大小:"<<sizeof(T)<<"\n";
    }
    return data;
}

// --- 查询结果结构体 ---
struct SearchResult {
    float recall;
    int64_t latency_us;
};

// --- main 函数入口 ---
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // --- A. 全局参数和数据加载 (仅Rank 0) ---
    if (world_rank == 0) {
        cout << "--- [main_mpi_s1_reorder_pq.cc] IVF MPI 测试 ---" << endl;
        cout << "--- 策略: 策略一 (广播) + 内存重排 + IVF-PQ (MPI ADC + 串行Rerank) ---" << endl;
        cout << "MPI 总进程数: " << world_size << endl;
    }

    size_t test_number_global = 0, base_number_global = 0, test_gt_d_global = 0, vecdim_global = 0;
    float* test_query_global_rank0 = nullptr;
    int* test_gt_global_rank0 = nullptr;
    float* base_data_for_build_rank0_main = nullptr; // Rank 0 持有原始base数据, 用于构建和精确重排

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

    // --- B. 实验配置 ---
    size_t test_number_to_run = std::min(test_number_global, (size_t)2000);
    const size_t k_final = 10;
    const size_t nk_for_reranking = 1500;
    size_t num_ivf_clusters = 2048;
    int kmeans_max_iterations = 20;
    size_t nprobe_param = 20;
    // MPI进程数 (用于ADC召回) 由 world_size 决定

    bool enable_memory_reordering = true; // 固定为启用内存重排

    if (world_rank == 0) {
        cout << "配置: 内存重排=启用, 数据划分=策略一(广播)" << endl;
        cout << "IVF参数: 簇数=" << num_ivf_clusters << ", nprobe=" << nprobe_param << ", 最终k=" << k_final << endl;
        cout << "PQ参数: ADC召回Nk=" << nk_for_reranking << ", MPI进程数(用于ADC)=" << world_size << endl;
    }

    // --- C. 全局PQ码本训练和数据编码 (Rank 0) ---
    PQCodebook global_pq_codebook_obj_main; // 所有进程都有这个对象，但只在rank0填充，其他通过指针接收广播内容
    const PQCodebook* pq_codebook_to_pass_to_ivfindex = &global_pq_codebook_obj_main;
    uint8_t* pq_codes_for_ivf_constructor_rank0 = nullptr; // Rank 0 的PQ编码数据源

    if (world_rank == 0) {
        global_pq_codebook_obj_main.params.M = 4; global_pq_codebook_obj_main.params.Ks = 256;
        global_pq_codebook_obj_main.params.D = vecdim_global;
        global_pq_codebook_obj_main.params.D_sub = vecdim_global / global_pq_codebook_obj_main.params.M;
        global_pq_codebook_obj_main.params.n_train = base_number_global;
        global_pq_codebook_obj_main.params.kmeans_iters = 20;
        train_pq(base_data_for_build_rank0_main, global_pq_codebook_obj_main); // 使用原始数据训练
        pq_codes_for_ivf_constructor_rank0 = new uint8_t[base_number_global * global_pq_codebook_obj_main.params.M];
        for (size_t i = 0; i < base_number_global; ++i) {
            encode_pq(base_data_for_build_rank0_main + i * vecdim_global, global_pq_codebook_obj_main,
                      pq_codes_for_ivf_constructor_rank0 + i * global_pq_codebook_obj_main.params.M);
        }
        cout << "Rank 0: 全局PQ码本训练和数据编码完成。" << endl;
    }

    // --- D. 创建IVFIndex_MPI对象 ---
    IVFIndex_MPI ivf_search_index(world_rank, world_size,
                                  base_data_for_build_rank0_main,       // Rank 0 的原始浮点base (用于KMeans源)
                                  pq_codes_for_ivf_constructor_rank0, // Rank 0 的全局PQ编码 (用于后续复制到广播区)
                                  pq_codebook_to_pass_to_ivfindex,    // 所有进程的码本指针
                                  base_number_global, vecdim_global,
                                  enable_memory_reordering);

    // --- E. Rank 0 构建索引 (KMeans + 内存重排) ---
    if (world_rank == 0) {
        if (!ivf_search_index.rank0_build_index_with_optional_reordering(num_ivf_clusters, kmeans_max_iterations)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // --- F. 所有进程接收广播的IVF核心数据 (质心、重排后倒排表、映射表) ---
    if (!ivf_search_index.receive_bcast_index_data_after_reordering(MPI_COMM_WORLD)) { MPI_Abort(MPI_COMM_WORLD, 1); }

    // --- G. 所有进程接收广播的PQ核心数据 ---
    if (world_rank == 0) {
        // Rank 0 将其编码好的PQ数据复制到它自己的IVFIndex实例的广播缓冲区
        if (ivf_search_index.base_pq_codes_global_all_processes_ && ivf_search_index.base_pq_codes_global_rank0_source_for_bcast_ &&
            base_number_global > 0 && ivf_search_index.pq_codebook_global_ && ivf_search_index.pq_codebook_global_->params.M > 0) {
            memcpy(ivf_search_index.base_pq_codes_global_all_processes_,
                   ivf_search_index.base_pq_codes_global_rank0_source_for_bcast_,
                   base_number_global * ivf_search_index.pq_codebook_global_->params.M * sizeof(uint8_t));
        } else if (base_number_global > 0 && ivf_search_index.pq_codebook_global_ && ivf_search_index.pq_codebook_global_->params.M > 0 ) {
             cerr << "Rank 0: 错误！PQ编码数据广播前，源或目标缓冲区未准备好。" << endl; MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (!ivf_search_index.receive_bcast_pq_data(MPI_COMM_WORLD)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    if (world_rank == 0) cout << "所有进程已接收PQ核心数据。" << endl;

    // --- H. 策略一：广播【重排后】的基础浮点数据 ---
    if (world_rank == 0) {
        const float* source_for_bcast_s1 = ivf_search_index.base_data_reordered_rank0_owner_;
        if (source_for_bcast_s1 && ivf_search_index.base_data_global_all_) {
            memcpy(ivf_search_index.base_data_global_all_, source_for_bcast_s1, base_number_global * vecdim_global * sizeof(float));
        } else if (base_number_global > 0) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (!ivf_search_index.receive_bcast_base_data_strategy1(MPI_COMM_WORLD)) { MPI_Abort(MPI_COMM_WORLD, 1); }
    if (world_rank == 0) cout << "策略一：重排后基础浮点数据广播完成。" << endl;

    // 注意：base_data_for_build_rank0_main 需要保留到精确重排阶段结束（如果rank0执行重排）
    // pq_codes_for_ivf_constructor_rank0 可以在PQ数据广播后由rank0释放

    // --- I. 在线查询与评估 ---
    vector<SearchResult> results_collector_rank0_final(test_number_to_run);
    vector<float> current_query_buffer_allranks(vecdim_global);
    vector<uint32_t> current_query_probed_ids_rank0_main_buffer;
    vector<uint32_t> probed_clusters_buffer_for_bcast_allranks_main;

    vector<float> local_top_nk_adc_dist_sendbuf(nk_for_reranking, std::numeric_limits<float>::max());
    vector<uint32_t> local_top_nk_adc_ids_sendbuf(nk_for_reranking, std::numeric_limits<uint32_t>::max());
    vector<float> gathered_all_adc_distances_recvbuf_rank0;
    vector<uint32_t> gathered_all_adc_ids_recvbuf_rank0;
    if (world_rank == 0) {
        gathered_all_adc_distances_recvbuf_rank0.resize(nk_for_reranking * world_size);
        gathered_all_adc_ids_recvbuf_rank0.resize(nk_for_reranking * world_size);
    }

    long long total_lut_us_r0_accum = 0;
    long long total_adc_recall_phase_us_r0_accum = 0;
    long long total_extract_ids_us_r0_accum = 0;
    long long total_rerank_serial_us_r0_accum = 0;
    long long total_sort_us_r0_accum = 0;
    double total_query_time_seconds_r0_overall = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) { total_query_time_seconds_r0_overall = MPI_Wtime(); }

    for (size_t i = 0; i < test_number_to_run; ++i) {
        std::vector<float> lut_for_this_query_allranks(ivf_search_index.pq_codebook_global_->params.M * ivf_search_index.pq_codebook_global_->params.Ks);
        chrono::high_resolution_clock::time_point t_chrono_stage_start_rank0; // Rank0的chrono计时器

        if (world_rank == 0) {
            t_chrono_stage_start_rank0 = chrono::high_resolution_clock::now();
            std::copy(test_query_global_rank0 + i * vecdim_global, test_query_global_rank0 + (i + 1) * vecdim_global, current_query_buffer_allranks.data());
            compute_inner_product_lut_neon(current_query_buffer_allranks.data(), *(ivf_search_index.pq_codebook_global_), lut_for_this_query_allranks.data());
            total_lut_us_r0_accum += chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t_chrono_stage_start_rank0).count();
        }
        MPI_Bcast(current_query_buffer_allranks.data(), vecdim_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lut_for_this_query_allranks.data(), lut_for_this_query_allranks.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Rank 0 定位候选簇
        current_query_probed_ids_rank0_main_buffer.clear();
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
            current_query_probed_ids_rank0_main_buffer.reserve(nprobe_param);
            for (size_t idx = 0; idx < nprobe_param && idx < cluster_distances_q_main.size(); ++idx) {
                current_query_probed_ids_rank0_main_buffer.push_back(cluster_distances_q_main[idx].second);
            }
        }
        size_t num_probed_to_bcast = (world_rank == 0) ? current_query_probed_ids_rank0_main_buffer.size() : 0;
        MPI_Bcast(&num_probed_to_bcast, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        probed_clusters_buffer_for_bcast_allranks_main.resize(num_probed_to_bcast);
        if (world_rank == 0 && num_probed_to_bcast > 0) { std::copy(current_query_probed_ids_rank0_main_buffer.begin(), current_query_probed_ids_rank0_main_buffer.end(), probed_clusters_buffer_for_bcast_allranks_main.data()); }
        if (num_probed_to_bcast > 0) MPI_Bcast(probed_clusters_buffer_for_bcast_allranks_main.data(), num_probed_to_bcast, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        vector<uint32_t> clusters_for_my_rank_to_search;
        if (num_probed_to_bcast > 0) {
            for (size_t cluster_idx_in_list = 0; cluster_idx_in_list < num_probed_to_bcast; ++cluster_idx_in_list) {
                if (cluster_idx_in_list % world_size == (size_t)world_rank) {
                    clusters_for_my_rank_to_search.push_back(probed_clusters_buffer_for_bcast_allranks_main[cluster_idx_in_list]);
                }
            }
        }

        // --- 计时ADC召回并行阶段（使用MPI_Wtime，由rank0记录墙上时间） ---
        MPI_Barrier(MPI_COMM_WORLD);
        double adc_recall_phase_start_time_mpi_iter = 0;
        if (world_rank == 0) { adc_recall_phase_start_time_mpi_iter = MPI_Wtime(); }

        priority_queue<pair<float, uint32_t>> local_adc_pq_results;
        if (!clusters_for_my_rank_to_search.empty()) {
            ivf_search_index.search_clusters_locally_pq_adc_for_rank(
                lut_for_this_query_allranks.data(), nk_for_reranking, clusters_for_my_rank_to_search,
                true, local_adc_pq_results, i);
        }
        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程完成本地搜索
        if(world_rank == 0) {
            total_sort_us_r0_accum += static_cast<long long>((MPI_Wtime() - adc_recall_phase_start_time_mpi_iter) * 1e6);
        }

        std::fill(local_top_nk_adc_dist_sendbuf.begin(), local_top_nk_adc_dist_sendbuf.end(), std::numeric_limits<float>::max());
        std::fill(local_top_nk_adc_ids_sendbuf.begin(), local_top_nk_adc_ids_sendbuf.end(), std::numeric_limits<uint32_t>::max());
        vector<pair<float, uint32_t>> temp_sorted_local_adc_vec;
        temp_sorted_local_adc_vec.reserve(local_adc_pq_results.size());
        while(!local_adc_pq_results.empty()){ temp_sorted_local_adc_vec.push_back(local_adc_pq_results.top()); local_adc_pq_results.pop(); }
        std::reverse(temp_sorted_local_adc_vec.begin(), temp_sorted_local_adc_vec.end());
        for(size_t j=0; j < temp_sorted_local_adc_vec.size() && j < nk_for_reranking; ++j){
            local_top_nk_adc_dist_sendbuf[j] = temp_sorted_local_adc_vec[j].first;
            local_top_nk_adc_ids_sendbuf[j] = temp_sorted_local_adc_vec[j].second;
        }

        MPI_Gather(local_top_nk_adc_dist_sendbuf.data(), nk_for_reranking, MPI_FLOAT, (world_rank == 0) ? gathered_all_adc_distances_recvbuf_rank0.data() : nullptr, nk_for_reranking, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(local_top_nk_adc_ids_sendbuf.data(), nk_for_reranking, MPI_UNSIGNED, (world_rank == 0) ? gathered_all_adc_ids_recvbuf_rank0.data() : nullptr, nk_for_reranking, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            t_chrono_stage_start_rank0 = chrono::high_resolution_clock::now(); // 提取ID开始
            priority_queue<pair<float, uint32_t>> globally_merged_adc_pq_rank0;
            for (int r = 0; r < world_size; ++r) {
                for (size_t res_idx = 0; res_idx < nk_for_reranking; ++res_idx) {
                    float adc_dist = gathered_all_adc_distances_recvbuf_rank0[r * nk_for_reranking + res_idx];
                    uint32_t original_id = gathered_all_adc_ids_recvbuf_rank0[r * nk_for_reranking + res_idx];
                    if (original_id != std::numeric_limits<uint32_t>::max()) {
                        if (globally_merged_adc_pq_rank0.size() < nk_for_reranking) {
                            globally_merged_adc_pq_rank0.push({adc_dist, original_id});
                        } else if (adc_dist < globally_merged_adc_pq_rank0.top().first) {
                            globally_merged_adc_pq_rank0.pop();
                            globally_merged_adc_pq_rank0.push({adc_dist, original_id});
                        }
                    }
                }
            }
            vector<uint32_t> ids_for_final_rerank_vec;
            ids_for_final_rerank_vec.reserve(globally_merged_adc_pq_rank0.size());
            while(!globally_merged_adc_pq_rank0.empty()){ ids_for_final_rerank_vec.push_back(globally_merged_adc_pq_rank0.top().second); globally_merged_adc_pq_rank0.pop(); }
            std::reverse(ids_for_final_rerank_vec.begin(), ids_for_final_rerank_vec.end());
            total_extract_ids_us_r0_accum += chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t_chrono_stage_start_rank0).count();

            t_chrono_stage_start_rank0 = chrono::high_resolution_clock::now(); // 串行重排开始
            vector<pair<float, uint32_t>> exact_reranked_list;
            exact_reranked_list.reserve(ids_for_final_rerank_vec.size());
            for (uint32_t original_id_to_rerank : ids_for_final_rerank_vec) {
                if (original_id_to_rerank < base_number_global) {
                    // 【使用最初加载的base数据，它在rank0上应该仍然是base_data_for_build_rank0_main】
                    const float* candidate_float_vector = base_data_for_build_rank0_main + original_id_to_rerank * vecdim_global;
                    float exact_distance = compute_inner_product_distance_neon_optimized(current_query_buffer_allranks.data(), candidate_float_vector, vecdim_global);
                    exact_reranked_list.push_back({exact_distance, original_id_to_rerank});
                }
            }
            total_rerank_serial_us_r0_accum += chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t_chrono_stage_start_rank0).count();

            t_chrono_stage_start_rank0 = chrono::high_resolution_clock::now(); // 最终排序开始
            sort(exact_reranked_list.begin(), exact_reranked_list.end());
            total_sort_us_r0_accum += chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t_chrono_stage_start_rank0).count();

            set<uint32_t> gt_set_for_this_query;
            for (size_t j = 0; j < k_final; ++j) {
                if ( (i * test_gt_d_global + j) < (test_number_global * test_gt_d_global) ) {
                     gt_set_for_this_query.insert(test_gt_global_rank0[j + i * test_gt_d_global]);
                }
            }
            size_t correct_hits_this_query = 0;
            for(size_t r_idx = 0; r_idx < k_final && r_idx < exact_reranked_list.size(); ++r_idx) {
                if(gt_set_for_this_query.count(exact_reranked_list[r_idx].second)){
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
        double overall_duration_seconds = total_processing_end_time_seconds_rank0 - total_query_time_seconds_r0_overall;
        double avg_latency_per_query_us = (test_number_to_run > 0) ? (overall_duration_seconds * 1e6) / test_number_to_run : 0;
        for(size_t i=0; i<test_number_to_run; ++i) results_collector_rank0_final[i].latency_us = static_cast<long long>(avg_latency_per_query_us);

        cout << "\n--- IVF-PQ MPI (策略一, 内存重排, ADC并行, 串行Rerank) 最终测试结果 ---" << endl;
        if (test_number_to_run > 0) {
            float avg_recall_final = 0; for(const auto& res : results_collector_rank0_final) avg_recall_final += res.recall;
            cout << "  平均召回率 Recall@" << k_final << ": " << avg_recall_final / test_number_to_run << endl;
            cout << "  平均查询总延迟 (us):               " << avg_latency_per_query_us << endl;
            cout << "  平均LUT预计算阶段延迟 (us):      " << (double)total_lut_us_r0_accum / test_number_to_run << endl;
            cout << "  平均ADC召回MPI并行阶段延迟 (us): " << (double)total_sort_us_r0_accum / test_number_to_run << endl;
            cout << "  平均提取ID阶段延迟 (us):           " << (double)total_extract_ids_us_r0_accum / test_number_to_run << endl;
            cout << "  平均串行重排计算阶段延迟 (us):   " << (double)total_rerank_serial_us_r0_accum / test_number_to_run << endl;
            cout << "  平均最终排序阶段延迟 (us):       " << (double)total_sort_us_r0_accum / test_number_to_run << endl;
        }
        cout << "------------------------------------" << endl;
    }

    // --- 清理 ---
    if (world_rank == 0) {
        delete[] test_query_global_rank0;
        delete[] test_gt_global_rank0;
        if (base_data_for_build_rank0_main != nullptr) delete[] base_data_for_build_rank0_main;
        if (pq_codes_for_ivf_constructor_rank0 != nullptr) delete[] pq_codes_for_ivf_constructor_rank0;
    }
    MPI_Finalize();
    return 0;
}
