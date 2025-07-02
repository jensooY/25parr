// in gpu_ivf.cu (完整修正版 - 使用朴素内核)

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <queue>
#include <chrono>
#include "gpu_ivf.h"

// ====================================================================
//                             CUDA 核函数
// =============================================

// 使用一个最简单、最直观的矩阵乘法核函数来确保逻辑正确性
// 计算 R(m, n) = Q(m, d) * B_T(d, n)
__global__ void naive_matrix_mult_kernel(const float* Q, const float* B, float* R, 
                                         int m, int n, int d) {
    // 每个线程计算 R 矩阵中的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            // Q[row][i] * B_T[i][col]  (B_T[i][col] 等于 B[col][i])
            sum += Q[row * d + i] * B[col * d + i];
        }
        R[row * n + col] = 1.0f - sum; // 计算IP距离
    }
}

// 核函数2: 查找 nprobe 个最近的质心 (这个函数逻辑是正确的，保持不变)
__global__ void find_nprobe_kernel(const float* d_dist_matrix, int* d_candidate_lists, int query_number, int nlist, int nprobe) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_number) return;
    float top_dists[32]; int top_indices[32];
    for (int i = 0; i < nprobe; ++i) { top_dists[i] = 1e38f; top_indices[i] = -1; }
    for (int col = 0; col < nlist; ++col) {
        float dist = d_dist_matrix[row * nlist + col];
        float max_val = top_dists[0]; int max_idx = 0;
        for(int i = 1; i < nprobe; ++i) { if (top_dists[i] > max_val) { max_val = top_dists[i]; max_idx = i; } }
        if (dist < max_val) { top_dists[max_idx] = dist; top_indices[max_idx] = col; }
    }
    for (int i = 0; i < nprobe; ++i) { d_candidate_lists[row * nprobe + i] = top_indices[i]; }
}

// ====================================================================
//                       C++ 接口函数 (Host端)
// ====================================================================
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_ivf_batch_search(const IVFIndex& index, const float* all_queries, size_t query_number, size_t nprobe, size_t k) {
    printf("\n--- 开始执行【混合模式-朴素内核版】GPU-IVF批量搜索 ---\n");
    const size_t vecdim = index.vecdim;
    const size_t nlist = index.nlist;

    // GPU 部分: 粗筛
    printf("  [GPU] 正在执行粗筛阶段...\n");
    auto gpu_start = std::chrono::high_resolution_clock::now();
    float *d_queries, *d_centroids, *d_dist_matrix; int *d_candidate_lists;
    cudaMalloc(&d_queries, query_number * vecdim * sizeof(float));
    cudaMalloc(&d_centroids, nlist * vecdim * sizeof(float));
    cudaMalloc(&d_dist_matrix, query_number * nlist * sizeof(float));
    cudaMalloc(&d_candidate_lists, query_number * nprobe * sizeof(int));
    cudaMemcpy(d_queries, all_queries, query_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, index.centroids.data(), nlist * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    
    // 【【【 核心修正 】】】
    // 调用朴素内核，使用2D线程块
    dim3 threadsPerBlock_mm(16, 16); 
    dim3 blocksPerGrid_mm(
        (nlist + threadsPerBlock_mm.x - 1) / threadsPerBlock_mm.x, 
        (query_number + threadsPerBlock_mm.y - 1) / threadsPerBlock_mm.y
    );
    naive_matrix_mult_kernel<<<blocksPerGrid_mm, threadsPerBlock_mm>>>(
        d_queries, d_centroids, d_dist_matrix, query_number, nlist, vecdim
    );

    int threadsPerBlock_np = 256; int blocksPerGrid_np = (query_number + threadsPerBlock_np - 1) / threadsPerBlock_np;
    find_nprobe_kernel<<<blocksPerGrid_np, threadsPerBlock_np>>>(d_dist_matrix, d_candidate_lists, query_number, nlist, nprobe);
    std::vector<int> h_candidate_lists(query_number * nprobe);
    cudaMemcpy(h_candidate_lists.data(), d_candidate_lists, query_number * nprobe * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_queries); cudaFree(d_centroids); cudaFree(d_dist_matrix); cudaFree(d_candidate_lists);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    long gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    printf("  [GPU] 粗筛阶段完成，耗时: %ld us\n", gpu_duration);

    // CPU 部分: 精排 (这部分代码保持不变)
    printf("  [CPU] 正在执行精排阶段...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> all_final_results(query_number);
    auto ip_distance_cpu = [&](const float* v1, const float* v2, size_t dim) { float sum = 0.0f; for (size_t i = 0; i < dim; ++i) sum += v1[i] * v2[i]; return 1.0f - sum; };
    for (size_t i = 0; i < query_number; ++i) {
        const float* current_query = all_queries + i * vecdim;
        for (size_t j = 0; j < nprobe; ++j) {
            int centroid_id = h_candidate_lists[i * nprobe + j];
            if (centroid_id < 0) continue;
            size_t start_offset = index.list_offsets.at(centroid_id);
            size_t end_offset = index.list_offsets.at(centroid_id + 1);
            for (size_t l = start_offset; l < end_offset; ++l) {
                const float* vector_ptr = &index.reordered_base_data[l * vecdim];
                float dist = ip_distance_cpu(current_query, vector_ptr, vecdim);
                uint32_t original_id = index.reordered_original_ids.at(l);
                if (all_final_results[i].size() < k) { all_final_results[i].push({dist, original_id});
                } else if (dist < all_final_results[i].top().first) { all_final_results[i].pop(); all_final_results[i].push({dist, original_id}); }
            }
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    long cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    printf("  [CPU] 精排阶段完成，耗时: %ld us\n", cpu_duration);
    
    return all_final_results;
}



// 【新核函数 1: 精排距离计算】
// 每个Block负责一个Query，Block内的所有Thread协同计算该Query需要的所有距离
__global__ void fine_search_kernel(
    const float* d_queries,                 // [query_num, vecdim] 所有查询向量
    const float* d_reordered_base,          // [N, vecdim] 重排后的基础数据库
    const uint32_t* d_reordered_ids,        // [N] 重排后的原始ID
    const int* d_candidate_lists,           // [query_num, nprobe] 粗筛结果，存的是 list_id
    const size_t* d_list_offsets,           // [nlist + 1] 倒排列表偏移量
    FloatUIntPair* d_intermediate_results,  // 中间结果存放处
    const int* d_query_result_offsets,      // [query_num + 1] 每个查询在中间结果中的写入起始位置
    int query_number, int vecdim, int nprobe
) {
    // 每个Block处理一个查询
    int query_idx = blockIdx.x;
    if (query_idx >= query_number) return;

    // 将当前查询向量加载到共享内存中，供Block内所有线程复用
    // extern __shared__ float query_shmem[]; // 动态共享内存，在调用时指定大小
    // 修正：对于固定vecdim，使用静态共享内存更简单清晰
    __shared__ float query_shmem[96]; // 假设 vecdim=96，请根据你的vecdim修改

    int tid = threadIdx.x;
    for (int i = tid; i < vecdim; i += blockDim.x) {
        query_shmem[i] = d_queries[query_idx * vecdim + i];
    }
    __syncthreads();

    // 获取当前查询的中间结果写入偏移量
    int result_base_offset = d_query_result_offsets[query_idx];
    
    // 使用原子操作来为Block内的线程分配写入位置，确保写入不冲突
    __shared__ int result_write_counter;
    if (tid == 0) {
        result_write_counter = 0;
    }
    __syncthreads();

    // 遍历该查询的 nprobe 个候选列表
    for (int i = 0; i < nprobe; ++i) {
        int list_id = d_candidate_lists[query_idx * nprobe + i];
        if (list_id < 0) continue; // 无效列表ID

        size_t start_offset = d_list_offsets[list_id];
        size_t end_offset = d_list_offsets[list_id + 1];
        size_t list_size = end_offset - start_offset;

        // Grid-Stride Loop: Block内的线程协同处理这个列表
        for (int j = tid; j < list_size; j += blockDim.x) {
            size_t vector_reordered_idx = start_offset + j;
            const float* vector_ptr = &d_reordered_base[vector_reordered_idx * vecdim];
            
            // 计算距离
            float sum = 0.0f;
            for (int d = 0; d < vecdim; ++d) {
                sum += query_shmem[d] * vector_ptr[d];
            }
            float dist = 1.0f - sum;

            // 获取写入位置并写入结果
            int write_idx = atomicAdd(&result_write_counter, 1);
            d_intermediate_results[result_base_offset + write_idx] = {dist, d_reordered_ids[vector_reordered_idx]};
        }
    }
}


// 【新核函数 2: 并行Top-K筛选】
// 每个Thread负责一个Query，从中间结果中筛选出Top-K
__global__ void find_topk_ivf_kernel(
    const FloatUIntPair* d_intermediate_results, // [total_results] 所有中间结果
    const int* d_query_result_offsets,           // [query_num + 1] 每个查询的结果偏移
    const int* d_query_result_counts,            // [query_num] 每个查询的结果数量
    FloatUIntPair* d_final_topk_results,         // [query_num, k] 最终输出
    int query_number, int k
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= query_number) return;

    // 在寄存器中模拟一个最大堆 (用于找最小值)
    FloatUIntPair local_topk[10]; // 假设 k <= 10，请根据你的k修改
    for (int i = 0; i < k; ++i) {
        local_topk[i] = {2.0f, 0}; // 初始化为最大可能的IP距离
    }

    int result_start = d_query_result_offsets[query_idx];
    int result_count = d_query_result_counts[query_idx];

    // 遍历这个查询对应的所有中间结果
    for (int i = 0; i < result_count; ++i) {
        FloatUIntPair candidate = d_intermediate_results[result_start + i];
        
        // 找到当前 topk 中最大的值
        float max_val = local_topk[0].value;
        int max_idx = 0;
        for(int j = 1; j < k; ++j) {
            if (local_topk[j].value > max_val) {
                max_val = local_topk[j].value;
                max_idx = j;
            }
        }
        // 如果新距离更小，则替换掉最大的那个
        if (candidate.value < max_val) {
            local_topk[max_idx] = candidate;
        }
    }

    // 将最终的Top-K结果写回全局内存
    for (int i = 0; i < k; ++i) {
        d_final_topk_results[query_idx * k + i] = local_topk[i];
    }
}


// ====================================================================
//         【【【 最终实现版本 】】】
// ====================================================================
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_ivf_batch_search_full_gpu(const IVFIndex& index, const float* all_queries, 
                              size_t query_number, size_t nprobe, size_t k,
                              void* d_workspace) // <-- 接收工作区指针
{
    printf("\n--- 开始执行【完全GPU模式-工作区版】GPU-IVF批量搜索 ---\n");
    const size_t vecdim = index.vecdim;
    const size_t nlist = index.nlist;

    // --- 0. 创建CUDA事件用于计时 ---
    cudaEvent_t start, stop;
    float elapsedTime_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 1. 工作区分配 (Assign) ---
    // 从预先分配好的 d_workspace 中“切”出所有需要的GPU内存指针
    // 这里的分配逻辑，必须和 main.cpp 中的规划逻辑完全一致！
    
    char* workspace_ptr = (char*)d_workspace;
    auto align_up = [&](size_t size) {
        size_t alignment = 256;
        return (size + alignment - 1) & ~(alignment - 1);
    };

    // 【【【只分配大小固定的变量】】】
    float* d_queries = (float*)workspace_ptr;   workspace_ptr += align_up(query_number * vecdim * sizeof(float));
    float* d_centroids = (float*)workspace_ptr; workspace_ptr += align_up(nlist * vecdim * sizeof(float));
    float* d_dist_matrix = (float*)workspace_ptr; workspace_ptr += align_up(query_number * nlist * sizeof(float));
    int* d_candidate_lists = (int*)workspace_ptr; workspace_ptr += align_up(query_number * nprobe * sizeof(int));
    float* d_reordered_base = (float*)workspace_ptr;   workspace_ptr += align_up(index.reordered_base_data.size() * sizeof(float));
    uint32_t* d_reordered_ids = (uint32_t*)workspace_ptr; workspace_ptr += align_up(index.reordered_original_ids.size() * sizeof(uint32_t));
    size_t* d_list_offsets = (size_t*)workspace_ptr;   workspace_ptr += align_up(index.list_offsets.size() * sizeof(size_t));
    int* d_query_result_offsets = (int*)workspace_ptr; workspace_ptr += align_up((query_number + 1) * sizeof(int));
    int* d_query_result_counts = (int*)workspace_ptr;  workspace_ptr += align_up(query_number * sizeof(int));
    FloatUIntPair* d_final_topk_results = (FloatUIntPair*)workspace_ptr;
    
    // 【【【为动态大小的变量单独 Malloc】】】
    FloatUIntPair* d_intermediate_results = nullptr;


    // ===================================================================
    //                        阶段计时开始
    // ===================================================================

    // --- 2. 粗筛阶段 ---
    cudaEventRecord(start); // 开始计时: 粗筛数据拷贝
    cudaMemcpy(d_queries, all_queries, query_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, index.centroids.data(), nlist * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [1.1] 粗筛 - 数据拷贝: \t\t%f ms\n", elapsedTime_ms);

    cudaEventRecord(start); // 开始计时: 粗筛距离计算
    // 使用性能最佳的 Naive Kernel
    dim3 threadsPerBlock_mm(16, 16); 
    dim3 blocksPerGrid_mm((nlist + 15) / 16, (query_number + 15) / 16);
    naive_matrix_mult_kernel<<<blocksPerGrid_mm, threadsPerBlock_mm>>>(d_queries, d_centroids, d_dist_matrix, query_number, nlist, vecdim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [1.2] 粗筛 - 距离计算内核: \t%f ms\n", elapsedTime_ms);

    cudaEventRecord(start); // 开始计时: nprobe选择
    int threadsPerBlock_np = 256; 
    int blocksPerGrid_np = (query_number + threadsPerBlock_np - 1) / threadsPerBlock_np;
    find_nprobe_kernel<<<blocksPerGrid_np, threadsPerBlock_np>>>(d_dist_matrix, d_candidate_lists, query_number, nlist, nprobe);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [1.3] 粗筛 - nprobe选择内核: \t%f ms\n", elapsedTime_ms);
    
    // --- 3. 规划阶段 ---
    cudaEventRecord(start); // 开始计时: 粗筛结果回传
    std::vector<int> h_candidate_lists(query_number * nprobe);
    cudaMemcpy(h_candidate_lists.data(), d_candidate_lists, query_number * nprobe * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [2.1] 规划 - 粗筛结果回传: \t%f ms\n", elapsedTime_ms);

    auto cpu_plan_start = std::chrono::high_resolution_clock::now();
    std::vector<int> h_query_result_offsets(query_number + 1, 0);
    std::vector<int> h_query_result_counts(query_number);
    int total_results_to_compute = 0;
    for (size_t i = 0; i < query_number; ++i) {
        h_query_result_offsets[i] = total_results_to_compute;
        int current_query_count = 0;
        for (size_t j = 0; j < nprobe; ++j) {
            int list_id = h_candidate_lists[i * nprobe + j];
            if (list_id >= 0 && list_id < nlist) {
                current_query_count += (index.list_offsets[list_id + 1] - index.list_offsets[list_id]);
            }
        }
        h_query_result_counts[i] = current_query_count;
        total_results_to_compute += current_query_count;
    }
    h_query_result_offsets[query_number] = total_results_to_compute;
    auto cpu_plan_end = std::chrono::high_resolution_clock::now();
    printf("  [2.2] 规划 - CPU计算: \t\t%ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(cpu_plan_end - cpu_plan_start).count());
    // 【【【 在这里添加 cudaMalloc 】】】
    cudaMalloc(&d_intermediate_results, total_results_to_compute * sizeof(FloatUIntPair));


    // --- 4. 精排阶段 ---
    cudaEventRecord(start); // 开始计时: 精排数据拷贝
    cudaMemcpy(d_reordered_base, index.reordered_base_data.data(), index.reordered_base_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reordered_ids, index.reordered_original_ids.data(), index.reordered_original_ids.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_list_offsets, index.list_offsets.data(), index.list_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_result_offsets, h_query_result_offsets.data(), (query_number + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_result_counts, h_query_result_counts.data(), query_number * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [3.1] 精排 - 数据拷贝: \t\t%f ms\n", elapsedTime_ms);

    cudaEventRecord(start); // 开始计时: 精排距离计算
    dim3 blocks_fine_search(query_number);
    dim3 threads_fine_search(256);
    fine_search_kernel<<<blocks_fine_search, threads_fine_search>>>(
        d_queries, d_reordered_base, d_reordered_ids, d_candidate_lists, d_list_offsets,
        d_intermediate_results, d_query_result_offsets, query_number, vecdim, nprobe);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [3.2] 精排 - 距离计算内核: \t%f ms\n", elapsedTime_ms);

    cudaEventRecord(start); // 开始计时: TopK选择
    dim3 threads_topk(256);
    dim3 blocks_topk((query_number + threads_topk.x - 1) / threads_topk.x);
    find_topk_ivf_kernel<<<blocks_topk, threads_topk>>>(
        d_intermediate_results, d_query_result_offsets, d_query_result_counts,
        d_final_topk_results, query_number, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [3.3] 精排 - TopK选择内核: \t%f ms\n", elapsedTime_ms);

    // --- 5. 结果回传 ---
    cudaEventRecord(start); 
    std::vector<FloatUIntPair> h_final_topk_results(query_number * k);
    cudaMemcpy(h_final_topk_results.data(), d_final_topk_results, query_number * k * sizeof(FloatUIntPair), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_ms, start, stop);
    printf("  [4.1] 结果回传: \t\t\t%f ms\n", elapsedTime_ms);
    
    // --- 6. CPU端后处理 ---
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> all_final_results(query_number);
    for(size_t i = 0; i < query_number; ++i) {
        for(size_t j = 0; j < k; ++j) {
            auto& res = h_final_topk_results[i * k + j];
            all_final_results[i].push({res.value, res.key});
        }
    }

    // --- 7. 销毁事件 ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 【【【 在这里添加 cudaFree 】】】
    cudaFree(d_intermediate_results);

    return all_final_results;
}
