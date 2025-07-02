#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_flat_scan.h" // 引入头文件
#include <chrono>
#define TILE_WIDTH 8
#include <iostream>

float ip_distance(const float* v1, const float* v2, size_t vecdim);

// 实现矩阵乘法的核函数：R(m, n) = Q(m, d) * B_T(d, n)
// B[col * d + i]实现B的转置
__global__ void matrix_mult_kernel(const float* Q, const float* B, float* R, int m, int n, int d) {
    // 计算当前线程负责的 R 矩阵的行(row)和列(col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查线程是否越界 (R矩阵的范围)
    if (row < m && col < n) {
        float sum = 0.0f;

        // 执行点积计算
        for (int i = 0; i < d; ++i) {
            // Q[row][i] * B[col][i] (因为B是未转置的，所以用col索引行)
            sum += Q[row * d + i] * B[col * d + i];
        }

        // 将结果写入 R 矩阵
        R[row * n + col] = 1.0f - sum; // 计算IP距离
    }
}

// 矩阵乘法的核函数，使用【共享内存】进行优化
// 最终修正版 - 矩阵乘法的核函数，使用【共享内存】进行优化
__global__ void matrix_mult_tiled_kernel(const float* Q, const float* B, float* R, 
                                         int m, int n, int d) {
    // 1. 声明共享内存
    __shared__ float Q_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    // 2. 计算线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // 3. 初始化累加器
    // 每个thread的私有变量
    float sum = 0.0f;

    // 4. 按块遍历维度d
    int num_tiles = (d + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        
        // --- 阶段A：协作从全局内存加载数据到共享内存 ---
        
        // 加载 Q_tile: 线程(ty, tx)负责加载 Q_tile[ty][tx]
        int q_load_col = tile_idx * TILE_WIDTH + tx;
        if (row < m && q_load_col < d) {
            Q_tile[ty][tx] = Q[row * d + q_load_col];
        } else {
            Q_tile[ty][tx] = 0.0f;
        }

        // 加载 B_tile: 线程(ty, tx)负责加载 B_tile[ty][tx]
        // 这一步是之前所有错误的核心，请仔细看
        int b_load_row = blockIdx.x * TILE_WIDTH + ty; // B的行由block的x坐标和线程的y坐标决定
        int b_load_col = tile_idx * TILE_WIDTH + tx; // B的列由tile编号和线程的x坐标决定
        if (b_load_row < n && b_load_col < d) {
            B_tile[ty][tx] = B[b_load_row * d + b_load_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // --- 阶段B：同步 ---
        __syncthreads();

        // --- 阶段C：从共享内存计算 ---
        for (int i = 0; i < TILE_WIDTH; ++i) {
            // 线程(ty,tx)计算R[row][col], 需要Q[row]和B[col]
            // Q[row]的数据块在 Q_tile[ty] 中
            // B[col]的数据块在 B_tile[tx] 中
            sum += Q_tile[ty][i] * B_tile[tx][i];
        }

        // --- 阶段D：再次同步 ---
        __syncthreads();
    }

    // 5. 将结果写回
    if (row < m && col < n) {
        R[row * n + col] = 1.0f - sum;
    }
}


// 定义一个简单的 pair 结构体，因为 __global__ 函数不能直接用 std::pair
struct FloatUIntPair {
    float value;
    uint32_t key;
};

// 核函数：每个线程负责一个查询，找到Top-K
__global__ void find_topk_kernel(const float* all_distances, // R 矩阵 (m x n)
                                 FloatUIntPair* topk_results, // 最终的小结果数组 (m x k)
                                 int m, int n, int k) {
    // 获取当前线程负责的查询行索引 (row)
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        // 在这个线程的私有寄存器/L1缓存中创建一个小数组来模拟优先队列
        FloatUIntPair local_topk[10]; 

        // 初始化
        for (int i = 0; i < k; ++i) {
            local_topk[i].value = 1e38; // 一个很大的值
            local_topk[i].key = 0;
        }

        // 遍历这个查询对应的所有距离 (n个)
        for (int col = 0; col < n; ++col) {
            float dist = all_distances[row * n + col];
            
            // 找到当前 topk 中最大的值
            float max_val = -1.0f;
            int max_idx = -1;
            for(int i = 0; i < k; ++i) {
                if (local_topk[i].value > max_val) {
                    max_val = local_topk[i].value;
                    max_idx = i;
                }
            }
            // 如果新距离更小，则替换掉最大的那个
            if (dist < max_val) {
                local_topk[max_idx].value = dist;
                local_topk[max_idx].key = col;
            }
        }
        // 把这个线程找到的 topk 结果写回到全局内存
        for (int i = 0; i < k; ++i) {
            topk_results[row * k + i] = local_topk[i];
        }
    }
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_batch_search(float* base, float* all_queries, 
                   size_t base_number, size_t query_number, 
                   size_t vecdim, size_t k) 
{
    
    // --- 0. 创建CUDA事件用于计时 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    printf("\n--- Detailed GPU Performance Breakdown ---\n");

    // --- 1. GPU内存分配 ---
    cudaEventRecord(start);
    float *d_base, *d_queries, *d_results;
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t result_size = query_number * base_number * sizeof(float);
    FloatUIntPair* d_final_topk;
    size_t final_topk_size = query_number * k * sizeof(FloatUIntPair);

    cudaMalloc(&d_base, base_size);
    cudaMalloc(&d_queries, query_size);
    cudaMalloc(&d_results, result_size); // 中间距离矩阵
    cudaMalloc(&d_final_topk, final_topk_size); // 最终的小结果
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for cudaMalloc: \t\t\t%f ms\n", elapsedTime);


    // --- 2. 数据传输 (CPU -> GPU) ---
    cudaEventRecord(start);
    cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, all_queries, query_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for cudaMemcpy (Host -> Device): \t%f ms\n", elapsedTime);


    // --- 3. 调用矩阵乘法核函数 ---
    cudaEventRecord(start);
    dim3 threadsPerBlock_mm(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid_mm(
        (base_number + threadsPerBlock_mm.x - 1) / threadsPerBlock_mm.x,
        (query_number + threadsPerBlock_mm.y - 1) / threadsPerBlock_mm.y
    );
    matrix_mult_tiled_kernel<<<blocksPerGrid_mm, threadsPerBlock_mm>>>(
        d_queries, d_base, d_results, 
        query_number, base_number, vecdim
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for Kernel 1 (Tiled Matrix Multiplication): %f ms\n", elapsedTime); 


    // --- 4. 调用Top-K核函数 ---
    cudaEventRecord(start);
    int threadsPerBlock_topk = 64;
    int blocksPerGrid_topk = (query_number + threadsPerBlock_topk - 1) / threadsPerBlock_topk;
    find_topk_kernel<<<blocksPerGrid_topk, threadsPerBlock_topk>>>(
        d_results, d_final_topk, 
        query_number, base_number, k
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for Kernel 2 (Find Top-K): \t\t%f ms\n", elapsedTime);


    // --- 5. 数据传输 (GPU -> CPU)，只传回小结果！ ---
    cudaEventRecord(start);
    FloatUIntPair* h_topk_results = new FloatUIntPair[query_number * k];
    cudaMemcpy(h_topk_results, d_final_topk, final_topk_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for cudaMemcpy (Device -> Host): \t%f ms (transferred %.2f KB)\n", elapsedTime, final_topk_size / 1024.0f);


    // --- 6. 释放GPU内存 ---
    cudaEventRecord(start);
    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_results); // 释放中间结果
    cudaFree(d_final_topk);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for cudaFree: \t\t\t%f ms\n", elapsedTime);


    // --- 7. CPU后处理 (现在非常快) ---
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> final_results(query_number);
    for(int i = 0; i < query_number; ++i) {
        for(int j = 0; j < k; ++j) {
            final_results[i].push({h_topk_results[i * k + j].value, h_topk_results[i * k + j].key});
        }
    }
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_time - cpu_start_time);
    printf("Time for CPU Post-Processing: \t\t%f ms\n", cpu_duration.count() / 1000.0);
    printf("--- End of Breakdown ---\n\n");

    // --- 销毁事件 ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    delete[] h_topk_results;
    return final_results;
}
