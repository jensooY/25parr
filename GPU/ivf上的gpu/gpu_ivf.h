// in gpu_ivf.h
#pragma once

#include "ivf.h"
#include <vector>
#include <queue>

// 定义这个结构体，以便在 C++ 和 CUDA 代码中共享
// 使用 #pragma pack(4) 来强制4字节对齐，避免编译器自动填充
#pragma pack(push, 4)
struct FloatUIntPair {
    float value;
    uint32_t key;
};
#pragma pack(pop)


// 保留我们之前的混合模式函数，用于对比或调试
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_ivf_batch_search(const IVFIndex& index, 
                     const float* all_queries,
                     size_t query_number, 
                     size_t nprobe, 
                     size_t k);


std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_ivf_batch_search_full_gpu(const IVFIndex& index,
                              const float* all_queries,
                              size_t query_number,
                              size_t nprobe,
                              size_t k,
                              void* d_workspace); // <-- 新增参数
