//gpu头文件
// in gpu_flat_scan.h
#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include "ivf.h"

// 返回值是一个包含每个查询结果的二维vector
std::vector<std::priority_queue<std::pair<float, uint32_t>>>
gpu_batch_search(float* base, float* all_queries, 
                   size_t base_number, size_t query_number, 
                   size_t vecdim, size_t k) ;

