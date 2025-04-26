#ifndef SIMD_FLAT_SCAN_H
#define SIMD_FLAT_SCAN_H

#include <queue>      // for std::priority_queue
#include <utility>    // for std::pair
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t


#include "simd_distance.h"
#include "pq_utils.h" 

// --- 朴素 SIMD 版本的 flat_search ---****************可替换ip，L2!!!!!!!!!*************************
// 结构和逻辑与原始 flat_search 相同，仅替换距离计算部分
inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_simd_naive(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    // 使用优先队列存储结果 (最大堆，按距离 first 排序，值小的优先)
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 遍历基础数据集中的每一个向量
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // *** 核心修改：调用朴素 SIMD 距离计算函数 ***
        //float dis = compute_inner_product_distance_neon_naive(query, base + i * vecdim, vecdim);
        float dis = compute_L2_distance_neon_naive(query, base + i * vecdim, vecdim);
        std::cerr<<"***************"<<std::endl;


        // 维护 Top-k 结果
        if(q.size() < k) {
            q.push({dis, i}); // 如果队列没满，直接加入
        }
        else
        {
            // 如果队列满了，并且当前距离小于队列中最大距离（堆顶）
            if(dis < q.top().first)
            {
                q.push({dis, i}); // 加入当前更好的结果
                q.pop();          // 移除原来最差的结果
            }
        }
    }
    return q; // 返回包含 Top-k 结果的优先队列
}


// --- 优化 SIMD 版本的 flat_search ---****************可替换ip，L2!!!!!!!!!*************************
// 结构和逻辑与原始 flat_search 相同，仅替换距离计算部分
inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_simd_optimized(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 遍历基础数据集中的每一个向量
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // *** 核心修改：调用优化 SIMD 距离计算函数 ***
        //float dis = compute_inner_product_distance_neon_optimized(query, base + i * vecdim, vecdim);
        float dis = compute_L2_distance_neon_optimized(query, base + i * vecdim, vecdim);
        

        // 维护 Top-k 结果 (逻辑同上)
        if(q.size() < k) {
            q.push({dis, i});
        }
        else
        {
            if(dis < q.top().first)
            {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}



// ***** 新增：使用 SQ 量化 + SIMD 计算 L2 的搜索函数 *****
// 注意：这个函数需要接收量化后的数据和量化参数
inline std::priority_queue<std::pair<uint32_t, uint32_t>> // 返回值距离类型改为 uint32_t
flat_search_sq_l2_neon(
    const uint8_t* base_uint8,      // 量化后的基础数据集
    const float* query_float,       // 原始 float 查询向量
    size_t base_number,
    size_t vecdim,
    size_t k,
    const SQParams& params)         // 量化参数
{
    // 使用优先队列存储结果 <距离(SSD), 索引>
    // 因为 SSD 越小越好，所以优先队列应该是最大堆
    std::priority_queue<std::pair<uint32_t, uint32_t>> q;

    // 为量化查询向量分配临时空间
    // C++ 中不推荐 VLA，使用 vector
    std::vector<uint8_t> query_uint8_vec(vecdim);
    uint8_t* query_uint8 = query_uint8_vec.data();

    // 在线量化查询向量
    quantize_query_sq(query_float, query_uint8, vecdim, params);

    // 遍历量化后的基础数据集
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // *** 核心修改：调用 uint8 SSD SIMD 计算函数 ***
        uint32_t dist_ssd = compute_L2_sq_neon_optimized_uint8(query_uint8, base_uint8 + i * vecdim, vecdim);

        // 维护 Top-k 结果 (SSD 越小越好)
        if(q.size() < k) {
            q.push({dist_ssd, i});
        }
        else
        {
            // 注意：标准库优先队列是最大堆，top() 是最大元素
            if(dist_ssd < q.top().first) // 如果当前 SSD < 堆顶的最大 SSD
            {
                q.push({dist_ssd, i});
                q.pop();
            }
        }
    }
    return q;
}
// ***** 新增：使用 SQ 量化 + SIMD 计算 ip 的搜索函数 *****
inline std::priority_queue<std::pair<int64_t, uint32_t>>
flat_search_sq_ip_neon(
    const uint8_t* base_uint8,      // 量化后的基础数据集
    const float* query_float,       // 原始 float 查询向量
    size_t base_number,
    size_t vecdim,
    size_t k,
    const SQParams& params)         // 量化参数
{
    //std::cerr<<"ip****************"<<std::endl;
    // 使用优先队列存储结果 <-点积, 索引>
    // 使用 int64_t 存储负点积，确保不会溢出
    std::priority_queue<std::pair<int64_t, uint32_t>> q;

    // 为量化查询向量分配临时空间
    std::vector<uint8_t> query_uint8_vec(vecdim);
    uint8_t* query_uint8 = query_uint8_vec.data();

    // 在线量化查询向量
    quantize_query_sq(query_float, query_uint8, vecdim, params);

    // 遍历量化后的基础数据集
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // *** 调用 uint8 点积 SIMD 计算函数 ***
        uint32_t dot_product_u = compute_ip_sq_neon_optimized_uint8(
            query_uint8, base_uint8 + i * vecdim, vecdim);

        // *** 使用点积的负数作为排序键 (距离) ***
        int64_t neg_dot_product = -static_cast<int64_t>(dot_product_u);

        // 维护 Top-k 结果 (负点积越小 = 原始点积越大 = 距离越近)
        if(q.size() < k) {
            q.push({neg_dot_product, i});
        }
        else
        {
            // 标准库优先队列是最大堆，top() 是最大元素 (即负点积最大 = 原始点积最小)
            if(neg_dot_product < q.top().first) // 如果当前负点积 < 堆顶的最大负点积
            {
                q.push({neg_dot_product, i}); // 加入点积更大的结果
                q.pop();                     // 移除点积最小的结果
            }
        }
    }
    return q;
}


// ***** 新增：使用 PQ ADC 近似内积距离的搜索函数 *****
// 注意：返回的距离是近似内积距离 (1 - approx_ip)
inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_adc(
    const uint8_t* base_pq_codes,   // 编码后的基础数据集 (N * M 的 uint8)
    const float* query_float,       // 原始 ！float 查询向量
    size_t base_number,
    const PQCodebook& codebook,     // PQ 码本和参数
    size_t nk)//***************************************************修改为nk，用于rerank
{
    const auto& params = codebook.params;
    // 使用优先队列存储结果 <距离, 索引>
    // 内积距离！！！越小越好，所以用默认最大堆，注意：内积距离≠内积。内积距离=1-内积。
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 1. 为当前查询预计算 LUT
    std::vector<float> lut(params.M * params.Ks); //4*256=1024
    // *** 在这里使用SIMD 计算 LUT ***
    compute_inner_product_lut_neon(query_float, codebook, lut.data());

    // 2. 遍历编码后的基础数据集
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // 获取第 i 个向量的 PQ 编码 (M 个 uint8_t)
        const uint8_t* pq_code = base_pq_codes + i * params.M;

        // 3. 使用 ADC 计算近似内积
        float approx_ip = approximate_inner_product_adc(pq_code, lut.data(), params);

        // 4. 计算近似内积距离
        float approx_dist = 1.0f - approx_ip;

        // 5. 维护 Top-k 结果 (距离越小越好)
        if(q.size() < nk) {//**************************************************修改为nk，用于rerank
            q.push({approx_dist, i});
        }
        else
        {
            if(approx_dist < q.top().first)
            {
                q.push({approx_dist, i});
                q.pop();
            }
        }
    }
    return q;
}
//***** 新增：使用 PQ ADC 近似！L2！距离的搜索函数 *****
inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_adc_l2(
    const uint8_t* base_pq_codes,   // 编码后的基础数据集 (N * M 的 uint8)
    const float* query_float,       // 原始 float 查询向量
    size_t base_number,
    const PQCodebook& codebook,     // PQ 码本和参数
    size_t nk) // 使用 nk 支持 rerank
{
    const auto& params = codebook.params;
    // 使用优先队列存储结果 <距离, 索引>
    // L2 距离越小越好，所以用默认最大堆
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 1. 为当前查询预计算 L2 平方距离 LUT
    std::vector<float> lut(params.M * params.Ks); // 存储平方 L2 距离
    // *** 调用计算 L2 LUT 的函数 ***
    compute_L2_lut_neon(query_float, codebook, lut.data());

    // 2. 遍历编码后的基础数据集
    for(uint32_t i = 0; i < base_number; ++i)
    {
        // 获取第 i 个向量的 PQ 编码 (M 个 uint8_t)
        const uint8_t* pq_code = base_pq_codes + i * params.M;

        // 3. 使用 ADC 计算近似 L2 距离
        // *** 调用近似 L2 距离的函数 ***
        float approx_dist = approximate_L2_distance_adc(pq_code, lut.data(), params);

        // 4. 维护 Top-k 结果 (L2 距离越小越好)
        if(q.size() < nk) { // 使用 nk
            q.push({approx_dist, i});
        }
        else
        {
            if(approx_dist < q.top().first) // L2 距离直接比较
            {
                q.push({approx_dist, i});
                q.pop();
            }
        }
    }
    return q;
}


#endif // SIMD_FLAT_SCAN_H