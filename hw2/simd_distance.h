#ifndef SIMD_DISTANCE_H
#define SIMD_DISTANCE_H

#include <arm_neon.h>
#include <cstddef>
#include <cassert>
#include <vector>
#include <cmath>   // for round
#include <limits>  // for numeric_limits
#include <algorithm> // for std::min, std::max

// ---------------------------------------- SQ 相关结构和函数 ---
// 用于存储 SQ 量化参数
struct SQParams {
    float min_val = 0.0f;
    float max_val = 1.0f;
    float range = 1.0f; // max_val - min_val
    float scale = 255.0f; // 用于量化的缩放因子
    float inv_scale = 1.0f / 255.0f;
};
// 离线函数：计算全局量化参数并量化 base 数据集
inline uint8_t* quantize_base_sq(const float* base, size_t n, size_t d, SQParams& params) {
    params.min_val = std::numeric_limits<float>::max();
    params.max_val = std::numeric_limits<float>::lowest();

    // 1. 找到全局 min 和 max
    for (size_t i = 0; i < n * d; ++i) {
        params.min_val = std::min(params.min_val, base[i]);
        params.max_val = std::max(params.max_val, base[i]);
    }
    params.range = params.max_val - params.min_val;
    // 避免除零
    if (params.range == 0) params.range = 1.0f;

    // 2. 量化 base 数据
    uint8_t* base_uint8 = new uint8_t[n * d];
    for (size_t i = 0; i < n * d; ++i) {
        float normalized = (base[i] - params.min_val) / params.range;
        // 使用 std::round 并钳位到 [0, 255]
        base_uint8[i] = static_cast<uint8_t>(
            std::max(0.0f, std::min(255.0f, std::round(normalized * params.scale)))
        );
    }

    // 存储用于反量化的参数
    params.inv_scale = 1.0f / params.scale;

    std::cerr << "SQ Quantization: min=" << params.min_val << ", max=" << params.max_val << std::endl;
    return base_uint8;
}
// 在线函数：量化查询向量
inline void quantize_query_sq(const float* query, uint8_t* query_uint8, size_t d, const SQParams& params) {
    for (size_t i = 0; i < d; ++i) {
        float normalized = (query[i] - params.min_val) / params.range;
        query_uint8[i] = static_cast<uint8_t>(
            std::max(0.0f, std::min(255.0f, std::round(normalized * params.scale)))
        );
    }
}

// ---------------------------------------- 底层距离计算函数 ---

// 1.1---计算 uint8 向量间差的平方和 (SSD) - L2 平方的近似
inline uint32_t compute_L2_sq_neon_optimized_uint8( // 函数名稍作修改
    const uint8_t* query_uint8,
    const uint8_t* base_vec_uint8,
    size_t dim) // 传入实际维度
{
    // 现在断言维度是 16 的倍数即可，因为收尾循环按 16 处理
    assert(dim % 16 == 0 && "维度必须是16的倍数");
    assert(dim == 96 && "此特定版本针对维度96优化"); // 可以加一个特定维度断言

    // 1. 初始化累加器 (保持不变)
    int32x4_t sum_sq_diff_acc0 = vmovq_n_s32(0);
    int32x4_t sum_sq_diff_acc1 = vmovq_n_s32(0);
    int32x4_t sum_sq_diff_acc2 = vmovq_n_s32(0);
    int32x4_t sum_sq_diff_acc3 = vmovq_n_s32(0);

    // --- 主循环：处理 64 的倍数部分 ---
    size_t main_loop_limit = (dim / 64) * 64; // 计算能被64整除的最大索引
    const size_t step64 = 64;
    size_t i = 0;
    for (; i < main_loop_limit; i += step64) {
        // (这里的代码和之前处理 64 个 uint8 的循环体完全一样)
        // --- 加载 4 组 16x uint8 数据 ---
        uint8x16_t q0_u8 = vld1q_u8(query_uint8 + i + 0);
        uint8x16_t b0_u8 = vld1q_u8(base_vec_uint8 + i + 0);
        uint8x16_t q1_u8 = vld1q_u8(query_uint8 + i + 16);
        uint8x16_t b1_u8 = vld1q_u8(base_vec_uint8 + i + 16);
        uint8x16_t q2_u8 = vld1q_u8(query_uint8 + i + 32);
        uint8x16_t b2_u8 = vld1q_u8(base_vec_uint8 + i + 32);
        uint8x16_t q3_u8 = vld1q_u8(query_uint8 + i + 48);
        uint8x16_t b3_u8 = vld1q_u8(base_vec_uint8 + i + 48);

        // --- 计算差值 (int16) ---
        // (省略重复的差值计算代码)
        int16x8_t b0_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b0_u8)));
        int16x8_t q0_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q0_u8)));
        int16x8_t diff0_s16_low = vsubq_s16(b0_s16_low, q0_s16_low);
        int16x8_t b0_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b0_u8)));
        int16x8_t q0_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q0_u8)));
        int16x8_t diff0_s16_high = vsubq_s16(b0_s16_high, q0_s16_high);
        // ... diff1, diff2, diff3 ...
        int16x8_t b1_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b1_u8)));
        int16x8_t q1_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q1_u8)));
        int16x8_t diff1_s16_low = vsubq_s16(b1_s16_low, q1_s16_low);
        int16x8_t b1_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b1_u8)));
        int16x8_t q1_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q1_u8)));
        int16x8_t diff1_s16_high = vsubq_s16(b1_s16_high, q1_s16_high);
        int16x8_t b2_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b2_u8)));
        int16x8_t q2_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q2_u8)));
        int16x8_t diff2_s16_low = vsubq_s16(b2_s16_low, q2_s16_low);
        int16x8_t b2_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b2_u8)));
        int16x8_t q2_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q2_u8)));
        int16x8_t diff2_s16_high = vsubq_s16(b2_s16_high, q2_s16_high);
        int16x8_t b3_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b3_u8)));
        int16x8_t q3_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q3_u8)));
        int16x8_t diff3_s16_low = vsubq_s16(b3_s16_low, q3_s16_low);
        int16x8_t b3_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b3_u8)));
        int16x8_t q3_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q3_u8)));
        int16x8_t diff3_s16_high = vsubq_s16(b3_s16_high, q3_s16_high);


        // --- 累加差值平方 ---
        // (省略重复的累加代码)
        sum_sq_diff_acc0 = vmlal_s16(sum_sq_diff_acc0, vget_low_s16(diff0_s16_low), vget_low_s16(diff0_s16_low));
        sum_sq_diff_acc1 = vmlal_s16(sum_sq_diff_acc1, vget_high_s16(diff0_s16_low), vget_high_s16(diff0_s16_low));
        sum_sq_diff_acc2 = vmlal_s16(sum_sq_diff_acc2, vget_low_s16(diff0_s16_high), vget_low_s16(diff0_s16_high));
        sum_sq_diff_acc3 = vmlal_s16(sum_sq_diff_acc3, vget_high_s16(diff0_s16_high), vget_high_s16(diff0_s16_high));
        sum_sq_diff_acc0 = vmlal_s16(sum_sq_diff_acc0, vget_low_s16(diff1_s16_low), vget_low_s16(diff1_s16_low));
        sum_sq_diff_acc1 = vmlal_s16(sum_sq_diff_acc1, vget_high_s16(diff1_s16_low), vget_high_s16(diff1_s16_low));
        sum_sq_diff_acc2 = vmlal_s16(sum_sq_diff_acc2, vget_low_s16(diff1_s16_high), vget_low_s16(diff1_s16_high));
        sum_sq_diff_acc3 = vmlal_s16(sum_sq_diff_acc3, vget_high_s16(diff1_s16_high), vget_high_s16(diff1_s16_high));
        sum_sq_diff_acc0 = vmlal_s16(sum_sq_diff_acc0, vget_low_s16(diff2_s16_low), vget_low_s16(diff2_s16_low));
        sum_sq_diff_acc1 = vmlal_s16(sum_sq_diff_acc1, vget_high_s16(diff2_s16_low), vget_high_s16(diff2_s16_low));
        sum_sq_diff_acc2 = vmlal_s16(sum_sq_diff_acc2, vget_low_s16(diff2_s16_high), vget_low_s16(diff2_s16_high));
        sum_sq_diff_acc3 = vmlal_s16(sum_sq_diff_acc3, vget_high_s16(diff2_s16_high), vget_high_s16(diff2_s16_high));
        sum_sq_diff_acc0 = vmlal_s16(sum_sq_diff_acc0, vget_low_s16(diff3_s16_low), vget_low_s16(diff3_s16_low));
        sum_sq_diff_acc1 = vmlal_s16(sum_sq_diff_acc1, vget_high_s16(diff3_s16_low), vget_high_s16(diff3_s16_low));
        sum_sq_diff_acc2 = vmlal_s16(sum_sq_diff_acc2, vget_low_s16(diff3_s16_high), vget_low_s16(diff3_s16_high));
        sum_sq_diff_acc3 = vmlal_s16(sum_sq_diff_acc3, vget_high_s16(diff3_s16_high), vget_high_s16(diff3_s16_high));
    }

    // --- 收尾循环：处理剩余部分 (步长为 16) ---
    const size_t step16 = 16;
    for (; i < dim; i += step16) { // 从主循环结束的地方继续
        // --- 加载 1 组 16x uint8 数据 ---
        uint8x16_t q_u8 = vld1q_u8(query_uint8 + i);
        uint8x16_t b_u8 = vld1q_u8(base_vec_uint8 + i);

        // --- 计算差值 (int16) ---
        int16x8_t b_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_u8)));
        int16x8_t q_s16_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q_u8)));
        int16x8_t diff_s16_low = vsubq_s16(b_s16_low, q_s16_low);
        int16x8_t b_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_u8)));
        int16x8_t q_s16_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q_u8)));
        int16x8_t diff_s16_high = vsubq_s16(b_s16_high, q_s16_high);

        // --- 累加差值平方 (只需要累加到 4 个累加器中的对应部分) ---
        sum_sq_diff_acc0 = vmlal_s16(sum_sq_diff_acc0, vget_low_s16(diff_s16_low), vget_low_s16(diff_s16_low));   // 0-3
        sum_sq_diff_acc1 = vmlal_s16(sum_sq_diff_acc1, vget_high_s16(diff_s16_low), vget_high_s16(diff_s16_low));  // 4-7
        sum_sq_diff_acc2 = vmlal_s16(sum_sq_diff_acc2, vget_low_s16(diff_s16_high), vget_low_s16(diff_s16_high));  // 8-11
        sum_sq_diff_acc3 = vmlal_s16(sum_sq_diff_acc3, vget_high_s16(diff_s16_high), vget_high_s16(diff_s16_high)); // 12-15
    }

    // 3. 合并 4 个累加器的结果 (代码不变)
    sum_sq_diff_acc0 = vaddq_s32(sum_sq_diff_acc0, sum_sq_diff_acc1);
    sum_sq_diff_acc2 = vaddq_s32(sum_sq_diff_acc2, sum_sq_diff_acc3);
    int32x4_t final_sum_vec = vaddq_s32(sum_sq_diff_acc0, sum_sq_diff_acc2);

    // 4. 高效水平加法 (代码不变)
    int32_t total_ssd_s32 = vaddvq_s32(final_sum_vec);

    // 5. 返回结果 (代码不变)
    assert(total_ssd_s32 >= 0 && "SSD result should be non-negative");
    return static_cast<uint32_t>(total_ssd_s32);
}
// 1.2---计算 uint8  ip 
inline uint32_t compute_ip_sq_neon_optimized_uint8(
    const uint8_t* query_uint8,
    const uint8_t* base_vec_uint8,
    size_t dim)
{
    assert(dim % 16 == 0 && "维度必须是16的倍数");
    assert(dim == 96 && "此特定版本针对维度96优化");

    // 1. 初始化 4 个累加器向量 (uint32x4_t)
    uint32x4_t sum_prod_acc0 = vmovq_n_u32(0);
    uint32x4_t sum_prod_acc1 = vmovq_n_u32(0);
    uint32x4_t sum_prod_acc2 = vmovq_n_u32(0);
    uint32x4_t sum_prod_acc3 = vmovq_n_u32(0);

    // --- 主循环：处理 64 的倍数部分 ---
    size_t main_loop_limit = (dim / 64) * 64;
    const size_t step64 = 64;
    size_t i = 0;
    for (; i < main_loop_limit; i += step64) {
        // --- 加载 ---
        uint8x16_t q0_u8 = vld1q_u8(query_uint8 + i + 0);
        uint8x16_t b0_u8 = vld1q_u8(base_vec_uint8 + i + 0);
        uint8x16_t q1_u8 = vld1q_u8(query_uint8 + i + 16); // 修正：删除重复声明
        uint8x16_t b1_u8 = vld1q_u8(base_vec_uint8 + i + 16); // 修正：删除重复声明
        uint8x16_t q2_u8 = vld1q_u8(query_uint8 + i + 32);  // 修正：修正拼写错误并删除重复声明
        uint8x16_t b2_u8 = vld1q_u8(base_vec_uint8 + i + 32);
        uint8x16_t q3_u8 = vld1q_u8(query_uint8 + i + 48);
        uint8x16_t b3_u8 = vld1q_u8(base_vec_uint8 + i + 48);

        // --- 计算第 1 组 ---
        uint16x8_t prod0_u16_low = vmull_u8(vget_low_u8(q0_u8), vget_low_u8(b0_u8));
        uint16x8_t prod0_u16_high = vmull_u8(vget_high_u8(q0_u8), vget_high_u8(b0_u8));
        // 修正：使用正确的累加器名称
        sum_prod_acc0 = vaddw_u16(sum_prod_acc0, vget_low_u16(prod0_u16_low));
        sum_prod_acc1 = vaddw_u16(sum_prod_acc1, vget_high_u16(prod0_u16_low));
        sum_prod_acc2 = vaddw_u16(sum_prod_acc2, vget_low_u16(prod0_u16_high));
        sum_prod_acc3 = vaddw_u16(sum_prod_acc3, vget_high_u16(prod0_u16_high));

        // --- 计算第 2 组 ---
        uint16x8_t prod1_u16_low = vmull_u8(vget_low_u8(q1_u8), vget_low_u8(b1_u8));
        uint16x8_t prod1_u16_high = vmull_u8(vget_high_u8(q1_u8), vget_high_u8(b1_u8));
        sum_prod_acc0 = vaddw_u16(sum_prod_acc0, vget_low_u16(prod1_u16_low));
        sum_prod_acc1 = vaddw_u16(sum_prod_acc1, vget_high_u16(prod1_u16_low));
        sum_prod_acc2 = vaddw_u16(sum_prod_acc2, vget_low_u16(prod1_u16_high));
        sum_prod_acc3 = vaddw_u16(sum_prod_acc3, vget_high_u16(prod1_u16_high));

        // --- 计算第 3 组 ---
        uint16x8_t prod2_u16_low = vmull_u8(vget_low_u8(q2_u8), vget_low_u8(b2_u8));
        uint16x8_t prod2_u16_high = vmull_u8(vget_high_u8(q2_u8), vget_high_u8(b2_u8));
        sum_prod_acc0 = vaddw_u16(sum_prod_acc0, vget_low_u16(prod2_u16_low));
        sum_prod_acc1 = vaddw_u16(sum_prod_acc1, vget_high_u16(prod2_u16_low));
        sum_prod_acc2 = vaddw_u16(sum_prod_acc2, vget_low_u16(prod2_u16_high));
        sum_prod_acc3 = vaddw_u16(sum_prod_acc3, vget_high_u16(prod2_u16_high));

        // --- 计算第 4 组 ---
        uint16x8_t prod3_u16_low = vmull_u8(vget_low_u8(q3_u8), vget_low_u8(b3_u8));
        uint16x8_t prod3_u16_high = vmull_u8(vget_high_u8(q3_u8), vget_high_u8(b3_u8));
        sum_prod_acc0 = vaddw_u16(sum_prod_acc0, vget_low_u16(prod3_u16_low));
        sum_prod_acc1 = vaddw_u16(sum_prod_acc1, vget_high_u16(prod3_u16_low));
        sum_prod_acc2 = vaddw_u16(sum_prod_acc2, vget_low_u16(prod3_u16_high));
        sum_prod_acc3 = vaddw_u16(sum_prod_acc3, vget_high_u16(prod3_u16_high));
    }

    // --- 收尾循环：处理剩余部分 (步长为 16) ---
    const size_t step16 = 16; // 定义一次即可
    for (; i < dim; i += step16) {
        uint8x16_t q_u8 = vld1q_u8(query_uint8 + i);
        uint8x16_t b_u8 = vld1q_u8(base_vec_uint8 + i);

        // 修正：使用正确的变量 q_u8 和 b_u8
        uint8x8_t q_low = vget_low_u8(q_u8);
        uint8x8_t b_low = vget_low_u8(b_u8);
        uint8x8_t q_high = vget_high_u8(q_u8); // 修正：使用 vget_high_u8
        uint8x8_t b_high = vget_high_u8(b_u8);

        uint16x8_t prod_low = vmull_u8(q_low, b_low);
        uint16x8_t prod_high = vmull_u8(q_high, b_high);

        // 修正：使用正确的累加器名称
        sum_prod_acc0 = vaddw_u16(sum_prod_acc0, vget_low_u16(prod_low));
        sum_prod_acc1 = vaddw_u16(sum_prod_acc1, vget_high_u16(prod_low));
        sum_prod_acc2 = vaddw_u16(sum_prod_acc2, vget_low_u16(prod_high));
        sum_prod_acc3 = vaddw_u16(sum_prod_acc3, vget_high_u16(prod_high));
    }

    // 3. 合并 4 个累加器的结果
    sum_prod_acc0 = vaddq_u32(sum_prod_acc0, sum_prod_acc1);
    sum_prod_acc2 = vaddq_u32(sum_prod_acc2, sum_prod_acc3);
    uint32x4_t final_sum_vec = vaddq_u32(sum_prod_acc0, sum_prod_acc2);

    // 4. 高效水平加法
    uint32_t total_dot_product = vaddvq_u32(final_sum_vec);

    // 修正：删除重复定义的 step16

    return total_dot_product;
}




// 2.---simd朴素版---
//ip
inline float compute_inner_product_distance_neon_naive(const float* query, const float* base_vec, size_t dim) 
{
    //指令解释：
    //v:向量操作
    //q:操作的目标是一个四元素的向量寄存器

    assert(dim % 4 == 0 && "Dimension must be a multiple of 4 for this NEON implementation");

    // 1. 初始化累加器向量 (4 个 float)，所有值为 0.0f
    float32x4_t sum_vec = vmovq_n_f32(0.0f);

    // 2. 循环处理向量，每次处理4个维度
    for (size_t i = 0; i < dim; i += 4) {
        // 加载数据片段
        float32x4_t query_part = vld1q_f32(query + i);
        float32x4_t base_part = vld1q_f32(base_vec + i);

        // 逐元素乘法
        float32x4_t chengji_part = vmulq_f32(query_part, base_part);

        // 将乘积累加到 sum_vec
        sum_vec = vaddq_f32(sum_vec, chengji_part);
    }

    // 3. 水平加法：将 sum_vec 中的 4 个 float 加起来
    //指导书示例中是将结果存回内存再标量求和，我们这里模拟该思路
    float tmp[4];
    vst1q_f32(tmp, sum_vec); // 将 SIMD 寄存器存回内存数组
    float dot_product = tmp[0] + tmp[1] + tmp[2] + tmp[3]; // 标量求和

    // 4. 计算最终距离
    return 1.0f - dot_product;
}
//L2
inline float compute_L2_distance_neon_naive(const float* query, const float* base_vec, size_t dim)
{
    // 指令解释保持不变，但计算逻辑改变

    assert(dim % 4 == 0 && "Dimension must be a multiple of 4 for this NEON implementation");

    // 1. 初始化累加器向量 (4 个 float)，用于累加差值的平方，所有值为 0.0f
    float32x4_t sum_sq_diff_vec = vmovq_n_f32(0.0f);

    // 2. 循环处理向量，每次处理4个维度
    for (size_t i = 0; i < dim; i += 4) {
        // 加载数据片段
        float32x4_t query_part = vld1q_f32(query + i);
        float32x4_t base_part = vld1q_f32(base_vec + i);

        // 计算差值向量 (base - query)
        float32x4_t diff_part = vsubq_f32(base_part, query_part);

        // 计算差值向量的平方 (逐元素 diff * diff)
        // 注意：这里使用 vmulq_f32 来计算平方
        float32x4_t sq_diff_part = vmulq_f32(diff_part, diff_part);

        // 将差值的平方累加到 sum_sq_diff_vec
        sum_sq_diff_vec = vaddq_f32(sum_sq_diff_vec, sq_diff_part);
    }

    // 3. 水平加法：将 sum_sq_diff_vec 中的 4 个 float (平方和) 加起来
    // 保持原始的低效方式
    float tmp[4];
    vst1q_f32(tmp, sum_sq_diff_vec); // 将 SIMD 寄存器存回内存数组
    // 这个变量现在代表平方距离的总和
    float squared_distance_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]; // 标量求和

    // 4. 计算最终的欧几里得距离 (开方)
    return std::sqrt(squared_distance_sum);
}



// 3.--simd优化---优化点：使用 FMA 指令合并乘加；使用更高效的水平加法指令
//ip
inline float compute_inner_product_distance_neon_optimized(const float* query, const float* base_vec, size_t dim)
{
    // 1. 初始化 4 个累加器向量，以增加指令级并行性
    float32x4_t sum_vec0 = vmovq_n_f32(0.0f);
    float32x4_t sum_vec1 = vmovq_n_f32(0.0f);
    float32x4_t sum_vec2 = vmovq_n_f32(0.0f);
    float32x4_t sum_vec3 = vmovq_n_f32(0.0f);

    // 2. 循环展开处理向量，每次处理 16 个维度 (4 * 4)
    // 对于 dim = 96, 循环将执行 96 / 16 = 6 次
    const size_t step = 16; // 每次循环处理的元素数
    for (size_t i = 0; i < dim; i += step) {
        // --- 第 1 组 (i + 0 to i + 3) ---
        float32x4_t q0 = vld1q_f32(query + i + 0);  // 加载 query[i...i+3]
        float32x4_t b0 = vld1q_f32(base_vec + i + 0); // 加载 base[i...i+3]
        sum_vec0 = vfmaq_f32(sum_vec0, q0, b0);      // sum0 += q0 * b0 (FMA)

        // --- 第 2 组 (i + 4 to i + 7) ---
        float32x4_t q1 = vld1q_f32(query + i + 4);  // 加载 query[i+4...i+7]
        float32x4_t b1 = vld1q_f32(base_vec + i + 4); // 加载 base[i+4...i+7]
        sum_vec1 = vfmaq_f32(sum_vec1, q1, b1);      // sum1 += q1 * b1 (FMA)

        // --- 第 3 组 (i + 8 to i + 11) ---
        float32x4_t q2 = vld1q_f32(query + i + 8);  // 加载 query[i+8...i+11]
        float32x4_t b2 = vld1q_f32(base_vec + i + 8); // 加载 base[i+8...i+11]
        sum_vec2 = vfmaq_f32(sum_vec2, q2, b2);      // sum2 += q2 * b2 (FMA)

        // --- 第 4 组 (i + 12 to i + 15) ---
        float32x4_t q3 = vld1q_f32(query + i + 12); // 加载 query[i+12...i+15]
        float32x4_t b3 = vld1q_f32(base_vec + i + 12); // 加载 base[i+12...i+15]
        sum_vec3 = vfmaq_f32(sum_vec3, q3, b3);      // sum3 += q3 * b3 (FMA)

    }

    // 3. 合并 4 个累加器的结果
    // sum_vec0 = {s00, s01, s02, s03} ... sum_vec3 = {s30, s31, s32, s33}
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1); // sum_vec0 += sum_vec1
    sum_vec2 = vaddq_f32(sum_vec2, sum_vec3); // sum_vec2 += sum_vec3
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec2); // sum_vec0 += sum_vec2
    // 现在 sum_vec0 中包含了所有 96 个乘积的和，分布在 4 个 float 中
    // sum_vec0 = { S0 = sum(sX0), S1 = sum(sX1), S2 = sum(sX2), S3 = sum(sX3) }

    // 4. 高效水平加法：将最终累加器向量 sum_vec0 中的 4 个 float 加起来 (AArch64)
    // vaddvq_f32 计算 S0 + S1 + S2 + S3
    float dot_product = vaddvq_f32(sum_vec0);

    // 5. 计算最终距离
    return 1.0f - dot_product;
}
//L2
inline float compute_L2_distance_neon_optimized(const float* query, const float* base_vec, size_t dim)
{
    // 断言，确保维度是16的倍数
    assert(dim % 16 == 0 && "维度必须是16的倍数以适配4倍循环展开");

    // 1. 初始化 4 个累加器向量，用于累加平方差
    float32x4_t sum_sq_diff0 = vmovq_n_f32(0.0f); // 第 0 组累加器
    float32x4_t sum_sq_diff1 = vmovq_n_f32(0.0f); // 第 1 组累加器
    float32x4_t sum_sq_diff2 = vmovq_n_f32(0.0f); // 第 2 组累加器
    float32x4_t sum_sq_diff3 = vmovq_n_f32(0.0f); // 第 3 组累加器

    // 2. 循环展开处理向量，每次处理 16 个维度
    const size_t step = 16; // 每次循环处理的元素数量
    for (size_t i = 0; i < dim; i += step) {
        // --- 第 1 组 (索引 i + 0 到 i + 3) ---
        float32x4_t q0 = vld1q_f32(query + i + 0);      // 加载 query[i...i+3]
        float32x4_t b0 = vld1q_f32(base_vec + i + 0);    // 加载 base[i...i+3]
        float32x4_t diff0 = vsubq_f32(b0, q0);         // 计算差值: diff = base - query
        // 使用 FMA 计算并累加平方差: sum = sum + (diff * diff)
        sum_sq_diff0 = vfmaq_f32(sum_sq_diff0, diff0, diff0);

        // --- 第 2 组 (索引 i + 4 到 i + 7) ---
        float32x4_t q1 = vld1q_f32(query + i + 4);
        float32x4_t b1 = vld1q_f32(base_vec + i + 4);
        float32x4_t diff1 = vsubq_f32(b1, q1);
        sum_sq_diff1 = vfmaq_f32(sum_sq_diff1, diff1, diff1);

        // --- 第 3 组 (索引 i + 8 到 i + 11) ---
        float32x4_t q2 = vld1q_f32(query + i + 8);
        float32x4_t b2 = vld1q_f32(base_vec + i + 8);
        float32x4_t diff2 = vsubq_f32(b2, q2);
        sum_sq_diff2 = vfmaq_f32(sum_sq_diff2, diff2, diff2);

        // --- 第 4 组 (索引 i + 12 到 i + 15) ---
        float32x4_t q3 = vld1q_f32(query + i + 12);
        float32x4_t b3 = vld1q_f32(base_vec + i + 12);
        float32x4_t diff3 = vsubq_f32(b3, q3);
        sum_sq_diff3 = vfmaq_f32(sum_sq_diff3, diff3, diff3);
    }

    // 3. 合并 4 个累加器的结果
    sum_sq_diff0 = vaddq_f32(sum_sq_diff0, sum_sq_diff1); // 合并 sum0 和 sum1
    sum_sq_diff2 = vaddq_f32(sum_sq_diff2, sum_sq_diff3); // 合并 sum2 和 sum3
    sum_sq_diff0 = vaddq_f32(sum_sq_diff0, sum_sq_diff2); // 最终合并到 sum0
    // 现在 sum_sq_diff0 中包含了所有维度平方差的和，分布在该向量的 4 个 float 元素中

    // 4. 高效水平加法：将最终累加器向量 sum_sq_diff0 中的 4 个 float (平方和) 加起来 (AArch64)
    float squared_distance_sum = vaddvq_f32(sum_sq_diff0); // 得到平方距离的总和

    // 5. 计算最终的欧几里得距离 (开平方根)
    if (squared_distance_sum < 0.0f) {
        squared_distance_sum = 0.0f; // 如果是负数，则钳制为 0
    }
    return std::sqrt(squared_distance_sum); // 返回平方和的平方根
}


// -------------------------------------------- 子向量距离计算 (供 PQ 使用) ---
// ---子向量内积 (SIMD)
inline float compute_sub_inner_product_neon(const float* v1, const float* v2, size_t d_sub) {
    // 断言：确保维度是 24 且是 8 的倍数
    assert(d_sub == 24 && "This function is optimized for d_sub = 24");
    assert(d_sub % 8 == 0 && "Sub-vector dimension must be a multiple of 8 for 2x unrolling");

    // 1. 初始化 2 个累加器向量
    float32x4_t sum0 = vmovq_n_f32(0.0f);
    float32x4_t sum1 = vmovq_n_f32(0.0f);

    // 2. 循环展开处理，每次处理 8 个维度 (步长 8)
    //    对于 d_sub = 24, 循环将执行 24 / 8 = 3 次
    const size_t step = 8;
    for (size_t i = 0; i < d_sub; i += step) {
        // 加载 2 组数据
        float32x4_t v1_0 = vld1q_f32(v1 + i + 0);
        float32x4_t v2_0 = vld1q_f32(v2 + i + 0);
        float32x4_t v1_1 = vld1q_f32(v1 + i + 4);
        float32x4_t v2_1 = vld1q_f32(v2 + i + 4);

        // 使用 FMA 进行独立的累加
        sum0 = vfmaq_f32(sum0, v1_0, v2_0); // sum0 += v1_0 * v2_0
        sum1 = vfmaq_f32(sum1, v1_1, v2_1); // sum1 += v1_1 * v2_1
    }

    // 3. 合并 2 个累加器的结果
    sum0 = vaddq_f32(sum0, sum1); // 合并 sum0 和 sum1

    // 4. 高效水平加法
    return vaddvq_f32(sum0);
}
//子向量L2（平方和）
inline float compute_sub_L2_sq_neon(const float* v1, const float* v2, size_t d_sub) {
    // 断言：确保维度是 24 且是 8 的倍数
    assert(d_sub == 24 && "This function is optimized for d_sub = 24");
    assert(d_sub % 8 == 0 && "Sub-vector dimension must be a multiple of 8 for 2x unrolling");

    // 1. 初始化 2 个累加器向量
    float32x4_t sum_sq0 = vmovq_n_f32(0.0f);
    float32x4_t sum_sq1 = vmovq_n_f32(0.0f);

    // 2. 循环展开处理，每次处理 8 个维度 (步长 8)
    //    对于 d_sub = 24, 循环将执行 24 / 8 = 3 次
    const size_t step = 8;
    for (size_t i = 0; i < d_sub; i += step) {
        // 加载 2 组数据
        float32x4_t v1_0 = vld1q_f32(v1 + i + 0);
        float32x4_t v2_0 = vld1q_f32(v2 + i + 0);
        float32x4_t v1_1 = vld1q_f32(v1 + i + 4);
        float32x4_t v2_1 = vld1q_f32(v2 + i + 4);

        // 计算差值
        float32x4_t diff0 = vsubq_f32(v1_0, v2_0); // v1 - v2 (或者 v2 - v1，平方后一样)
        float32x4_t diff1 = vsubq_f32(v1_1, v2_1);

        // 使用 FMA 计算并累加平方差: sum = sum + (diff * diff)
        sum_sq0 = vfmaq_f32(sum_sq0, diff0, diff0);
        sum_sq1 = vfmaq_f32(sum_sq1, diff1, diff1);
    }

    // 3. 合并 2 个累加器的结果
    sum_sq0 = vaddq_f32(sum_sq0, sum_sq1); // 合并 sum0 和 sum1

    // 4. 高效水平加法，得到平方和
    return vaddvq_f32(sum_sq0); // 直接返回平方和
}

#endif // SIMD_DISTANCE_H