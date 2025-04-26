// 存放 
// PQ 相关的结构体定义
// 离线的 train_pq(训练函数实现 K-Means 来训练每个子空间的码本)
// 离线的 encode_pq (编码函数将 float 向量编码为 PQ 码字索引)
// 在线的 compute_inner_product_lut (包含标量和 SIMD 版本,LUT计算函数计算查询向量与所有码本中心的内积) 
// approximate_inner_product_adc (ADC函数通过查表求和近似计算内积)#ifndef PQ_UTILS_H
#ifndef PQ_UTILS_H
#define PQ_UTILS_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <iostream> // For cerr/cout
#include <arm_neon.h> // For SIMD

// --- PQ 参数和码本结构 ---
struct PQParams {
    size_t M = 4;         // 子空间数量
    size_t Ks = 256;      // 每个子空间的码字数量
    size_t D = 96;        // 原始向量维度
    size_t D_sub = 24;    // 子向量维度 D / M
    size_t n_train = 10000; // 用于训练的向量数量 
    int kmeans_iters = 20; // K-Means 迭代次数
};
struct PQCodebook {
    PQParams params;
    // 码本存储: M 个子空间，每个子空间 Ks 个中心，每个中心 D_sub 维 float
    // 布局：[M][Ks][D_sub] -> 线性存储为 M * Ks * D_sub
    std::vector<float> centroids; // 存储所有码本中心

    // 获取第 m 个子空间，第 k 个中心的指针
    const float* get_centroid(size_t m, size_t k) const {
        assert(m < params.M && k < params.Ks);
        return centroids.data() + (m * params.Ks + k) * params.D_sub;
    }

    float* get_centroid(size_t m, size_t k) {
         assert(m < params.M && k < params.Ks);
        return centroids.data() + (m * params.Ks + k) * params.D_sub;
    }
};



// --- 离线函数：训练 PQ 码本 ---
inline void train_pq(const float* train_data, PQCodebook& codebook) 
{
    //****目的: 训练生成 PQ 算法所需的码本 (Codebook)。码本本质上是每个子空间中最具代表性的 Ks 个中心点。

    const auto& params = codebook.params;
    codebook.centroids.resize(params.M * params.Ks * params.D_sub);//M * Ks * D_sub 个 ！float

    std::vector<float> sub_vectors(params.n_train * params.D_sub);//临时变量：用于存储从 train_data 中提取出的某个特定子空间的所有训练子向量
    //以下两行：初始化随机数生成器，用于 K-Means 的初始中心选择
    std::random_device rd;
    std::mt19937 gen(rd());//MT19937 是一种高质量的伪随机数生成算法

    for (size_t m = 0; m < params.M; ++m) 
    //遍历每个子空间：找到256个中心
    {
        std::cerr << "Training PQ subspace " << m << "/" << params.M << std::endl;
        // 1. 提取当前子空间的训练数据
        for (size_t i = 0; i < params.n_train; ++i) {
            const float* vec_start = train_data + i * params.D + m * params.D_sub;//从每个训练向量中提取出第 m 个子向量（长度为 D_sub）
            std::copy(vec_start, vec_start + params.D_sub, sub_vectors.begin() + i * params.D_sub);//复制到之前声明的临时变量里去
        }

        // 2. 初始化 K-Means 中心 (随机选择训练点)
        std::vector<size_t> initial_indices(params.n_train);// 1. 初始化索引向量
        for(size_t i=0; i<params.n_train; ++i) initial_indices[i] = i;// 2. 填充索引
        std::shuffle(initial_indices.begin(), initial_indices.end(), gen);// 3. 随机打乱索引
        for (size_t k = 0; k < params.Ks; ++k)
        {
            //从 sub_vectors 中随机选择 Ks 个不重复的子向量作为初始的 Ks 个聚类中心，
            // 并将它们复制到 codebook.centroids 中对应子空间 m 的位置。
             size_t initial_idx = initial_indices[k];
             const float* src = sub_vectors.data() + initial_idx * params.D_sub;
             float* dst = codebook.get_centroid(m, k);
             std::copy(src, src + params.D_sub, dst);
        }


        // 3. K-Means 迭代
        std::vector<size_t> assignments(params.n_train);
        std::vector<float> new_centroids(params.Ks * params.D_sub, 0.0f);
        std::vector<size_t> counts(params.Ks, 0);

        for (int iter = 0; iter < params.kmeans_iters; ++iter) {
            // // ***** 添加调试输出：打印当前迭代 *****
            // std::cerr << "  Subspace " << m << ", Iteration " << iter + 1 << "/" << params.kmeans_iters << std::endl;
            // // **************************************
            // a. 分配步骤：将每个点分配给最近的中心
            for (size_t i = 0; i < params.n_train; ++i) {
                const float* point = sub_vectors.data() + i * params.D_sub;
                float max_ip = -std::numeric_limits<float>::max();
                size_t best_k = 0;
                for (size_t k = 0; k < params.Ks; ++k) {
                    // 使用内积作为相似度（等价于最小化 L2 距离的负值）
                    float ip = compute_sub_inner_product_neon(point, codebook.get_centroid(m, k), params.D_sub);
                    if (ip > max_ip) {
                        max_ip = ip;
                        best_k = k;
                    }
                }
                assignments[i] = best_k;
            }

            // b. 更新步骤：重新计算中心
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);
            for (size_t i = 0; i < params.n_train; ++i) {
                size_t k = assignments[i];
                const float* point = sub_vectors.data() + i * params.D_sub;
                float* center_sum = new_centroids.data() + k * params.D_sub;
                for(size_t d=0; d<params.D_sub; ++d) {
                    center_sum[d] += point[d];
                }
                counts[k]++;
            }

            // 计算平均值作为新中心
            for (size_t k = 0; k < params.Ks; ++k) {
                 if (counts[k] > 0) {
                     float* new_center = new_centroids.data() + k * params.D_sub;
                     float* current_center = codebook.get_centroid(m, k);
                     for(size_t d=0; d<params.D_sub; ++d) {
                         current_center[d] = new_center[d] / counts[k];
                     }
                 } else {
                     // 如果一个中心没有分配到点，可以随机重新初始化，或者保持不变
                     // 这里简单保持不变
                 }
            }
             // std::cerr << "Iter " << iter << " finished." << std::endl;
        }
    }
    std::cerr << "PQ training finished." << std::endl;
}

// --- 离线函数：将 float 向量编码为 PQ 码字 ---
inline void encode_pq(const float* vec, const PQCodebook& codebook, uint8_t* pq_code) {
    const auto& params = codebook.params;
    for (size_t m = 0; m < params.M; ++m) 
    {   //对于每个子空间，分别找到该子空间中最接近的码字
        const float* sub_vec = vec + m * params.D_sub;//获取当前子向量当前子空间的起始地址
        float max_ip = -std::numeric_limits<float>::max();//将其设置为 float 类型能表示的最小可能值
        uint8_t best_k = 0;//这里就用uint8_t了
        for (size_t k = 0; k < params.Ks; ++k) 
        {   //遍历这个子空间的所有码字，开始找最接近的，这里用内积！！，内积越大越接近
            float ip = compute_sub_inner_product_neon(sub_vec, codebook.get_centroid(m, k), params.D_sub);
            if (ip > max_ip) 
            {
                max_ip = ip;
                best_k = static_cast<uint8_t>(k);
            }
        }
        pq_code[m] = best_k;
    }
}


//************ip************************************************************** */
// --- 在线函数：计算查询向量与所有码本中心的内积 LUT ---
// SIMD 
inline void compute_inner_product_lut_neon(const float* query, const PQCodebook& codebook, float* lut) 
{
    const auto& params = codebook.params;
    for (size_t m = 0; m < params.M; ++m) 
    {   //遍历每个子空间
        const float* query_sub = query + m * params.D_sub;//获取子向量当前子空间的起始地址
        float* lut_m = lut + m * params.Ks;
        for (size_t k = 0; k < params.Ks; ++k) {
            // 使用 SIMD 计算子向量内积
            lut_m[k] = compute_sub_inner_product_neon(query_sub, codebook.get_centroid(m, k), params.D_sub);
        }
    }
}
// --- 在线函数：通过 ADC 近似计算内积 ---输入：编码后的向量 pq_code (M 个 uint8_t), 预计算的 LUT；输出：近似的内积值
inline float approximate_inner_product_adc(const uint8_t* pq_code, const float* lut, const PQParams& params) {
    float approx_ip = 0.0f;
    for (size_t m = 0; m < params.M; ++m) 
    {   //遍历每个子空间
        uint8_t k = pq_code[m]; // 获取第 m 个子空间的码字索引
        approx_ip += lut[m * params.Ks + k]; // 从 LUT 查表并累加
    }
    // 注意：这里得到的是近似内积，近似距离是 1 - approx_ip
    return approx_ip;
}

//************L2************************************************************** */
inline void compute_L2_lut_neon(const float* query, const PQCodebook& codebook, float* lut)
{
    const auto& params = codebook.params;
    for (size_t m = 0; m < params.M; ++m)
    {   //遍历每个子空间
        const float* query_sub = query + m * params.D_sub;//获取子向量当前子空间的起始地址
        float* lut_m = lut + m * params.Ks;
        for (size_t k = 0; k < params.Ks; ++k) {
            // *** 调用计算子向量平方 L2 距离的函数 ***
            lut_m[k] = compute_sub_L2_sq_neon(query_sub, codebook.get_centroid(m, k), params.D_sub);
        }
    }
}
inline float approximate_L2_distance_adc(const uint8_t* pq_code, const float* lut, const PQParams& params) {
    float approx_l2_sq = 0.0f; // 累加平方距离
    for (size_t m = 0; m < params.M; ++m)
    {   //遍历每个子空间
        uint8_t k = pq_code[m]; // 获取第 m 个子空间的码字索引
        approx_l2_sq += lut[m * params.Ks + k]; // 从 L2 平方 LUT 查表并累加
    }

    // 处理可能的浮点精度问题导致的负值
    if (approx_l2_sq < 0.0f) {
        approx_l2_sq = 0.0f;
    }

    // 返回开方后的近似 L2 距离
    return std::sqrt(approx_l2_sq);
}





#endif // PQ_UTILS_H