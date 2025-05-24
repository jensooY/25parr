#ifndef KMEANS_H // 包含守卫，防止重复包含
#define KMEANS_H

#include <vector>    // 标准库：动态数组
#include <cstdint>   // 标准库：固定宽度整数类型，如 uint32_t
#include <limits>    // 标准库：数值极限，如 std::numeric_limits
#include "simd_distance.h" // 你的SIMD内积距离计算头文件
#include <algorithm>
#include <cmath> 
#include <random> 
using namespace std;

// KMeans聚类结果的结构体
struct KMeansResult {
    vector<float> centroids;            // 质心数据，扁平化存储：簇数量 * 维度
    vector<vector<uint32_t>> inverted_lists; // 倒排列表，inverted_lists[i] 存储属于簇i的数据点ID
    size_t num_clusters;                // 簇的数量
    size_t dim;                         // 数据维度
};


// 辅助函数：随机初始化质心（例如使用Forgy方法）
// const float* base_data: 原始数据
// size_t num_points: 数据点数量
// size_t dim: 数据维度
// size_t num_clusters: 簇数量
// vector<float>& centroids: 用于存储初始化后的质心
void initialize_centroids_randomly
    (const float* base_data, size_t num_points, size_t dim,
        size_t num_clusters, vector<float>& centroids) 
{
    centroids.resize(num_clusters * dim); //调整质心容器大小
    vector<uint32_t> chosen_indices;      // 已选择的初始质心在原始数据中的索引
    chosen_indices.reserve(num_clusters); // 预分配空间

    // 使用 Mersenne Twister 伪随机数生成器
    std::mt19937 rng(std::random_device{}());
    // 生成 [0, num_points - 1] 范围内的均匀分布整数
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);

    for (size_t i = 0; i < num_clusters; ++i) 
    {
        size_t rand_idx;
        bool unique = false;
        // 确保选取的初始质心是唯一的,因为我们不希望同一个数据点被多次选为不同的初始质心
        while(!unique)
        {
            rand_idx = dist(rng); // 随机选取一个索引
            unique = true;
            for(uint32_t chosen_idx : chosen_indices)
            {
                if(chosen_idx == rand_idx)
                { // 如果已选过
                    unique = false;
                    break;
                }
            }
        }
        chosen_indices.push_back(rand_idx); // 记录已选索引
        // 将选中的数据点复制为初始质心
        for (size_t d = 0; d < dim; ++d) 
        {
            centroids[i * dim + d] = base_data[rand_idx * dim + d];
        }
    }
}


// KMeans聚类函数的具体实现
// const float* base_data: 指向原始 N*D float类型数据的指针
// size_t num_points: N, 数据点的数量
// size_t dim: D, 数据点的维度
// size_t num_clusters_to_find: K, 期望找到的簇的数量
// int max_iterations: KMeans算法的最大迭代次数
// vector<float>& centroids_output: 输出参数，用于存储计算得到的扁平化质心数据
// vector<vector<uint32_t>>& inverted_lists_output: 输出参数，用于存储每个簇包含的数据点ID
// 返回值: 如果成功执行则为true，否则为false
bool run_kmeans
    (const float* base_data,
        size_t num_points,
        size_t dim,
        size_t num_clusters_to_find,
        int max_iterations,
        vector<float>& centroids_output,
        vector<vector<uint32_t>>& inverted_lists_output)
{   // 输出：倒排列表

    centroids_output.resize(num_clusters_to_find * dim); // 调整输出质心容器大小
    vector<float> previous_centroids(num_clusters_to_find * dim); // 用于检查收敛性的前一轮质心

    // 1. 初始化质心 (随机从数据点中选取K个)
    initialize_centroids_randomly(base_data, num_points, dim, num_clusters_to_find, centroids_output);
    vector<uint32_t> assignments(num_points); // assignments[i] 存储数据点i所属的簇ID
    // 初始化倒排列表，为每个簇创建一个空的vector用于存储点ID
    inverted_lists_output.assign(num_clusters_to_find, vector<uint32_t>());

    // KMeans迭代过程
    for (int iter = 0; iter < max_iterations; ++iter) 
    {
        // 存储当前质心，用于后续比较以判断是否收敛
        previous_centroids = centroids_output;

        // 清空上一轮的倒排列表，准备重新分配
        for (auto& list : inverted_lists_output)
            list.clear();

        // 2. 分配步骤：将每个数据点分配给最近的质心
        for (size_t i = 0; i < num_points; ++i) 
        {
            float min_dist = std::numeric_limits<float>::max(); // 初始化最小距离为最大浮点数
            uint32_t best_cluster = 0;                          // 记录最近的簇ID
            const float* current_point = base_data + i * dim;   // 当前数据点的指针

            // 计算当前点与所有质心的距离
            for (size_t j = 0; j < num_clusters_to_find; ++j) 
            {
                const float* centroid_vec = centroids_output.data() + j * dim; // 当前质心的指针
                float dist = compute_inner_product_distance_neon_optimized(current_point, centroid_vec, dim);
                if (dist < min_dist) 
                { // 如果找到更近的质心
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster; // 记录点i所属的簇
            inverted_lists_output[best_cluster].push_back(i); // 将点i的ID加入对应簇的倒排列表
        }

        // 3. 更新步骤：重新计算每个簇的质心（簇内所有点的均值）
        fill(centroids_output.begin(), centroids_output.end(), 0.0f); // 将质心数据清零
        vector<uint32_t> cluster_counts(num_clusters_to_find, 0);    // 记录每个簇中点的数量

        // 累加每个簇内所有点的向量值
        for (size_t i = 0; i < num_points; ++i) 
        {
            uint32_t cluster_id = assignments[i];
            const float* current_point = base_data + i * dim;
            for (size_t d = 0; d < dim; ++d) 
                centroids_output[cluster_id * dim + d] += current_point[d];
            cluster_counts[cluster_id]++; // 对应簇的点数量加1
        }

        // 计算每个簇的新质心（均值）
        for (size_t c = 0; c < num_clusters_to_find; ++c) 
        {
            if (cluster_counts[c] > 0) { // 如果簇不为空
                for (size_t d = 0; d < dim; ++d) 
                    centroids_output[c * dim + d] /= cluster_counts[c];
            } 
            else 
            {
                // 处理空簇的情况：可以重新随机初始化，或从数据集中选取一个点
                // 简单处理：如果一个簇变空，用数据集中的一个随机点重新初始化它
                cerr << "警告: KMeans的簇 " << c << " 变为空。正在重新初始化。" << endl;
                size_t rand_idx = std::rand() % num_points; // 基本的随机选择
                for (size_t d_val = 0; d_val < dim; ++d_val) 
                    centroids_output[c * dim + d_val] = base_data[rand_idx * dim + d_val];
            }
        }

        // 4. 检查是否收敛（例如，质心变化很小）
        float centroid_shift = 0.0f; // 记录质心总体的移动量
        for(size_t i=0; i < centroids_output.size(); ++i)
            centroid_shift += std::fabs(centroids_output[i] - previous_centroids[i]);
        // 用于调试：cout << "KMeans迭代 " << iter << ", 质心移动量: " << centroid_shift << endl;
        if (centroid_shift < 1e-4) 
        {   // 设置一个收敛阈值
            // 用于调试：cout << "KMeans在 " << iter + 1 << " 次迭代后收敛。" << endl;
            break; // 如果变化很小，则认为已收敛，跳出迭代
        }
        if (iter == max_iterations -1)
        { // 如果达到最大迭代次数
             // 用于调试：cout << "KMeans达到最大迭代次数。" << endl;
        }
    }
    return true; // KMeans执行完毕
}

#endif // KMEANS_H