#ifndef KMEANS_H // 包含守卫，防止重复包含
#define KMEANS_H

#include <vector>    
#include <cstdint>   
#include <limits>    /
#include "simd_distance.h" /
#include <algorithm>
#include <cmath>     // 标准库：数学函数
#include <random>    // 标准库：随机数
#include <iostream>  // 标准库：输入输出
#include <fstream>   // 【新增】标准库：文件流，用于写入文件
#include <string>    // 【新增】标准库：字符串，用于文件名

// 使用std命名空间
using namespace std;

// KMeans聚类结果的结构体 (保持不变)
struct KMeansResult {
    vector<float> centroids;
    vector<vector<uint32_t>> inverted_lists;
    size_t num_clusters;
    size_t dim;
};


// 辅助函数：随机初始化质心 (保持不变)
inline void initialize_centroids_randomly // 标记为inline，因为在头文件中定义
    (const float* base_data, size_t num_points, size_t dim,
        size_t num_clusters, vector<float>& centroids)
{
    centroids.resize(num_clusters * dim);
    vector<uint32_t> chosen_indices;
    chosen_indices.reserve(num_clusters);

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_points - 1);

    for (size_t i = 0; i < num_clusters; ++i)
    {
        size_t rand_idx;
        bool unique = false;
        int attempts = 0; // 防止在num_clusters接近num_points时死循环
        while(!unique && attempts < num_points * 2) // 增加尝试次数上限
        {
            attempts++;
            rand_idx = dist(rng);
            unique = true;
            for(uint32_t chosen_idx : chosen_indices)
            {
                if(chosen_idx == rand_idx)
                {
                    unique = false;
                    break;
                }
            }
        }
        // 如果在多次尝试后仍未找到唯一的，并且还需要更多质心，
        // 这可能在 num_clusters 非常接近 num_points 的情况下发生。
        // 这里简单地允许非唯一（如果尝试次数用尽），或者可以抛出错误/采取其他策略。
        // 对于大多数情况，上述循环应该能找到足够的唯一索引。

        chosen_indices.push_back(rand_idx);
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[i * dim + d] = base_data[rand_idx * dim + d];
        }
    }
}


// KMeans聚类函数的具体实现
// 【修改函数签名，增加一个用于输出统计文件路径的参数】
inline bool run_kmeans // 标记为inline
    (const float* base_data,
        size_t num_points,
        size_t dim,
        size_t num_clusters_to_find,
        int max_iterations,
        vector<float>& centroids_output,
        vector<vector<uint32_t>>& inverted_lists_output,
        const std::string& cluster_stats_filepath = "cluster_point_counts.txt") // <<< 新增参数，并提供默认文件名
{
    // 参数校验 (保持不变)
    if (!base_data || num_points == 0 || dim == 0 || num_clusters_to_find == 0 || num_clusters_to_find > num_points) {
        cerr << "KMeans: 无效的输入参数。" << endl;
        return false;
    }

    centroids_output.resize(num_clusters_to_find * dim);
    vector<float> previous_centroids(num_clusters_to_find * dim);

    initialize_centroids_randomly(base_data, num_points, dim, num_clusters_to_find, centroids_output);
    vector<uint32_t> assignments(num_points);
    inverted_lists_output.assign(num_clusters_to_find, vector<uint32_t>());

    bool converged = false; // 用于标记是否已收敛

    // KMeans迭代过程 (保持不变)
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        previous_centroids = centroids_output;
        for (auto& list : inverted_lists_output)
            list.clear();

        for (size_t i = 0; i < num_points; ++i)
        {
            float min_dist = std::numeric_limits<float>::max();
            uint32_t best_cluster = 0;
            const float* current_point = base_data + i * dim;
            for (size_t j = 0; j < num_clusters_to_find; ++j)
            {
                const float* centroid_vec = centroids_output.data() + j * dim;
                float dist = compute_inner_product_distance_neon_optimized(current_point, centroid_vec, dim);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
            inverted_lists_output[best_cluster].push_back(i);
        }

        fill(centroids_output.begin(), centroids_output.end(), 0.0f);
        vector<uint32_t> cluster_counts(num_clusters_to_find, 0);
        for (size_t i = 0; i < num_points; ++i)
        {
            uint32_t cluster_id = assignments[i];
            const float* current_point = base_data + i * dim;
            for (size_t d = 0; d < dim; ++d)
                centroids_output[cluster_id * dim + d] += current_point[d];
            cluster_counts[cluster_id]++;
        }

        for (size_t c = 0; c < num_clusters_to_find; ++c)
        {
            if (cluster_counts[c] > 0) {
                for (size_t d = 0; d < dim; ++d)
                    centroids_output[c * dim + d] /= cluster_counts[c];
            }
            else
            {
                // cerr << "警告: KMeans的簇 " << c << " 变为空。正在重新初始化。" << endl; // 调试信息
                if (num_points > 0) { // 确保 num_points 不是0
                    size_t rand_idx = std::rand() % num_points;
                    for (size_t d_val = 0; d_val < dim; ++d_val)
                        centroids_output[c * dim + d_val] = base_data[rand_idx * dim + d_val];
                }
            }
        }

        float centroid_shift = 0.0f;
        for(size_t i=0; i < centroids_output.size(); ++i)
            centroid_shift += std::fabs(centroids_output[i] - previous_centroids[i]);
        if (centroid_shift < 1e-4)
        {
            converged = true; // 标记已收敛
            // cout << "KMeans在 " << iter + 1 << " 次迭代后收敛。" << endl; // 调试信息
            break;
        }
        if (iter == max_iterations -1)
        {
             // cout << "KMeans达到最大迭代次数。" << endl; // 调试信息
        }
    }

    // --- 【新增代码块：统计每个簇的数据点数量并写入文件】 ---
    std::ofstream stats_file(cluster_stats_filepath); // 打开（或创建）用于输出统计数据的文件
    if (!stats_file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << cluster_stats_filepath << " 来写入簇统计数据。" << std::endl;
        // 即使文件打不开，KMeans本身可能还是成功的，所以这里不直接返回false，但打印错误
    } else {
        stats_file << "ClusterID\tPointCount" << std::endl; // 写入表头
        size_t total_assigned_points = 0;
        // 确保 inverted_lists_output 的大小与 num_clusters_to_find 一致
        if (inverted_lists_output.size() == num_clusters_to_find) {
            for (size_t c = 0; c < num_clusters_to_find; ++c) {
                size_t point_count_in_cluster = inverted_lists_output[c].size();
                stats_file << c << "\t" << point_count_in_cluster << std::endl;
                total_assigned_points += point_count_in_cluster;
            }
        } else {
            std::cerr << "错误: KMeans内部 inverted_lists_output 大小与簇数量不符，无法统计。" << std::endl;
        }
        stats_file.close(); // 关闭文件
        std::cout << "  KMeans: 簇内数据点数量统计已保存到 " << cluster_stats_filepath << std::endl; // 提示信息

        // 校验总点数
        if (inverted_lists_output.size() == num_clusters_to_find && total_assigned_points != num_points) {
            std::cerr << "  KMeans警告: 统计的总分配点数 (" << total_assigned_points
                      << ") 与原始点数 (" << num_points << ") 不符!" << std::endl;
        }
    }
    // ----------------------------------------------------------

    return true; 
}

#endif // KMEANS_H
