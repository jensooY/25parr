#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <chrono> 
#include <iomanip>
#include "gpu_ivf.h"
#include <cuda_runtime.h> 


#include "ivf.h" 



// 数据加载函数
template<typename T>
T *LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "错误: 无法打开数据文件 " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    size_t sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read((char*)data + i * d * sz, d * sz);
    }
    fin.close();

    std::cout << "已从 " << data_path << " 加载数据\n";
    std::cout << "  - 维度: " << d << "\n";
    std::cout << "  - 向量数量: " << n << "\n\n";

    return data;
}

// 存储每个查询的结果
struct SearchResult {
    float recall;
    int64_t latency;
};


int main(int argc, char *argv[]) {
    // --- 1. 数据加载 ---
    size_t test_number = 0, base_number = 0, vecdim = 0, test_gt_d = 0;
    std::string data_path = "anndata/"; 
    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    int* test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;

    // --- 2. IVF 索引构建或加载 ---
    const size_t nlist = 2048;
    std::string index_filename = "ivf_index_nlist" + std::to_string(nlist) + ".bin";
    
    IVFIndex index;
    if (!index.load(index_filename)) {
        index = build_ivf_index(base, base_number, vecdim, nlist);
        index.save(index_filename);
    }
    const size_t k = 10;
    const size_t nprobe = 20;

    // 【【【 新增部分：规划并预留工作区内存 】】】
    printf("\n--- Pre-allocating GPU Workspace ---\n");
    
    // Step 1: 规划 (Plan)
    size_t workspace_size = 0;
    size_t alignment = 256; // 内存对齐，256字节通常是个安全的选择

    // 辅助函数，用于计算对齐后的尺寸
    auto align_up = [&](size_t size) {
        return (size + alignment - 1) & ~(alignment - 1);
    };

    // 规划 d_queries
    size_t queries_size = align_up(test_number * vecdim * sizeof(float));
    workspace_size += queries_size;
    
    // 规划 d_centroids
    size_t centroids_size = align_up(nlist * vecdim * sizeof(float));
    workspace_size += centroids_size;

    // 规划 d_dist_matrix
    size_t dist_matrix_size = align_up(test_number * nlist * sizeof(float));
    workspace_size += dist_matrix_size;

    // 规划 d_candidate_lists
    size_t candidate_lists_size = align_up(test_number * nprobe * sizeof(int));
    workspace_size += candidate_lists_size;

    // 规划 d_reordered_base
    size_t reordered_base_size = align_up(index.reordered_base_data.size() * sizeof(float));
    workspace_size += reordered_base_size;
    
    // 规划 d_reordered_ids
    size_t reordered_ids_size = align_up(index.reordered_original_ids.size() * sizeof(uint32_t));
    workspace_size += reordered_ids_size;

    // 规划 d_list_offsets
    size_t list_offsets_size = align_up(index.list_offsets.size() * sizeof(size_t));
    workspace_size += list_offsets_size;
    
    // 规划 d_intermediate_results (使用保守估计)
    // 最坏情况是 nprobe 个最大的簇，这里我们用一个更安全的上界：整个数据库的大小
    int max_intermediate_results = base_number;

    // 规划 d_query_result_offsets
    size_t query_offsets_size = align_up((test_number + 1) * sizeof(int));
    workspace_size += query_offsets_size;

    // 规划 d_query_result_counts
    size_t query_counts_size = align_up(test_number * sizeof(int));
    workspace_size += query_counts_size;

    // 规划 d_final_topk_results
    size_t final_results_size = align_up(test_number * k * sizeof(FloatUIntPair));
    workspace_size += final_results_size;

    printf("  - Required workspace size: %.2f MB\n", workspace_size / (1024.0 * 1024.0));

    // Step 2: 预留 (Allocate)
    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_size);

    // --- 3. 【修改部分】调用GPU-IVF进行测试 ---


    std::cout << "\n--- 开始执行搜索测试 (GPU-IVF版本) ---" << std::endl;
    std::cout << "参数: nlist=" << nlist << ", nprobe=" << nprobe << ", k=" << k << std::endl;

    auto search_start = std::chrono::high_resolution_clock::now();
    
    // 调用新的“完全GPU”函数
    // <<<--- 【【【核心修改】】】调用时传入 d_workspace
    auto all_res = gpu_ivf_batch_search_full_gpu(index, test_query, test_number, nprobe, k, d_workspace);

    auto search_end = std::chrono::high_resolution_clock::now();
    long long diff = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();

    
    // --- 4. 验证召回率 ---
    // <<<--- 3. 这部分逻辑需要适配新的返回值类型 (和你的brute-force GPU一样)
    double total_recall = 0.0;
    for (size_t i = 0; i < test_number; ++i) {
        std::set<uint32_t> ground_truth;
        for (size_t j = 0; j < k; ++j) {
            ground_truth.insert(test_gt[j + i * test_gt_d]);
        }
        
        size_t hit_count = 0;
        // 如果 all_res[i] 是空的，这部分不会执行，hit_count就是0
        auto res_queue = all_res[i]; 
        while (!res_queue.empty()) {
            if (ground_truth.count(res_queue.top().second)) {
                hit_count++;
            }
            res_queue.pop();
        }
        total_recall += (float)hit_count / k;
    }

    // --- 5. 统计并输出最终结果 ---
    std::cout << "\n--- 最终测试结果 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "平均召回率@"<< k <<": \t" << total_recall / test_number << std::endl;
    std::cout << "总耗时 (us): \t\t" << diff << std::endl;
    std::cout << "平均延迟 (us): \t" << (double)diff / test_number << std::endl;

    // --- 6. 释放内存 ---
    cudaFree(d_workspace); // 释放工作区
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
