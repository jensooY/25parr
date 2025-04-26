#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <queue>     // 确保包含
#include <utility>   // 确保包含
#include <cstdint>   // 确保包含
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include "simd_distance.h" // 包含距离计算
#include "simd_flat_scan.h" 
using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    return data;
}

//---用于存储单次查询的结果：recall (召回率) 和 latency (延迟，单位微秒)。
struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};


void build_index(float* base, size_t base_number, size_t vecdim)
{
    //---使用 HNSW (Hierarchical Navigable Small World) 算法构建一个相似度搜索索引。

    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    //**************************************************用于rerank
    const size_t nk = 1800; // *** 定义 Re-ranking 候选集大小 Nk ***

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    // ------获取数据
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // -----只测试前2000条查询
    test_number = 2000;
    // -----每条查询只要top10
    const size_t k = 10;

    // -----存储查询结果：results大小为2000*2（recall,latency）
    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
    // --- PQ 离线训练和编码 -------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------
    PQCodebook pq_codebook;// PQ 码本
    uint8_t* base_pq_codes = nullptr;
    //对PQ码本的参数进行初始化
    pq_codebook.params.M = 4; // 子空间数量 (示例)
    pq_codebook.params.Ks = 256; // 每个子空间码字数
    pq_codebook.params.D = vecdim;
    pq_codebook.params.D_sub = vecdim / pq_codebook.params.M;
    pq_codebook.params.n_train = base_number; // 训练数据量,直接全部进行训练
    std::cerr << "********************..." << std::endl;
    std::cout<<"**************训练数: "<<pq_codebook.params.n_train<<std::endl;
    std::cout<<"**************nk: "<<nk<<std::endl;
    pq_codebook.params.kmeans_iters = 20; // K-Means 迭代

    std::cerr << "PQ codebook训练开始..." << std::endl;//cerr标准错误流对象，通常用于输出错误信息或调试信息。
    // 使用原始 base 训练
    train_pq(base, pq_codebook); 
    std::cerr << "PQ codebook训练结束." << std::endl;
    // 接下来开始进行编码，也就是float数据转化为uint8_t数据
    std::cerr << "编码base开始..." << std::endl;
    base_pq_codes = new uint8_t[base_number * pq_codebook.params.M];
    for (size_t i = 0; i < base_number; ++i) 
    {   //将数据库中原始每个float向量都转化为4个uint8_t，存在base_pq_codes中去
        encode_pq(base + i * vecdim, pq_codebook, base_pq_codes + i * pq_codebook.params.M);
    }
    std::cerr << "编码base结束." << std::endl;
    // --------------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------
    std::cerr << "开始查询..." << std::endl;



    // // ********** SQ 离线量化  **********
    // std::cerr << "开始SQ 离线量化..." << std::endl;
    // SQParams sq_params;
    // uint8_t* base_uint8 = quantize_base_sq(base, base_number, vecdim, sq_params); // 使用原始 base 指针
    // std::cerr << "结束SQ 离线量化..." << std::endl;
    // // **********************************



    // 查询测试代码
    for(int i = 0; i < test_number; ++i) 
    {
        // 以下三行：精确测量一小段代码执行时间的常用方法
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // 1----测试暴力版本 
        //auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
        //auto res = flat_search_L2(base, test_query + i*vecdim, base_number, vecdim, k);
        // 2----测试朴素 SIMD 版本 
        //auto res = flat_search_simd_naive(base, test_query + i*vecdim, base_number, vecdim, k);
        // 3----测试优化 SIMD 版本
        //auto res = flat_search_simd_optimized(base, test_query + i*vecdim, base_number, vecdim, k);
        // 4 sq
        //auto res = flat_search_sq_ip_neon(base_uint8, test_query + i * vecdim,base_number, vecdim, k, sq_params);
        // 5 pq      传入编码后的 base，原始 float query，和码本
        //auto res = flat_search_pq_adc(base_pq_codes, test_query + i * vecdim, base_number, pq_codebook, k);

      
        // ************************************************** Re-ranking 步骤 *****
        std::priority_queue<std::pair<float, uint32_t>> res_pq_nk; // 存储 Top-Nk 近似结果
        // 6 pq_rerank******ip or L2**************************************************
        res_pq_nk = flat_search_pq_adc(base_pq_codes, test_query + i * vecdim, base_number, pq_codebook, nk);
        //res_pq_nk = flat_search_pq_adc_l2(base_pq_codes, test_query + i * vecdim, base_number, pq_codebook, nk);

        std::vector<std::pair<float, uint32_t>> rerank_candidates; // 存储 <精确距离, 索引>
        rerank_candidates.reserve(nk); // 预分配空间

        while (!res_pq_nk.empty()) {
            uint32_t candidate_idx = res_pq_nk.top().second; // 获取候选者索引
            res_pq_nk.pop();

            if (candidate_idx >= base_number) continue; // 安全检查

            // 获取原始 float 向量指针
            const float* candidate_vec = base + candidate_idx * vecdim;
            const float* current_query = test_query + i * vecdim;

            // *** 计算精确********ip or L2*********************************************************
            float exact_dist = compute_inner_product_distance_neon_optimized(current_query, candidate_vec, vecdim);
            //float exact_dist = compute_L2_distance_neon_optimized(current_query, candidate_vec, vecdim);

            rerank_candidates.push_back({exact_dist, candidate_idx});
        }

        // *** 根据精确距离排序 (升序) ***
        std::sort(rerank_candidates.begin(), rerank_candidates.end());

        // ************************************************** 结束 Re-ranking *****





        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        //“标准答案”集合
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;//用来累计或计算你的搜索算法找到的结果中，有多少个是正确的

        // while (res.size()) {   
        //     int x = res.top().second;
        //     if(gtset.find(x) != gtset.end()){
        //         ++acc;
        //     }
        //     res.pop();
        // }

        
        // ************************************************** Re-ranking *****
        for (size_t rank = 0; rank < std::min((size_t)k, rerank_candidates.size()); ++rank) {
            int x = rerank_candidates[rank].second; // 获取重排后第 rank 个结果的索引
            if(gtset.count(x)) {
                acc++;
            }
        }
        // ************************************************** Re-ranking *****

        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
