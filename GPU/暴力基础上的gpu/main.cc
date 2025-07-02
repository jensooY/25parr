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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 把 include 路径改得更简洁
#include "hnswlib.h" 
// 可以自行添加需要的头文件
#include "gpu_flat_scan.h"

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

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
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
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 计时开始
    const unsigned long Converter = 1000 * 1000;
    struct timeval val;
    int ret = gettimeofday(&val, NULL);

    // 【新的调用方式】一次性处理所有查询
    auto all_res = gpu_batch_search(base, test_query, base_number, test_number, vecdim, k);

    // 计时结束
    struct timeval newVal;
    ret = gettimeofday(&newVal, NULL);
    int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);


    // 【新的结果验证逻辑】
    float total_recall = 0;
    for (int i = 0; i < test_number; ++i) {
        std::set<uint32_t> gtset;
        for (int j = 0; j < k; ++j) {
            gtset.insert(test_gt[j + i * test_gt_d]);
        }

        size_t acc = 0;
        auto res_queue = all_res[i]; // 获取第i个查询的结果
        while (!res_queue.empty()) {
            if (gtset.count(res_queue.top().second)) {
                ++acc;
            }
            res_queue.pop();
        }
        total_recall += (float)acc / k;
    }

    std::cout << "average recall: " << total_recall / test_number << "\n";
    std::cout << "total latency for " << test_number << " queries (us): " << diff << "\n";
    std::cout << "average latency (us): " << (float)diff / test_number << "\n";

    return 0;
}

