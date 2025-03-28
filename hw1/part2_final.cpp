#include <iostream>
#include <functional>
#include <iomanip>
#include <windows.h>
#include <immintrin.h>  // 用于 SIMD 内部函数

using namespace std;

// 初始化数组
void init(int*& a, int n) {
    a = new int[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}

// 平凡算法：逐一累加
void ordinary(int n) {
    int* a;
    init(a, n);
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    delete[] a;
}

// 多路链式算法：分为两组分别累加
void chain(int n) {
    int* a;
    init(a, n);
    long long sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        if (i + 1 < n)
            sum2 += a[i + 1];
    }
    long long sum = sum1 + sum2;
    delete[] a;
}

// 多路链式算法：采用 4 路展开，充分利用独立累加器（ILP优化）
void chainILP(int n) {
    int* a;
    init(a, n);
    long long sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    int i = 0;
    int limit = n - (n % 4);
    for (; i < limit; i += 4) {
        sum0 += a[i];
        sum1 += a[i + 1];
        sum2 += a[i + 2];
        sum3 += a[i + 3];
    }
    for (; i < n; i++) {
        sum0 += a[i];
    }
    long long sum = sum0 + sum1 + sum2 + sum3;
    delete[] a;
}

// 递归算法：原版递归，不易利用ILP，这里保留
void digui(int n) {
    int* a;
    init(a, n);
    if (n == 1) {
        delete[] a;
        return;
    }
    for (int i = 0; i < n / 2; i++) {
        a[i] += a[n - i - 1];
    }
    // 递归调用时采用规模缩小一半
    digui(n / 2);
    delete[] a;
}

// 二重循环算法：外层控制规模，内层合并相邻元素
void erchong(int n) {
    int* a;
    init(a, n);
    for (int m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
    delete[] a;
}

// 二重循环算法：内层循环展开为 4 路处理
void erchongILP(int n) {
    int* a;
    init(a, n);
    for (int m = n; m > 1; m /= 2) {
        int half = m / 2;
        int limit = half - (half % 4);
        int i = 0;
        for (; i < limit; i += 4) {
            a[i] = a[i * 2] + a[i * 2 + 1];
            a[i + 1] = a[(i + 1) * 2] + a[(i + 1) * 2 + 1];
            a[i + 2] = a[(i + 2) * 2] + a[(i + 2) * 2 + 1];
            a[i + 3] = a[(i + 3) * 2] + a[(i + 3) * 2 + 1];
        }
        for (; i < half; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
    delete[] a;
}

// SIMD 算法：利用 AVX2 指令进行向量化累加，展示指令级并行效果
void simd_sum(int n) {
    int* a;
    init(a, n);
    __m256i vsum = _mm256_setzero_si256();
    int i = 0;
    // 每次处理 8 个 int（256位寄存器）
    for (; i <= n - 8; i += 8) {
        //__m256i 是一个 256 位的整数向量类型，用于 AVX2 指令集。它可以存储 8 个 32 位的整数。
        //_mm256_loadu_si256：这是一个 AVX2 指令，用于从内存中加载 8 个整数到 256 位的向量寄存器中。
        //(__m256i*) & a[i]：将数组 a 中从索引 i 开始的 8 个整数的地址转换为 __m256i 指针类型，以便 _mm256_loadu_si256 可以从中加载数据。
        __m256i v = _mm256_loadu_si256((__m256i*) & a[i]);
        //_mm256_add_epi32：一个 AVX2 指令，用于将两个 256 位的整数向量相加。epi32 表示每个元素是 32 位的整数。
        vsum = _mm256_add_epi32(vsum, v);
    }
    // 将向量寄存器中的 8 个 int 相加
    int sum_arr[8];
    //_mm256_storeu_si256:一个 AVX2 指令，用于将 256 位的整数向量存储到内存中的非对齐位置。
    _mm256_storeu_si256((__m256i*)sum_arr, vsum);
    long long sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
        sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    // 处理剩余部分
    for (; i < n; i++) {
        sum += a[i];
    }
    delete[] a;
}

// 测试函数：执行指定算法，测量平均运行时间（毫秒）
void test(const string& methodName, function<void(int)> func, int n, int loops) {
    LARGE_INTEGER freq, start, finish;
    QueryPerformanceFrequency(&freq);
    long long totalTime = 0;
    for (int i = 0; i < loops; i++) {
        QueryPerformanceCounter(&start);
        func(n);
        QueryPerformanceCounter(&finish);
        totalTime += (finish.QuadPart - start.QuadPart);
    }
    double avgTimeMs = (totalTime * 1000.0 / loops) / freq.QuadPart;
    // 中文对齐打印：规模、方法、耗时
    cout << setw(10) << n << "  "
        << setw(20) << methodName << " "
        << fixed << setprecision(6) << avgTimeMs << " 毫秒" << endl;
}

int main() {
    int loops = 10;
    // 打印表头
    cout << "规模      方法                    耗时" << endl;
    cout << "------------------------------------------------" << endl;
    // n 从 2 开始，每次乘以 2，直到 2^28
    for (int n = 4096; n <=4099; n *= 2) {
        test("平凡算法", ordinary, n, loops);
        //test("多路链式", chain, n, loops);
        test("多路链式（ILP）", chainILP, n, loops);
        test("递归算法", digui, n, loops);
        //test("二重循环", erchong, n, loops);
        test("二重循环（ILP）", erchongILP, n, loops);
        test("SIMD 算法", simd_sum, n, loops);
        cout << "------------------------------------------------" << endl;
    }
    return 0;
}
