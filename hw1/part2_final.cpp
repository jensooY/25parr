#include <iostream>
#include <functional>
#include <iomanip>
#include <windows.h>
#include <immintrin.h>  // SIMD

using namespace std;

// 初始化数组
void init(int*& a, int n) {
    a = new int[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}

// 平凡算法
void ordinary_core(int* a, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
}

// 多路链式算法（2 路）
void chain_core(int* a, int n) {
    long long sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        if (i + 1 < n)
            sum2 += a[i + 1];
    }
    long long sum = sum1 + sum2;
}

// 多路链式算法（4 路 ILP）
void chainILP_core(int* a, int n) {
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
}

// 递归算法
void digui_core(int* a, int n) {
    if (n == 1)
        return;
    for (int i = 0; i < n / 2; i++) {
        a[i] += a[n - i - 1];
    }
    digui_core(a, n / 2);
}

// 二重循环算法
void erchong_core(int* a, int n) {
    for (int m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
}

// 二重循环算法（4 路 ILP）
void erchongILP_core(int* a, int n) {
    for (int m = n; m > 1; m /= 2) {
        int half = m / 2;
        int limit = half - (half % 4);
        int i = 0;
        for (; i < limit; i += 4) {
            a[i]     = a[i * 2]     + a[i * 2 + 1];
            a[i + 1] = a[(i + 1) * 2] + a[(i + 1) * 2 + 1];
            a[i + 2] = a[(i + 2) * 2] + a[(i + 2) * 2 + 1];
            a[i + 3] = a[(i + 3) * 2] + a[(i + 3) * 2 + 1];
        }
        for (; i < half; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
}

// SIMD 算法
void simd_sum_core(int* a, int n) {
    __m256i vsum = _mm256_setzero_si256();
    int i = 0;
    // 每次处理 8 个 int
    for (; i <= n - 8; i += 8) {
        __m256i v = _mm256_loadu_si256((__m256i*) &a[i]);
        vsum = _mm256_add_epi32(vsum, v);
    }
    int sum_arr[8];
    _mm256_storeu_si256((__m256i*) sum_arr, vsum);
    long long sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                    sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    // 处理剩余部分
    for (; i < n; i++) {
        sum += a[i];
    }
}

// 测试函数：只记录核心算法
void test(const string& methodName, void (*func)(int*, int), int n, int loops) {
    LARGE_INTEGER freq, start, finish;
    QueryPerformanceFrequency(&freq);
    long long totalTime = 0;
    for (int i = 0; i < loops; i++) {
        int* a;
        init(a, n);  
        QueryPerformanceCounter(&start);
        func(a, n); 
        QueryPerformanceCounter(&finish);
        totalTime += (finish.QuadPart - start.QuadPart);
        delete[] a;
    }
    double avgTimeMs = (totalTime * 1000.0 / loops) / freq.QuadPart;
    cout << setw(10) << n << "  " << setw(20) << methodName << " "
         << fixed << setprecision(6) << avgTimeMs << " 毫秒" << endl;
}

int main() {
    int loops = 10;
    cout << "规模      方法                    耗时" << endl;
    cout << "------------------------------------------------" << endl;
    for (int n = 2; n <= 67108864*4; n *= 2) {
        test("平凡算法", ordinary_core, n, loops);
        //test("多路链式", chain_core, n, loops);
        test("多路链式（ILP）", chainILP_core, n, loops);
        test("递归算法", digui_core, n, loops);
        //test("二重循环", erchong_core, n, loops);
        test("二重循环（ILP）", erchongILP_core, n, loops);
        test("SIMD 算法", simd_sum_core, n, loops);
        cout << "------------------------------------------------" << endl;
    }
    return 0;
}
