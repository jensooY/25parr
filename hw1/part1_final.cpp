#include <iostream>
#include <iomanip>
#include <windows.h>

using namespace std;

// 初始化矩阵和向量
void initializeMatrix(int**& matrix, int*& a, int n) {
    matrix = new int* [n];
    for (int i = 0; i < n; i++)
        matrix[i] = new int[n];
    a = new int[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            matrix[i][j] = i + j;
        a[i] = i;
    }
}

// 释放矩阵和向量的内存
void cleanup(int** matrix, int* a, int* sum, int n) {
    for (int i = 0; i < n; i++)
        delete[] matrix[i];
    delete[] matrix;
    delete[] a;
    delete[] sum;
}

// 平凡算法
void original(int** matrix, int* a, int* sum, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            sum[i] += matrix[j][i] * a[j];
    }
}

// cache 优化算法
void cache(int** matrix, int* a, int* sum, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            sum[i] += matrix[j][i] * a[j];
    }
}

// 循环展开算法——基于cache算法
void unroll(int** matrix, int* a, int* sum, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i += 2) {
            sum[i] += matrix[j][i] * a[j];
            if (i + 1 < n) {
                sum[i + 1] += matrix[j][i + 1] * a[j];
            }
        }
    }
}

// 循环展开算法——基于平凡算法
void unroll_ord(int** matrix, int* a, int* sum, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 2) {
            sum[i] += matrix[j][i] * a[j];
            if (j + 1 < n) {
                sum[i] += matrix[j + 1][i] * a[j];
            }
        }
    }
}

void test(void (*algor)(int**, int*, int*, int), int n, int& counter, double& seconds) {
    int** matrix;
    int* a;
    initializeMatrix(matrix, a, n);
    int* sum = new int[n];
    for (int i = 0; i < n; i++)
        sum[i] = 0;

    LARGE_INTEGER frequency, start, finish;
    QueryPerformanceFrequency(&frequency);

    counter = 0;
    // 开始计时
    QueryPerformanceCounter(&start);
    while (true) {
        counter++;
        algor(matrix, a, sum, n);
        QueryPerformanceCounter(&finish);
        seconds = static_cast<double>(finish.QuadPart - start.QuadPart) / frequency.QuadPart;
        if (seconds >= 3.0)
            break;
        //VT时用
        // if (counter >= 10)
        //     break;
    }
    cleanup(matrix, a, sum, n);
}

int main() {
    int step = 10; // 通过修改 step 及循环条件，测试 cache hit\miss
    for (int n = 0; n <= 10000; n += step) {
        int counter;
        double seconds;

        test(original, n, counter, seconds);
        cout << left
            << setw(10) << "original:"
            << setw(10) << n
            << setw(10) << counter
            << setw(10) << seconds
            << setw(10) << (seconds / counter)
            << endl;

        test(cache, n, counter, seconds);
        cout << left
            << setw(10) << "cache:"
            << setw(10) << n
            << setw(10) << counter
            << setw(10) << seconds
            << setw(10) << (seconds / counter)
            << endl;

         test(unroll, n, counter, seconds);
         cout << left  
              << setw(10) << "unroll:"
              << setw(10) << n
              << setw(10) << counter
              << setw(10) << seconds
              << setw(10) << (seconds / counter)
              << endl;
        
         test(unroll_ord, n, counter, seconds);
         cout << left
              << setw(10) << "unroll_ord:"
              << setw(10) << n
              << setw(10) << counter
              << setw(10) << seconds
              << setw(10) << (seconds / counter)
              << endl;

        if (n == 100)
            step = 100;
        if (n == 1000)
            step = 1000;
    }
    return 0;
}
