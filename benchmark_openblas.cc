#include <chrono>
#include <cstdio>

#include "cblas.h"

void BenchmarkOpenBLAS(int rows, int depth, int cols) {
  openblas_set_num_threads(1);
  float *a = new float[rows * depth];
  float *b = new float[depth * cols];
  float *c = new float[rows * cols];

  int t = 1 << 10;
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  for (int i = 0; i < t; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, depth, 1,
                a, depth, b, cols, 0, c, cols);
  }
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  int64_t flops_per_iter = rows * depth * cols * 2;
  int64_t flops = flops_per_iter * t;
  double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();
  double flops_per_second = flops / seconds;
  printf("%.4f GFLOPS\n", flops_per_second / (1e9));

  delete[] a;
  delete[] b;
  delete[] c;
}

int main() {
  BenchmarkOpenBLAS(512, 512, 512);
  return 0;
}