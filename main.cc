#include <x86intrin.h>

#include <chrono>
#include <iostream>
#include <random>

#include "gemm.h"

const int depth_padded = 512;

alignas(32) float a[8 * depth_padded];
alignas(32) float b[8 * depth_padded];
alignas(32) float c[64];

int main() {
  std::default_random_engine engine;
  for (int i = 0; i < 8 * depth_padded; i++) {
    a[i] = std::uniform_real_distribution<>(-1, 1)(engine);
    b[i] = std::uniform_real_distribution<>(-1, 1)(engine);
  }

  int t = 1 << 20;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  for (int i = 0; i < t; i++) {
    ComputeBlock(a, b, depth_padded, c);
  }
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();

  int flag = 1;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      float tot = 0;
      for (int k = 0; k < depth_padded; k++) {
        tot += a[k * 8 + i] * b[k * 8 + j];
      }
      if (std::abs(tot - c[i * 8 + j]) > 1e-4) {
        flag = 0;
        break;
      }
    }

  puts(flag ? "PASS" : "FAIL");

  if (flag) {
    int64_t flops_per_iter = 8 * depth_padded * 8 * 2;
    int64_t flops = flops_per_iter * t;
    double flops_per_second = flops / seconds;
    printf("%.4f GFLOPS\n", flops_per_second / (1 << 30));
  }

  return 0;
}