#include <x86intrin.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "gemm.h"
#include "utils.h"

// lhs: ColMajor, rhs: RowMajor, output: RowMajor
void MatMulRef(int64_t rows, int64_t depth, int64_t cols,
               const std::vector<float> &lhs, const std::vector<float> &rhs,
               std::vector<float> *output) {
  output->resize(rows * cols);
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++) {
      int64_t idx = i * cols + j;
      output->at(idx) = 0;
      for (int64_t k = 0; k < depth; k++) {
        // (i, k) * (k, j)
        int64_t lhs_idx = k * rows + i;
        int64_t rhs_idx = k * cols + j;
        output->at(idx) += lhs[lhs_idx] * rhs[rhs_idx];
      }
    }
}

double BenchmarkGEMM(int64_t rows, int64_t depth, int64_t cols,
                     int64_t num_runs) {
  // initialize data
  std::vector<float> lhs(rows * depth), rhs(cols * depth), answer, output;
  std::default_random_engine engine;
  for (int i = 0; i < lhs.size(); i++) {
    lhs[i] = std::uniform_real_distribution<>(-1, 1)(engine);
  }
  for (int i = 0; i < rhs.size(); i++) {
    rhs[i] = std::uniform_real_distribution<>(-1, 1)(engine);
  }

  // perform reference implementation
  MatMulRef(rows, depth, cols, lhs, rhs, &answer);

  // allocate memory for fast GEMM
  int64_t rows_padded = RoundUp(rows, 8);
  int64_t depth_padded = RoundUp(depth, 4);
  int64_t cols_padded = RoundUp(cols, 8);
  float *packed_lhs = (float *)_mm_malloc(
      rows_padded * depth_padded * sizeof(float), 8 * sizeof(float));
  float *packed_rhs = (float *)_mm_malloc(
      cols_padded * depth_padded * sizeof(float), 8 * sizeof(float));
  float *packed_output = (float *)_mm_malloc(
      rows_padded * cols_padded * sizeof(float), 8 * sizeof(float));

  output.resize(rows * cols);

  constexpr int64_t eight = 8;
  // packing
  for (int64_t i = 0; i < rows; i += 8) {
    int64_t width = std::min(eight, rows - i);
    Pack8x4(width, depth, rows, lhs.data() + i, packed_lhs + i * depth_padded);
  }
  for (int64_t i = 0; i < cols; i += 8) {
    int64_t width = std::min(eight, cols - i);
    Pack8x4(width, depth, cols, rhs.data() + i, packed_rhs + i * depth_padded);
  }

  std::chrono::high_resolution_clock::time_point t0 =
      std::chrono::high_resolution_clock::now();
  for (int64_t iter = 0; iter < num_runs; iter++) {
    // perform fast GEMM
    for (int64_t i = 0; i < rows_padded; i += 8)
      for (int64_t j = 0; j < cols_padded; j += 8) {
        ComputeBlock(packed_lhs + i * depth_padded,
                     packed_rhs + j * depth_padded, depth_padded,
                     packed_output + i * cols_padded + j * 8);
      }
  }
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  // unpacking
  for (int64_t i = 0; i < rows_padded; i += 8)
    for (int64_t j = 0; j < cols_padded; j += 8) {
      Unpack8x8(packed_output + i * cols_padded + j * 8,
                std::min(eight, rows - i), std::min(eight, cols - j), cols,
                output.data() + i * cols + j);
    }

  // free memory
  _mm_free(packed_lhs);
  _mm_free(packed_rhs);
  _mm_free(packed_output);

  for (int64_t i = 0; i < rows * cols; i++) {
    if (std::abs(answer[i] - output[i]) > 1e-4) {
      fprintf(stderr,
              "Assertion failed: answer[%ld] (%.4f) != output[%ld] (%.4f)\n", i,
              answer[i], i, output[i]);
    }
  }

  double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
          .count();
  double flops_per_iter = rows * depth * cols * 2;
  double flops = flops_per_iter * num_runs;

  return flops / seconds / (1e9);
}

int main() {
  double gflops = BenchmarkGEMM(512, 768, 768, 1 << 10);
  printf("%.4f GFLOPS\n", gflops);

  return 0;
}