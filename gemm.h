#ifndef GEMM_H_
#define GEMM_H_

#include <cstdint>

void ComputeBlock(const float *packed_lhs_data, const float *packed_rhs_data,
                  const int64_t depth_padded, float *packed_output_data);

void Pack8x4(int64_t width, int64_t depth, int64_t stride, const float *matrix,
             float *packed_matrix);

void Unpack8x8(const float *packed_output, int64_t rows, int64_t cols,
               int64_t stride, float *output);

#endif