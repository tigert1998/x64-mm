#ifndef GEMM_H_
#define GEMM_H_

#include <cstdint>

void ComputeBlock(const float *packed_lhs_data, const float *packed_rhs_data,
                  const int64_t depth_padded, float *packed_output_data);

#endif