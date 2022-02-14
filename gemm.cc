#include "gemm.h"

#include <x86intrin.h>

#include <cstdlib>
#include <cstring>

#include "utils.h"

void Pack8x4(int64_t width, int64_t depth, int64_t stride, const float *matrix,
             float *packed_matrix) {
  int64_t depth_padded = RoundUp(depth, 4);
  if (depth_padded > depth) {
    memset(packed_matrix + depth * 8, 0,
           sizeof(float) * (depth_padded - depth) * 8);
  }

  const float *data = matrix;
  float *packed_ptr = packed_matrix;
  if (width < 8) {
    for (int64_t i = 0; i < depth; i++) {
      memcpy(packed_ptr, data, width * sizeof(float));
      memset(packed_ptr + width, 0, (8 - width) * sizeof(float));
      packed_ptr += 8;
      data += stride;
    }
  } else {
    for (int64_t i = 0; i < depth; i++) {
      __m256 vec = _mm256_loadu_ps(data);
      _mm256_storeu_ps(packed_ptr, vec);
      packed_ptr += 8;
      data += stride;
    }
  }
}

void Unpack8x8(const float *packed_output, int64_t rows, int64_t cols,
               int64_t stride, float *output) {
  const float *packed_ptr = packed_output;
  float *output_ptr = output;
  if (cols < 8) {
    for (int64_t i = 0; i < rows; i++) {
      memcpy(output_ptr, packed_ptr, cols * sizeof(float));
      packed_ptr += 8;
      output_ptr += stride;
    }
  } else {
    for (int64_t i = 0; i < rows; i++) {
      __m256 vec = _mm256_loadu_ps(packed_ptr);
      _mm256_storeu_ps(output_ptr, vec);
      packed_ptr += 8;
      output_ptr += stride;
    }
  }
}

void ComputeBlock(const float *packed_lhs_data, const float *packed_rhs_data,
                  const int64_t depth_padded, float *packed_output_data) {
  const float *lhs_ptr = packed_lhs_data;
  const float *rhs_ptr = packed_rhs_data;
  const int64_t depth_block_count = depth_padded / 4;

  int64_t r_depth_block_count = depth_block_count;

  asm volatile(R"(
vxorps %%ymm8, %%ymm8, %%ymm8
vxorps %%ymm9, %%ymm9, %%ymm9
vxorps %%ymm10, %%ymm10, %%ymm10
vxorps %%ymm11, %%ymm11, %%ymm11
vxorps %%ymm12, %%ymm12, %%ymm12
vxorps %%ymm13, %%ymm13, %%ymm13
vxorps %%ymm14, %%ymm14, %%ymm14
vxorps %%ymm15, %%ymm15, %%ymm15

loop:

vmovaps 0(%[rhs_ptr]), %%ymm4
vmovaps 32(%[rhs_ptr]), %%ymm5
vmovaps 64(%[rhs_ptr]), %%ymm6
vmovaps 96(%[rhs_ptr]), %%ymm7

vbroadcastss (%[lhs_ptr]), %%ymm0
vbroadcastss 4(%[lhs_ptr]), %%ymm1
vbroadcastss 8(%[lhs_ptr]), %%ymm2
vbroadcastss 12(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm4, %%ymm8
vfmadd231ps %%ymm1, %%ymm4, %%ymm9
vfmadd231ps %%ymm2, %%ymm4, %%ymm10
vfmadd231ps %%ymm3, %%ymm4, %%ymm11

vbroadcastss 16(%[lhs_ptr]), %%ymm0
vbroadcastss 20(%[lhs_ptr]), %%ymm1
vbroadcastss 24(%[lhs_ptr]), %%ymm2
vbroadcastss 28(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm4, %%ymm12
vfmadd231ps %%ymm1, %%ymm4, %%ymm13
vfmadd231ps %%ymm2, %%ymm4, %%ymm14
vfmadd231ps %%ymm3, %%ymm4, %%ymm15

vbroadcastss 32(%[lhs_ptr]), %%ymm0
vbroadcastss 36(%[lhs_ptr]), %%ymm1
vbroadcastss 40(%[lhs_ptr]), %%ymm2
vbroadcastss 44(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm5, %%ymm8
vfmadd231ps %%ymm1, %%ymm5, %%ymm9
vfmadd231ps %%ymm2, %%ymm5, %%ymm10
vfmadd231ps %%ymm3, %%ymm5, %%ymm11

vbroadcastss 48(%[lhs_ptr]), %%ymm0
vbroadcastss 52(%[lhs_ptr]), %%ymm1
vbroadcastss 56(%[lhs_ptr]), %%ymm2
vbroadcastss 60(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm5, %%ymm12
vfmadd231ps %%ymm1, %%ymm5, %%ymm13
vfmadd231ps %%ymm2, %%ymm5, %%ymm14
vfmadd231ps %%ymm3, %%ymm5, %%ymm15

vbroadcastss 64(%[lhs_ptr]), %%ymm0
vbroadcastss 68(%[lhs_ptr]), %%ymm1
vbroadcastss 72(%[lhs_ptr]), %%ymm2
vbroadcastss 76(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm6, %%ymm8
vfmadd231ps %%ymm1, %%ymm6, %%ymm9
vfmadd231ps %%ymm2, %%ymm6, %%ymm10
vfmadd231ps %%ymm3, %%ymm6, %%ymm11

vbroadcastss 80(%[lhs_ptr]), %%ymm0
vbroadcastss 84(%[lhs_ptr]), %%ymm1
vbroadcastss 88(%[lhs_ptr]), %%ymm2
vbroadcastss 92(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm6, %%ymm12
vfmadd231ps %%ymm1, %%ymm6, %%ymm13
vfmadd231ps %%ymm2, %%ymm6, %%ymm14
vfmadd231ps %%ymm3, %%ymm6, %%ymm15

vbroadcastss 96(%[lhs_ptr]), %%ymm0
vbroadcastss 100(%[lhs_ptr]), %%ymm1
vbroadcastss 104(%[lhs_ptr]), %%ymm2
vbroadcastss 108(%[lhs_ptr]), %%ymm3

vfmadd231ps %%ymm0, %%ymm7, %%ymm8
vfmadd231ps %%ymm1, %%ymm7, %%ymm9
vfmadd231ps %%ymm2, %%ymm7, %%ymm10
vfmadd231ps %%ymm3, %%ymm7, %%ymm11

vbroadcastss 112(%[lhs_ptr]), %%ymm0
vbroadcastss 116(%[lhs_ptr]), %%ymm1
vbroadcastss 120(%[lhs_ptr]), %%ymm2
vbroadcastss 124(%[lhs_ptr]), %%ymm3

addq $128, %[lhs_ptr]
addq $128, %[rhs_ptr]

vfmadd231ps %%ymm0, %%ymm7, %%ymm12
vfmadd231ps %%ymm1, %%ymm7, %%ymm13
vfmadd231ps %%ymm2, %%ymm7, %%ymm14
vfmadd231ps %%ymm3, %%ymm7, %%ymm15

addq $-1, %[r_depth_block_count]
cmpq $0, %[r_depth_block_count]

jg loop

vmovaps %%ymm8, (%[packed_output_data])
vmovaps %%ymm9, 32(%[packed_output_data])
vmovaps %%ymm10, 64(%[packed_output_data])
vmovaps %%ymm11, 96(%[packed_output_data])
vmovaps %%ymm12, 128(%[packed_output_data])
vmovaps %%ymm13, 160(%[packed_output_data])
vmovaps %%ymm14, 192(%[packed_output_data])
vmovaps %%ymm15, 224(%[packed_output_data])
  )"
               : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
                 [r_depth_block_count] "+r"(r_depth_block_count)
               : [packed_output_data] "r"(packed_output_data)
               : "cc", "memory");
}