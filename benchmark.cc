#include <chrono>
#include <iostream>

int main() {
  int t = 1 << 28;

  float tmp[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float* tmp_ptr = tmp;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  asm volatile(R"(
vbroadcastss 0(%[tmp_ptr]), %%ymm0
vbroadcastss 4(%[tmp_ptr]), %%ymm1
vbroadcastss 8(%[tmp_ptr]), %%ymm2
vbroadcastss 12(%[tmp_ptr]), %%ymm3
vbroadcastss 16(%[tmp_ptr]), %%ymm4
vbroadcastss 20(%[tmp_ptr]), %%ymm5
vbroadcastss 24(%[tmp_ptr]), %%ymm6
vbroadcastss 28(%[tmp_ptr]), %%ymm7
vbroadcastss 32(%[tmp_ptr]), %%ymm8
vbroadcastss 36(%[tmp_ptr]), %%ymm9

loop:

vfmadd231ps %%ymm0, %%ymm0, %%ymm0
vfmadd231ps %%ymm1, %%ymm1, %%ymm1
vfmadd231ps %%ymm2, %%ymm2, %%ymm2
vfmadd231ps %%ymm3, %%ymm3, %%ymm3
vfmadd231ps %%ymm4, %%ymm4, %%ymm4
vfmadd231ps %%ymm5, %%ymm5, %%ymm5
vfmadd231ps %%ymm6, %%ymm6, %%ymm6
vfmadd231ps %%ymm7, %%ymm7, %%ymm7
vfmadd231ps %%ymm8, %%ymm8, %%ymm8
vfmadd231ps %%ymm9, %%ymm9, %%ymm9

addl $-1, %1
jne loop
  )"
               : [tmp_ptr] "+r"(tmp_ptr)
               : "r"(t));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();

  int64_t flops_per_iter = 10 * 8 * 2;
  int64_t flops = flops_per_iter * t;
  double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();
  double flops_per_second = flops / seconds;
  printf("%.4f GFLOPS\n", flops_per_second / (1e9));

  return 0;
}