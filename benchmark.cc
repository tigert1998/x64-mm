#include <chrono>
#include <iostream>

int main() {
  int t = 1 << 20;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  asm volatile(R"(
vxorps %%ymm0, %%ymm0, %%ymm0
vxorps %%ymm1, %%ymm1, %%ymm1
vxorps %%ymm2, %%ymm2, %%ymm2
vxorps %%ymm3, %%ymm3, %%ymm3
vxorps %%ymm4, %%ymm4, %%ymm4
vxorps %%ymm5, %%ymm5, %%ymm5
vxorps %%ymm6, %%ymm6, %%ymm6
vxorps %%ymm7, %%ymm7, %%ymm7
vxorps %%ymm8, %%ymm8, %%ymm8
vxorps %%ymm9, %%ymm9, %%ymm9

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

addl $-1, %0
jne loop
  )" ::"r"(t));
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