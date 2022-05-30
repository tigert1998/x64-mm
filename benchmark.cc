#include <chrono>
#include <iostream>

int main() {
  int t = 1 << 20;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  asm volatile(R"(
loop:

addl $-1, %0

vfmadd231ps %%ymm0, %%ymm1, %%ymm8
vfmadd231ps %%ymm1, %%ymm2 ,%%ymm9
vfmadd231ps %%ymm2, %%ymm3, %%ymm10
vfmadd231ps %%ymm3, %%ymm4, %%ymm11
vfmadd231ps %%ymm4, %%ymm5, %%ymm12
vfmadd231ps %%ymm5, %%ymm6, %%ymm13
vfmadd231ps %%ymm6, %%ymm7, %%ymm14
vfmadd231ps %%ymm7, %%ymm0, %%ymm15

cmpl $0, %0

jg loop
  )" ::"r"(t));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();

  int64_t flops_per_iter = 8 * 8 * 2;
  int64_t flops = flops_per_iter * t;
  double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();
  double flops_per_second = flops / seconds;
  printf("%.4f GFLOPS\n", flops_per_second / (1e9));

  return 0;
}