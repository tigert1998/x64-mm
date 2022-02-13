#include <chrono>
#include <iostream>
#include <random>

float a[32], b[32], c[64];
float tmp0[8], tmp1[8];

int main() {
  std::default_random_engine engine;
  for (int i = 0; i < 32; i++) {
    a[i] = std::uniform_real_distribution<>(-1, 1)(engine);
    b[i] = std::uniform_real_distribution<>(-1, 1)(engine);
  }

  int t = 1 << 10;

  double seconds = 0;
  for (int i = 0; i < t; i++) {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    asm volatile(R"(
vxorps %%ymm8, %%ymm8, %%ymm8
vxorps %%ymm9, %%ymm9, %%ymm9
vxorps %%ymm10, %%ymm10, %%ymm10
vxorps %%ymm11, %%ymm11, %%ymm11
vxorps %%ymm12, %%ymm12, %%ymm12
vxorps %%ymm13, %%ymm13, %%ymm13
vxorps %%ymm14, %%ymm14, %%ymm14
vxorps %%ymm15, %%ymm15, %%ymm15

vmovaps 0(%1), %%ymm4
vmovaps 32(%1), %%ymm5
vmovaps 64(%1), %%ymm6
vmovaps 96(%1), %%ymm7

vbroadcastss (%0), %%ymm0
vbroadcastss 4(%0), %%ymm1
vbroadcastss 8(%0), %%ymm2
vbroadcastss 12(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm4, %%ymm8
vfmadd231ps %%ymm1, %%ymm4, %%ymm9
vfmadd231ps %%ymm2, %%ymm4, %%ymm10
vfmadd231ps %%ymm3, %%ymm4, %%ymm11

vbroadcastss 16(%0), %%ymm0
vbroadcastss 20(%0), %%ymm1
vbroadcastss 24(%0), %%ymm2
vbroadcastss 28(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm4, %%ymm12
vfmadd231ps %%ymm1, %%ymm4, %%ymm13
vfmadd231ps %%ymm2, %%ymm4, %%ymm14
vfmadd231ps %%ymm3, %%ymm4, %%ymm15

vbroadcastss 32(%0), %%ymm0
vbroadcastss 36(%0), %%ymm1
vbroadcastss 40(%0), %%ymm2
vbroadcastss 44(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm5, %%ymm8
vfmadd231ps %%ymm1, %%ymm5, %%ymm9
vfmadd231ps %%ymm2, %%ymm5, %%ymm10
vfmadd231ps %%ymm3, %%ymm5, %%ymm11

vbroadcastss 48(%0), %%ymm0
vbroadcastss 52(%0), %%ymm1
vbroadcastss 56(%0), %%ymm2
vbroadcastss 60(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm5, %%ymm12
vfmadd231ps %%ymm1, %%ymm5, %%ymm13
vfmadd231ps %%ymm2, %%ymm5, %%ymm14
vfmadd231ps %%ymm3, %%ymm5, %%ymm15

vbroadcastss 64(%0), %%ymm0
vbroadcastss 68(%0), %%ymm1
vbroadcastss 72(%0), %%ymm2
vbroadcastss 76(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm6, %%ymm8
vfmadd231ps %%ymm1, %%ymm6, %%ymm9
vfmadd231ps %%ymm2, %%ymm6, %%ymm10
vfmadd231ps %%ymm3, %%ymm6, %%ymm11

vbroadcastss 80(%0), %%ymm0
vbroadcastss 84(%0), %%ymm1
vbroadcastss 88(%0), %%ymm2
vbroadcastss 92(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm6, %%ymm12
vfmadd231ps %%ymm1, %%ymm6, %%ymm13
vfmadd231ps %%ymm2, %%ymm6, %%ymm14
vfmadd231ps %%ymm3, %%ymm6, %%ymm15

vbroadcastss 96(%0), %%ymm0
vbroadcastss 100(%0), %%ymm1
vbroadcastss 104(%0), %%ymm2
vbroadcastss 108(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm7, %%ymm8
vfmadd231ps %%ymm1, %%ymm7, %%ymm9
vfmadd231ps %%ymm2, %%ymm7, %%ymm10
vfmadd231ps %%ymm3, %%ymm7, %%ymm11

vbroadcastss 112(%0), %%ymm0
vbroadcastss 116(%0), %%ymm1
vbroadcastss 120(%0), %%ymm2
vbroadcastss 124(%0), %%ymm3

vfmadd231ps %%ymm0, %%ymm7, %%ymm12
vfmadd231ps %%ymm1, %%ymm7, %%ymm13
vfmadd231ps %%ymm2, %%ymm7, %%ymm14
vfmadd231ps %%ymm3, %%ymm7, %%ymm15

vmovaps %%ymm8, (%2)
vmovaps %%ymm9, 32(%2)
vmovaps %%ymm10, 64(%2)
vmovaps %%ymm11, 96(%2)
vmovaps %%ymm12, 128(%2)
vmovaps %%ymm13, 160(%2)
vmovaps %%ymm14, 192(%2)
vmovaps %%ymm15, 224(%2)
  )" ::"r"(a),
                 "r"(b), "r"(c));

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    seconds +=
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
  }

  int flag = 1;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      float tot = 0;
      for (int k = 0; k < 4; k++) {
        tot += a[k * 8 + i] * b[k * 8 + j];
      }
      if (std::abs(tot - c[i * 8 + j]) > 1e-4) {
        flag = 0;
        break;
      }
    }

  puts(flag ? "PASS" : "FAIL");

  if (flag) {
    int64_t flops_per_iter = 8 * 4 * 8 * 2;
    int64_t flops = flops_per_iter * t;
    double flops_per_second = flops / seconds;
    printf("%.4f GFLOPS\n", flops_per_second / (1 << 30));
  }

  return 0;
}