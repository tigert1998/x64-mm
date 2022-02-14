project(x64-mm)

add_executable(benchmark "benchmark.cc")

add_executable(main "main.cc" "gemm.cc")
target_compile_options(main PUBLIC -mavx2)