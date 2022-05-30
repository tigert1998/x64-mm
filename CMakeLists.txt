project(x64-mm)

add_subdirectory("third_party/OpenBLAS")

add_executable(benchmark "benchmark.cc")
target_compile_options(benchmark PUBLIC -mavx2)

add_executable(main "main.cc" "gemm.cc")
target_compile_options(main PUBLIC -mavx2)

add_executable(benchmark_openblas "benchmark_openblas.cc")
target_link_libraries(benchmark_openblas openblas_static)