cmake_minimum_required(VERSION 3.25)
project(fp64_gb_emulation LANGUAGES CXX CUDA)

# Configure CUDA standard and architecture
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target Blackwell GPUs; adjust if your device differs
# Common values: 100 (GB100), 110 (Blackwell Ultra). Use 110 by default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA toolkit and libraries
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# DGEMM Benchmark (existing)
# ============================================================================
add_executable(dgemm_benchmark dgemm_benchmark.cu)

# Compiler options
target_compile_options(dgemm_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xptxas -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Link cuBLAS (and CUDA runtime)
target_link_libraries(dgemm_benchmark PRIVATE CUDA::cublas CUDA::cudart)

# ============================================================================
# Linpack Benchmark (HPL-like LU solver)
# ============================================================================
add_executable(linpack_benchmark linpack_benchmark.cu)

# Compiler options
target_compile_options(linpack_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xptxas -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Link cuBLAS, cuSOLVER, and CUDA runtime
target_link_libraries(linpack_benchmark PRIVATE 
  CUDA::cublas 
  CUDA::cusolver 
  CUDA::cudart
)

# On Windows, ensure proper subsystem and definitions
if(WIN32)
  target_compile_definitions(dgemm_benchmark PRIVATE NOMINMAX)
endif()

# Print summary
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
