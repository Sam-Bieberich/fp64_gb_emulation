cmake_minimum_required(VERSION 3.25)
project(fp64_emulation_example LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Blackwell architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Simple example executable
add_executable(example_simple example_simple.cu)

target_compile_options(example_simple PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xptxas -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(example_simple PRIVATE CUDA::cublas CUDA::cudart)

if(WIN32)
  target_compile_definitions(example_simple PRIVATE NOMINMAX)
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
