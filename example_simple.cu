#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _status = (call); \
        if (_status != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(_status) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _status = (call); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error: " << _status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

int main()
{
    std::cout << "==================================================" << std::endl;
    std::cout << "FP64 Emulation Toggle Example" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Get device info
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << " (SM " << prop.major << prop.minor << ")" << std::endl;
    std::cout << std::endl;

    // Simple matrix multiply: C = A * B (1024x1024 doubles)
    const int N = 1024;
    const size_t bytes = N * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Initialize with simple pattern
    CUDA_CHECK(cudaMemset(d_A, 0, bytes));
    CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha = 1.0;
    const double beta = 0.0;

    // =====================================================
    // Test 1: Native FP64 (Pedantic Math Mode)
    // =====================================================
    std::cout << "Test 1: Native FP64 (CUBLAS_PEDANTIC_MATH)" << std::endl;
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, N,
                             d_B, N,
                             &beta,
                             d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_native = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_native, start, stop));
    std::cout << "  Runtime: " << ms_native << " ms" << std::endl;
    std::cout << std::endl;

    // =====================================================
    // Test 2: Emulated FP64 (Default Math Mode)
    // =====================================================
    std::cout << "Test 2: Emulated FP64 (CUBLAS_DEFAULT_MATH)" << std::endl;
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, N,
                             d_B, N,
                             &beta,
                             d_C, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_emulated = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_emulated, start, stop));
    std::cout << "  Runtime: " << ms_emulated << " ms" << std::endl;
    std::cout << std::endl;

    // =====================================================
    // Summary
    // =====================================================
    std::cout << "==================================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Native FP64:   " << ms_native << " ms" << std::endl;
    std::cout << "  Emulated FP64: " << ms_emulated << " ms" << std::endl;
    std::cout << "  Speedup:       " << (ms_native / ms_emulated) << "x" << std::endl;
    std::cout << "==================================================" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
