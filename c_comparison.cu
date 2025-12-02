
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

int main() {
    // Matrix dimensions
    int m = 1024, n = 1024, k = 1024;

    // Initialize cuBLAS and cuBLASLt
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // Allocate FP64 matrices
    double *d_A64, *d_B64, *d_C64;
    cudaMalloc(&d_A64, m * k * sizeof(double));
    cudaMalloc(&d_B64, k * n * sizeof(double));
    cudaMalloc(&d_C64, m * n * sizeof(double));

    // Allocate INT8 matrices
    int8_t *d_A8, *d_B8;
    int32_t *d_C8;
    cudaMalloc(&d_A8, m * k * sizeof(int8_t));
    cudaMalloc(&d_B8, k * n * sizeof(int8_t));
    cudaMalloc(&d_C8, m * n * sizeof(int32_t));

    // FP64 GEMM
    double alpha64 = 1.0, beta64 = 0.0;
    auto start_fp64 = std::chrono::high_resolution_clock::now();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha64,
                d_A64, m,
                d_B64, k,
                &beta64,
                d_C64, m);
    cudaDeviceSynchronize();
    auto end_fp64 = std::chrono::high_resolution_clock::now();
    double fp64_time = std::chrono::duration<double>(end_fp64 - start_fp64).count();

    // INT8 GEMM
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, m, k, m);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, k, n, k);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, m, n, m);
    int32_t alpha8 = 1, beta8 = 0;

    auto start_int8 = std::chrono::high_resolution_clock::now();
    cublasLtMatmul(ltHandle,
                   opDesc,
                   &alpha8,
                   d_A8, layoutA,
                   d_B8, layoutB,
                   &beta8,
                   d_C8, layoutC,
                   d_C8, layoutC,
                   nullptr, nullptr, 0, 0);
    cudaDeviceSynchronize();
    auto end_int8 = std::chrono::high_resolution_clock::now();
    double int8_time = std::chrono::duration<double>(end_int8 - start_int8).count();

    // Print results
    std::cout << "FP64 GEMM time: " << fp64_time << " s\n";
    std::cout << "INT8 GEMM time: " << int8_time << " s\n";

    // Cleanup
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);
    cublasDestroy(handle);
    cudaFree(d_A64); cudaFree(d_B64); cudaFree(d_C64);
    cudaFree(d_A8); cudaFree(d_B8); cudaFree(d_C8);

    return 0;
}
