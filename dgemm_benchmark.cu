#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _status = (call); \
        if (_status != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(_status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _status = (call); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error: " << _status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

static void init_device_matrix(double* d_ptr, int64_t elements, double value)
{
    // Simple kernel-less init via cudaMemset2D is not applicable to doubles; use host buffer
    std::vector<double> h(elements, value);
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
}

// Runs DGEMM N x N and returns GFLOP/s
double run_dgemm_test(cublasHandle_t handle, int N, const std::string& test_name)
{
    const int64_t elems = static_cast<int64_t>(N) * static_cast<int64_t>(N);
    const size_t bytes = static_cast<size_t>(elems) * sizeof(double);

    double *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&C), bytes));

    init_device_matrix(A, elems, 1.0);
    init_device_matrix(B, elems, 1.0);
    init_device_matrix(C, elems, 0.0);

    const double alpha = 1.0;
    const double beta  = 0.0;

    const cublasOperation_t transA = CUBLAS_OP_N;
    const cublasOperation_t transB = CUBLAS_OP_N;

    // Warm-up iterations to stabilize clocks
    const int warmup_iters = 2;
    for (int i = 0; i < warmup_iters; ++i) {
        CUBLAS_CHECK(cublasDgemm(handle,
                                 transA, transB,
                                 N, N, N,
                                 &alpha,
                                 A, N,
                                 B, N,
                                 &beta,
                                 C, N));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing over fixed iterations
    const int iters = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasDgemm(handle,
                                 transA, transB,
                                 N, N, N,
                                 &alpha,
                                 A, N,
                                 B, N,
                                 &beta,
                                 C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Average time per iteration in seconds
    const double avg_sec = (static_cast<double>(ms) / 1000.0) / static_cast<double>(iters);
    const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / (avg_sec * 1e9);

    std::cout << test_name << " GFLOP/s: " << gflops << std::endl;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return gflops;
}

int main(int argc, char** argv)
{
    // Matrix size N=16384 by default; allow override via argv
    int N = 16384;
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N <= 0) N = 16384;
    }

    // Print device info
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device: " << prop.name << " (SM " << prop.major << prop.minor << ")" << std::endl;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Phase 1: Native FP64 Baseline (pedantic math)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const double native_gflops = run_dgemm_test(handle, N, "Native FP64 (Pedantic)");

    // Phase 2: Emulated FP64 via ADP (default math)
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const double emu_gflops = run_dgemm_test(handle, N, "Emulated FP64 (ADP/Default)");

    // Speedup
    const double speedup = emu_gflops / native_gflops;
    std::cout << "Performance Speedup (Emulated / Native): " << speedup << "x" << std::endl;

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
