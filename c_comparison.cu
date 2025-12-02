
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << _s << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

int main() {
    // Matrix dimensions (change as needed)
    const int m = 1024, n = 1024, k = 1024;

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Host allocations (FP64)
    std::vector<double> h_A(m * k), h_B(k * n), h_C_ref(m * n);

    // Initialize host data with random values in [-1,1]
    for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = (rand() / (double)RAND_MAX) * 2.0 - 1.0;

    // Device allocations FP64
    double *d_A64 = nullptr, *d_B64 = nullptr, *d_C64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A64, sizeof(double) * h_A.size()));
    CUDA_CHECK(cudaMalloc(&d_B64, sizeof(double) * h_B.size()));
    CUDA_CHECK(cudaMalloc(&d_C64, sizeof(double) * h_C_ref.size()));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A64, h_A.data(), sizeof(double) * h_A.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B64, h_B.data(), sizeof(double) * h_B.size(), cudaMemcpyHostToDevice));

    // Warm-up and FP64 GEMM timing using cuda events
    cudaEvent_t start_fp64, stop_fp64;
    CUDA_CHECK(cudaEventCreate(&start_fp64));
    CUDA_CHECK(cudaEventCreate(&stop_fp64));

    const double alpha64 = 1.0, beta64 = 0.0;
    // warmup
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k,
                            &alpha64,
                            d_A64, m,
                            d_B64, k,
                            &beta64,
                            d_C64, m));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_fp64));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k,
                            &alpha64,
                            d_A64, m,
                            d_B64, k,
                            &beta64,
                            d_C64, m));
    CUDA_CHECK(cudaEventRecord(stop_fp64));
    CUDA_CHECK(cudaEventSynchronize(stop_fp64));

    float ms_fp64 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fp64, start_fp64, stop_fp64));
    double fp64_time = ms_fp64 / 1000.0;

    // Copy FP64 result back to host reference
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C64, sizeof(double) * h_C_ref.size(), cudaMemcpyDeviceToHost));

    // Compute GFLOPS for FP64
    double fp64_gflops = (2.0 * (double)m * (double)n * (double)k) / (fp64_time * 1e9);

    // --- INT8 path: quantize on host, run cublasGemmEx with int8 inputs and int32 accumulation ---
    // Compute per-tensor symmetric scales (signed int8)
    double maxA = 0.0, maxB = 0.0;
    for (double v : h_A) maxA = std::max(maxA, std::abs(v));
    for (double v : h_B) maxB = std::max(maxB, std::abs(v));
    double scaleA = (maxA == 0.0) ? 1.0 : (maxA / 127.0);
    double scaleB = (maxB == 0.0) ? 1.0 : (maxB / 127.0);

    // Quantize to int8 (host)
    std::vector<int8_t> h_A8(h_A.size()), h_B8(h_B.size());
    for (size_t i = 0; i < h_A.size(); ++i) {
        int q = (int)std::round(h_A[i] / scaleA);
        if (q < -127) q = -127; if (q > 127) q = 127;
        h_A8[i] = static_cast<int8_t>(q);
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        int q = (int)std::round(h_B[i] / scaleB);
        if (q < -127) q = -127; if (q > 127) q = 127;
        h_B8[i] = static_cast<int8_t>(q);
    }

    // Device allocations for INT8 path
    int8_t *d_A8 = nullptr, *d_B8 = nullptr;
    int32_t *d_C32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A8, sizeof(int8_t) * h_A8.size()));
    CUDA_CHECK(cudaMalloc(&d_B8, sizeof(int8_t) * h_B8.size()));
    CUDA_CHECK(cudaMalloc(&d_C32, sizeof(int32_t) * (m * n)));

    CUDA_CHECK(cudaMemcpy(d_A8, h_A8.data(), sizeof(int8_t) * h_A8.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B8, h_B8.data(), sizeof(int8_t) * h_B8.size(), cudaMemcpyHostToDevice));

    // Prepare alpha/beta int32
    int32_t alpha8 = 1;
    int32_t beta8 = 0;

    // Warm-up for INT8
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha8,
                              d_A8, CUDA_R_8I, m,
                              d_B8, CUDA_R_8I, k,
                              &beta8,
                              d_C32, CUDA_R_32I, m,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time INT8 GEMM
    cudaEvent_t start_int8, stop_int8;
    CUDA_CHECK(cudaEventCreate(&start_int8));
    CUDA_CHECK(cudaEventCreate(&stop_int8));

    CUDA_CHECK(cudaEventRecord(start_int8));
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha8,
                              d_A8, CUDA_R_8I, m,
                              d_B8, CUDA_R_8I, k,
                              &beta8,
                              d_C32, CUDA_R_32I, m,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaEventRecord(stop_int8));
    CUDA_CHECK(cudaEventSynchronize(stop_int8));

    float ms_int8 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_int8, start_int8, stop_int8));
    double int8_time = ms_int8 / 1000.0;

    // Copy int32 result back and dequantize to double
    std::vector<int32_t> h_C32(m * n);
    CUDA_CHECK(cudaMemcpy(h_C32.data(), d_C32, sizeof(int32_t) * h_C32.size(), cudaMemcpyDeviceToHost));

    std::vector<double> h_C_dequant(m * n);
    double out_scale = scaleA * scaleB;
    for (size_t i = 0; i < h_C32.size(); ++i) h_C_dequant[i] = static_cast<double>(h_C32[i]) * out_scale;

    // Compute relative L2 error between h_C_ref and h_C_dequant
    double norm_diff = 0.0, norm_ref = 0.0;
    for (size_t i = 0; i < h_C_ref.size(); ++i) {
        double diff = h_C_ref[i] - h_C_dequant[i];
        norm_diff += diff * diff;
        norm_ref += h_C_ref[i] * h_C_ref[i];
    }
    double rel_l2 = std::sqrt(norm_diff) / (std::sqrt(norm_ref) + 1e-16);

    // Compute GOPS for INT8 (counting MACs similarly)
    double int8_gops = (2.0 * (double)m * (double)n * (double)k) / (int8_time * 1e9);

    // Print parseable results
    std::cout << "FP64 GEMM time: " << fp64_time << " s\n";
    std::cout << "FP64 GFLOP/s: " << fp64_gflops << "\n";
    std::cout << "INT8 GEMM time: " << int8_time << " s\n";
    std::cout << "INT8 GOPS: " << int8_gops << "\n";
    std::cout << "INT8 rel_l2_error: " << rel_l2 << "\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_fp64));
    CUDA_CHECK(cudaEventDestroy(stop_fp64));
    CUDA_CHECK(cudaEventDestroy(start_int8));
    CUDA_CHECK(cudaEventDestroy(stop_int8));

    CUDA_CHECK(cudaFree(d_A64)); CUDA_CHECK(cudaFree(d_B64)); CUDA_CHECK(cudaFree(d_C64));
    CUDA_CHECK(cudaFree(d_A8)); CUDA_CHECK(cudaFree(d_B8)); CUDA_CHECK(cudaFree(d_C32));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
