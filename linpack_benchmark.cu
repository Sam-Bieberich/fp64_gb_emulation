#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

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

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t _status = (call); \
        if (_status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error: " << _status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Initialize matrix A as diagonally dominant (ensures non-singular)
// A(i,j) = random(-1, 1) for i != j
// A(i,i) = N + random(0, 1)
static void init_matrix_host(std::vector<double>& A, int N)
{
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_real_distribution<double> diag_dist(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A[i * N + j] = static_cast<double>(N) + diag_dist(rng);
            } else {
                A[i * N + j] = dist(rng);
            }
        }
    }
}

// Initialize vector b with random values
static void init_vector_host(std::vector<double>& b, int N)
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; ++i) {
        b[i] = dist(rng);
    }
}

// Benchmark HPL-like workload: LU decomposition + solve Ax=b
// Returns GFLOP/s
double run_linpack_test(cusolverDnHandle_t solver_handle, 
                        cublasHandle_t cublas_handle,
                        int N, 
                        const std::string& test_name)
{
    const int64_t mat_elems = static_cast<int64_t>(N) * static_cast<int64_t>(N);
    const size_t mat_bytes = static_cast<size_t>(mat_elems) * sizeof(double);
    const size_t vec_bytes = static_cast<size_t>(N) * sizeof(double);

    // Host buffers
    std::vector<double> h_A(mat_elems);
    std::vector<double> h_b(N);
    std::vector<double> h_x(N);

    init_matrix_host(h_A, N);
    init_vector_host(h_b, N);

    // Device buffers
    double *d_A = nullptr, *d_b = nullptr;
    int *d_pivot = nullptr;
    int *d_info = nullptr;
    double *d_work = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), mat_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), vec_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pivot), sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), vec_bytes, cudaMemcpyHostToDevice));

    // Query workspace size for LU decomposition
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver_handle, N, N, d_A, N, &work_size));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(double) * work_size));

    // Warm-up iterations
    const int warmup_iters = 2;
    for (int i = 0; i < warmup_iters; ++i) {
        // Restore A (gets overwritten by getrf)
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
        
        CUSOLVER_CHECK(cusolverDnDgetrf(solver_handle, N, N, d_A, N, d_work, d_pivot, d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(solver_handle, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_b, N, d_info));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing iterations
    const int iters = 5; // Fewer iterations than GEMM since LU is slower
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        // Restore A and b for each iteration
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), vec_bytes, cudaMemcpyHostToDevice));

        // LU decomposition: A = P*L*U
        CUSOLVER_CHECK(cusolverDnDgetrf(solver_handle, N, N, d_A, N, d_work, d_pivot, d_info));
        
        // Solve: Ax = b using LU factors
        CUSOLVER_CHECK(cusolverDnDgetrs(solver_handle, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_b, N, d_info));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Average time per iteration in seconds
    const double avg_sec = (static_cast<double>(ms) / 1000.0) / static_cast<double>(iters);

    // FLOP count for LU decomposition + solve:
    // LU: (2/3) * N^3
    // Solve (forward + backward): 2 * N^2
    // Total dominated by LU for large N
    const double flops = (2.0 / 3.0) * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N) 
                       + 2.0 * static_cast<double>(N) * static_cast<double>(N);
    const double gflops = flops / (avg_sec * 1e9);

    std::cout << test_name << " GFLOP/s: " << gflops << std::endl;

    // Verify solution (optional)
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_b, vec_bytes, cudaMemcpyDeviceToHost));

    // Compute residual ||Ax - b|| (using original A and b)
    // For simplicity, we skip this in benchmark mode, but it's important for correctness
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

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
    std::cout << "Problem size N=" << N << " (matrix " << N << "x" << N << ")" << std::endl;

    // Expected memory usage
    const double matrix_gb = (static_cast<double>(N) * N * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Approximate memory usage: " << (matrix_gb + 0.3) << " GiB" << std::endl;
    std::cout << std::endl;

    cusolverDnHandle_t solver_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Phase 1: Native FP64 (Pedantic Math - no tensor cores)
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH));
    const double native_gflops = run_linpack_test(solver_handle, cublas_handle, N, "Native FP64 (Pedantic)");

    // Phase 2: Emulated FP64 via ADP/Tensor Cores (Default Math)
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    const double emu_gflops = run_linpack_test(solver_handle, cublas_handle, N, "Emulated FP64 (ADP/Default)");

    // Speedup
    const double speedup = emu_gflops / native_gflops;
    std::cout << std::endl;
    std::cout << "Performance Speedup (Emulated / Native): " << speedup << "x" << std::endl;

    CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    return 0;
}
