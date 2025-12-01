# HPL Integration Research for FP64 cuBLAS Benchmarking

## Executive Summary

**Recommended Approach:** Build a **custom HPL-like benchmark** using cuBLAS/cuSOLVER rather than integrating full HPL.

**Rationale:**
- Standard HPL requires MPI and is designed for multi-node clusters
- NVIDIA does not provide an official HPL-CUDA library or simple API
- For single-GPU benchmarking with math mode control, a custom LU-based solver provides equivalent metrics with far simpler integration
- HPL.dat configuration and standalone executable architecture makes programmatic math mode switching complex

---

## 1. Available HPL Implementations for NVIDIA GPUs

### 1.1 Standard HPL (Netlib HPL 2.3)
- **Source:** https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
- **Description:** CPU-based distributed memory implementation
- **Requirements:**
  - MPI (Message Passing Interface) - **mandatory**
  - BLAS library (can use cuBLAS)
  - Designed for cluster computing (process grid P×Q)
- **GPU Acceleration:** Can link against cuBLAS for `DGEMM`, but core algorithm remains CPU-driven
- **Windows Support:** Limited; primarily Linux-focused

### 1.2 NVIDIA's HPL Binaries
- **Availability:** NVIDIA provides pre-compiled HPL binaries as part of HPC SDK for **Linux only**
- **Location:** Part of NVIDIA HPC SDK (formerly PGI compilers)
- **Limitations:**
  - Not available for Windows
  - No library API—standalone executable only
  - Designed for multi-GPU, multi-node clusters
  - Cannot programmatically control cuBLAS math modes mid-run

### 1.3 HPL-GPU Variants
- **Examples:** 
  - HPL-NVIDIA (proprietary, used for TOP500 submissions)
  - Community ports (GitHub, limited maintenance)
- **Status:** 
  - No officially maintained open-source HPL-GPU with cuBLAS backend for Windows
  - NVIDIA's internal HPL for benchmarking is not publicly released as a library

### 1.4 Conclusion on HPL Availability
**No turnkey HPL solution exists** that:
- Works on Windows with CUDA 13+
- Exposes a library API (vs. standalone executable)
- Allows programmatic switching of cuBLAS math modes
- Supports single-GPU runs without MPI overhead

---

## 2. Integration Approaches (If Using Standard HPL)

### 2.1 Build from Source (Netlib HPL)

#### Requirements
- **MPI Implementation:**
  - **Windows:** MS-MPI SDK (https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  - Install MS-MPI SDK and runtime
- **BLAS:** cuBLAS (CUDA Toolkit)
- **Compiler:** Visual Studio 2019+ or MinGW with CUDA support

#### Build Steps
```bash
# Download HPL 2.3
wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xzf hpl-2.3.tar.gz
cd hpl-2.3

# Create Make.<arch> configuration
cp setup/Make.Linux_PII_CBLAS Make.Windows_cuBLAS

# Edit Make.Windows_cuBLAS:
# - Set TOPdir = $(HOME)/hpl-2.3
# - Set MPdir = C:/Program Files (x86)/Microsoft SDKs/MPI
# - Set LAdir = <CUDA_PATH>/lib/x64
# - Set LAinc = <CUDA_PATH>/include
# - Set LAlib = -lcublas
# - Set CC = cl.exe (or gcc)
# - Set LINKER = link.exe (or gcc)

make arch=Windows_cuBLAS
```

#### CMake Integration (Hypothetical)
```cmake
include(ExternalProject)

ExternalProject_Add(
    HPL
    URL https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy 
        ${CMAKE_CURRENT_SOURCE_DIR}/Make.Windows_cuBLAS 
        <SOURCE_DIR>/Make.Windows_cuBLAS
    BUILD_COMMAND make arch=Windows_cuBLAS
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)
```

**Issues:**
- HPL Makefile system not CMake-native (requires manual `Make.<arch>` setup)
- Windows build not officially supported
- No library output—produces `xhpl` executable only

### 2.2 Dependencies Beyond CUDA/cuBLAS

| Dependency | Purpose | Notes |
|------------|---------|-------|
| **MPI** | Inter-process communication | Required even for single-node runs |
| **BLAS** | Matrix operations | Can use cuBLAS for `DGEMM` |
| **CBLAS (optional)** | C interface to BLAS | Not strictly required |
| **LAPACK (optional)** | Panel factorization | HPL has built-in routines |

**For Single-GPU:** MPI is still required for HPL process grid initialization, even with P=1, Q=1.

---

## 3. HPL API and Control

### 3.1 Execution Model
HPL is **not a library**—it's a standalone program:

```bash
mpirun -np 1 ./xhpl
```

**Input:** `HPL.dat` configuration file (must be in working directory)

**Output:** Text printed to stdout with GFLOP/s results

### 3.2 HPL.dat Configuration
```plaintext
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
16384        Ns
1            # of NBs
256          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

**Key Parameters:**
- **N:** Problem size (matrix dimension)
- **NB:** Block size for panel factorization
- **P × Q:** Process grid (1×1 for single GPU)

### 3.3 Programmatic Control (Not Possible)

**Problem:** HPL reads `HPL.dat` at startup—no API to:
- Set cuBLAS math mode programmatically
- Run multiple configurations in one process
- Parse results without text scraping

**Workaround (Fragile):**
```cpp
// Pseudo-code - NOT RECOMMENDED
for (auto mode : {CUBLAS_PEDANTIC_MATH, CUBLAS_DEFAULT_MATH}) {
    cublasSetMathMode(globalHandle, mode);
    
    // Write HPL.dat
    std::ofstream dat("HPL.dat");
    dat << "... N=16384 ...\n";
    dat.close();
    
    // Fork process
    system("mpirun -np 1 ./xhpl > output.txt");
    
    // Parse output.txt for GFLOP/s
    std::ifstream result("output.txt");
    // ... regex search for "WR.*Gflops" ...
}
```

**Issues:**
- Requires MPI runtime on Windows
- cuBLAS handle in parent process doesn't affect child `xhpl` process
- Requires forking/spawning processes (unreliable)

---

## 4. Problem Sizing for N=16384

### 4.1 HPL.dat Parameters

```plaintext
N     = 16384         # Matrix dimension
NB    = 256           # Block size (tunable)
P     = 1             # Process rows
Q     = 1             # Process columns
```

### 4.2 Memory Requirements

**Total Memory:**
```
Matrix A: N × N × 8 bytes (FP64) = 16384² × 8 = 2,147,483,648 bytes ≈ 2 GiB
Matrix B: N × 1 × 8 bytes        = 16384 × 8   = 131,072 bytes    ≈ 128 KiB
Matrix X: N × 1 × 8 bytes        = 16384 × 8   = 131,072 bytes    ≈ 128 KiB
Workspace: ~NB × N × 8 (panel)   = 256 × 16384 × 8 ≈ 32 MiB
──────────────────────────────────────────────────────────────────
Total:  ~2.2 GiB
```

For your existing DGEMM benchmark:
```cpp
A, B, C: 3 × (16384² × 8) = 6 GiB  // Three full matrices
```

HPL uses **less memory** since it solves Ax=b (one matrix + vectors vs. three matrices).

### 4.3 Optimal NB (Block Size) for Blackwell

**General Guidance:**
- **NB = 256** to **512** for modern GPUs (Ampere/Blackwell)
- Larger NB → better GEMM performance (more work per kernel launch)
- Smaller NB → better parallelism in panel factorization

**For Blackwell (SM 10.0/11.0):**
- Start with **NB = 256** (safe default)
- Tune upward: 384, 512 (diminishing returns beyond 512)
- HPL performance is ~70-90% of peak `DGEMM` GFLOP/s

**Tuning Process:**
1. Run HPL with NB ∈ {128, 192, 256, 384, 512}
2. Select NB with highest GFLOP/s
3. Typically NB=256 is near-optimal for most GPUs

### 4.4 P × Q Process Grid

**Single GPU:** Always use **P=1, Q=1**

**Multi-GPU (if scaling):**
- Use square grids when possible: P=Q (e.g., 2×2, 4×4)
- Prefer row-major mapping (PMAP=0)

---

## 5. Alternative: Custom HPL-like Benchmark

### 5.1 Rationale

Instead of integrating full HPL, implement the core algorithm using cuBLAS/cuSOLVER:

**Algorithm:** LU factorization with partial pivoting → solve Ax=b

**Advantages:**
- **Full control** over cuBLAS math modes
- **No MPI dependency**
- **Windows compatible**
- **Programmatic API** (no text file I/O)
- **Simpler CMake integration**

**Trade-offs:**
- Not "official" HPL (but equivalent workload)
- No process grid (single-GPU only)
- Slightly different optimization strategies than HPL's look-ahead

### 5.2 Core Algorithm

HPL solves **Ax=b** using:
1. **LU Decomposition:** A = P × L × U (with partial pivoting)
2. **Forward Substitution:** Solve Ly = Pb for y
3. **Backward Substitution:** Solve Ux = y for x

**FLOP Count:**
```
LU decomposition: (2/3) × N³ FLOPS
Solve (forward + backward): 2 × N² FLOPS
Total: (2/3) × N³ + 2N² ≈ (2/3) × N³ (for large N)
```

For N=16384:
```
FLOPS = (2/3) × 16384³ = 2.93 × 10^12 ≈ 2.93 TFLOPS
```

### 5.3 Implementation Using cuSOLVER

**cuSOLVER** provides `cusolverDnDgetrf` (LU decomposition) and `cusolverDnDgetrs` (solve).

**Sample Code:**

```cpp
#include <cusolverDn.h>
#include <cublas_v2.h>

double benchmark_hpl_like(int N, cublasMath_t math_mode) {
    cusolverDnHandle_t cusolverH;
    cublasHandle_t     cublasH;
    
    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);
    cublasSetMathMode(cublasH, math_mode);  // Control math mode
    
    // Allocate A (N×N), b (N×1), x (N×1)
    double *d_A, *d_b, *d_x;
    int    *d_pivot;
    cudaMalloc(&d_A, sizeof(double) * N * N);
    cudaMalloc(&d_b, sizeof(double) * N);
    cudaMalloc(&d_x, sizeof(double) * N);
    cudaMalloc(&d_pivot, sizeof(int) * N);
    
    // Initialize A (random SPD or diagonally dominant)
    // ... (use cuRAND or CPU initialization)
    
    // LU decomposition: A = P × L × U
    int bufferSize = 0;
    cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_A, N, &bufferSize);
    
    double *d_buffer;
    cudaMalloc(&d_buffer, sizeof(double) * bufferSize);
    
    int *d_info;
    cudaMalloc(&d_info, sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    cusolverDnDgetrf(cusolverH, N, N, d_A, N, d_buffer, d_pivot, d_info);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    cusolverDnDgetrf(cusolverH, N, N, d_A, N, d_buffer, d_pivot, d_info);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Compute GFLOP/s
    double gflops = ((2.0 / 3.0) * N * N * N) / (ms * 1e6);  // GFLOP/s
    
    // Solve Ax=b (optional, adds ~2N² FLOPS)
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_x, N, d_info);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_pivot);
    cudaFree(d_buffer); cudaFree(d_info);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
    
    return gflops;
}

int main() {
    int N = 16384;
    
    double native_gflops = benchmark_hpl_like(N, CUBLAS_PEDANTIC_MATH);
    double emu_gflops    = benchmark_hpl_like(N, CUBLAS_DEFAULT_MATH);
    
    std::cout << "Native FP64 (Pedantic): " << native_gflops << " GFLOP/s\n";
    std::cout << "Emulated FP64 (ADP):    " << emu_gflops << " GFLOP/s\n";
    std::cout << "Speedup: " << (emu_gflops / native_gflops) << "x\n";
    
    return 0;
}
```

### 5.4 Why This Is Equivalent to HPL

| Metric | HPL | Custom cuSOLVER Benchmark |
|--------|-----|---------------------------|
| **Core Operation** | LU decomposition (DGETRF) | LU decomposition (cusolverDnDgetrf) |
| **FLOP Count** | (2/3)N³ + 2N² | (2/3)N³ + 2N² (identical) |
| **Dominant Kernel** | DGEMM (via BLAS) | DGEMM (via cuBLAS, called internally by cuSOLVER) |
| **Math Mode Control** | Global cuBLAS handle (hard to set) | Direct `cublasSetMathMode()` before each run |
| **Result** | GFLOP/s for solving Ax=b | GFLOP/s for solving Ax=b |

**Key Insight:** HPL's performance is dominated by **DGEMM** calls during LU factorization. cuSOLVER's `Dgetrf` uses cuBLAS internally, so you're benchmarking the same operations.

### 5.5 CMake Integration

```cmake
find_package(CUDAToolkit REQUIRED)

add_executable(hpl_benchmark hpl_benchmark.cu)

target_link_libraries(hpl_benchmark PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
    CUDA::curand  # If using device-side random init
)
```

**No external dependencies** beyond CUDA Toolkit.

---

## 6. Comparison: HPL vs. DGEMM Benchmark

| Aspect | Your Current DGEMM | HPL (Standard) | Custom LU Solver |
|--------|-------------------|----------------|------------------|
| **Operation** | C = A × B | Solve Ax=b via LU | Solve Ax=b via LU |
| **FLOP Count (N=16384)** | 2N³ = 8.79 TFLOP | (2/3)N³ = 2.93 TFLOP | (2/3)N³ = 2.93 TFLOP |
| **Memory (N=16384)** | 6 GiB (A, B, C) | ~2.2 GiB (A, b, workspace) | ~2.2 GiB |
| **Math Mode Control** | ✅ Easy (`cublasSetMathMode`) | ❌ Hard (fork processes) | ✅ Easy |
| **Windows Support** | ✅ Native | ❌ Limited (MPI issues) | ✅ Native |
| **Industry Standard** | BLAS benchmark | **Linpack (TOP500 metric)** | Similar to Linpack |
| **Complexity** | Simple (1 cuBLAS call) | Complex (MPI, HPL.dat, build) | Moderate (cuSOLVER API) |

**Recommendation:** 
- If you need the **HPL name** for official comparisons → Use custom LU solver and call it "HPL-like" or "Linpack-style"
- If you want **simplicity** → Stick with DGEMM benchmark (already implemented)
- If you want **broader coverage** → Add both DGEMM + LU solver benchmarks

---

## 7. Final Recommendations

### Option A: Custom LU Solver (Recommended)

**Pros:**
- Clean, self-contained C++ code
- Full control over cuBLAS math modes
- No MPI dependency
- Equivalent FLOP workload to HPL
- Windows compatible

**Implementation:**
```cpp
// See Section 5.3 for full code
double gflops = benchmark_hpl_like(16384, CUBLAS_PEDANTIC_MATH);
```

**Integration:**
```cmake
add_executable(linpack_benchmark linpack_benchmark.cu)
target_link_libraries(linpack_benchmark PRIVATE CUDA::cublas CUDA::cusolver CUDA::cudart)
```

### Option B: Keep DGEMM Benchmark

**Pros:**
- Already implemented and working
- Simpler than LU decomposition
- Still a valid HPC benchmark (BLAS Level 3)

**Enhancement:**
Add cuSOLVER LU benchmark as second metric:

```cpp
int main() {
    double dgemm_native = run_dgemm_test(handle, 16384, CUBLAS_PEDANTIC_MATH);
    double dgemm_emu    = run_dgemm_test(handle, 16384, CUBLAS_DEFAULT_MATH);
    
    double lu_native = run_lu_solver(handle, 16384, CUBLAS_PEDANTIC_MATH);
    double lu_emu    = run_lu_solver(handle, 16384, CUBLAS_DEFAULT_MATH);
    
    std::cout << "DGEMM Speedup: " << (dgemm_emu / dgemm_native) << "x\n";
    std::cout << "LU Solver Speedup: " << (lu_emu / lu_native) << "x\n";
}
```

### Option C: Official HPL (Not Recommended for Your Use Case)

**Only if:**
- You need to submit to TOP500 (requires official HPL)
- You have multi-node cluster with Linux
- You can tolerate MPI complexity

**For single-GPU Windows benchmarking:** Not worth the effort.

---

## 8. Additional Resources

### cuSOLVER Documentation
- **API Reference:** https://docs.nvidia.com/cuda/cusolver/index.html
- **Dgetrf (LU decomposition):** https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-getrf
- **Dgetrs (solve):** https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-getrs

### CUDA Samples
- **cuSolverDn_LinearSolver:** CUDA Samples → `4_CUDA_Libraries/cuSolverDn_LinearSolver`
  - Demonstrates Cholesky, LU, QR solvers
  - Includes timing and residual computation

### HPL References
- **HPL Documentation:** https://www.netlib.org/benchmark/hpl/documentation.html
- **HPL Tuning:** https://www.netlib.org/benchmark/hpl/tuning.html
- **TOP500 List:** https://www.top500.org/ (uses HPL as official benchmark)

### Math Mode Documentation
- **cuBLAS Math Modes:** https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
- **CUBLAS_PEDANTIC_MATH:** Forces FP64 (no tensor core acceleration)
- **CUBLAS_DEFAULT_MATH:** Allows ADP/tensor cores for FP64 emulation

---

## 9. Conclusion

**For your project (single-GPU, Windows, programmatic math mode control):**

1. **Best Approach:** Implement custom LU solver using cuSOLVER
   - Equivalent to HPL workload
   - Full control over cuBLAS math modes
   - No external dependencies beyond CUDA Toolkit

2. **Alternative:** Keep existing DGEMM benchmark
   - Simpler, already working
   - Add LU solver as optional second metric

3. **Avoid:** Integrating standard HPL
   - Requires MPI (complex on Windows)
   - No programmatic API
   - Cannot switch math modes easily

**Sample Code Structure:**

```
fp64_gb_emulation/
├── dgemm_benchmark.cu       # Your existing GEMM benchmark
├── linpack_benchmark.cu     # New LU solver benchmark (recommended)
├── CMakeLists.txt           # Add cusolver linkage
└── README.md
```

**Next Steps:**
1. Implement `linpack_benchmark.cu` using cuSOLVER API (see Section 5.3)
2. Add CMake target linking `CUDA::cusolver`
3. Run both DGEMM + Linpack benchmarks with Pedantic vs. Default math modes
4. Compare speedup ratios

This approach gives you **HPL-equivalent metrics** with **minimal complexity** and **full Windows support**.
