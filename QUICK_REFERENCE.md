# FP64 Emulation Quick Reference

## Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `CUBLAS_MATH_MODE` | `CUBLAS_PEDANTIC_MATH` | Native FP64 only (baseline) |
| `CUBLAS_MATH_MODE` | `CUBLAS_DEFAULT_MATH` | Enable Tensor Core acceleration |
| `CUBLAS_EMULATION_STRATEGY` | `performant` | Use fastest emulation path (recommended) |
| `CUBLAS_EMULATION_STRATEGY` | `eager` | Aggressive optimization (less deterministic) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Small workspace (faster startup) |
| `CUBLAS_WORKSPACE_CONFIG` | `:16384:8` | Larger workspace (better performance) |

## API Calls (C/C++)

```cpp
// Include header
#include <cublas_v2.h>

// Create handle
cublasHandle_t handle;
cublasCreate(&handle);

// Enable emulation
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

// Disable emulation (native FP64)
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

// Cleanup
cublasDestroy(handle);
```

## Common Use Cases

### 1. Enable emulation for entire session

```bash
export CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH
export CUBLAS_EMULATION_STRATEGY=performant
./my_cuda_app
```

### 2. Single command with emulation

```bash
CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH ./my_cuda_app
```

### 3. Compare native vs emulated

```bash
./compare_fp64.sh ./my_app_script.sh
```

### 4. Programmatic toggle in code

```cpp
// Start with native
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
run_critical_section();

// Switch to emulated for performance
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
run_bulk_computation();
```

## Supported Operations

FP64 emulation applies to:
- `cublasDgemm` (double precision matrix multiply)
- `cublasZgemm` (complex double precision)
- `cublasDsyrk` (symmetric rank-k update)
- `cublasDtrsm` (triangular solve)
- Most Level 3 BLAS operations in double precision

Check cuBLAS documentation for full compatibility list.

## Expected Speedups

| Workload Type | Typical Speedup | Notes |
|---------------|-----------------|-------|
| Large GEMM (N>8192) | 5-10x | Best case scenario |
| Medium GEMM (N=2048-8192) | 3-7x | Good acceleration |
| Small GEMM (N<2048) | 1-3x | Limited benefit |
| Mixed operations | 2-5x | Depends on FP64 % |

Actual speedups depend on problem size, memory bandwidth, and GPU utilization.

## Troubleshooting

### No speedup observed?

1. Verify GPU is Blackwell (GB100/GB200) or newer
2. Check `nvidia-smi` shows correct driver version (r580+)
3. Ensure CUDA Toolkit 13.0+ is installed
4. Verify environment variables are set (use `env | grep CUBLAS`)
5. Profile to confirm FP64 operations dominate runtime

### Numerical differences?

FP64 emulation maintains FP64-level accuracy via iterative refinement, but:
- Some edge cases may differ slightly from native
- Non-associative operations may produce different rounding
- For strict reproducibility, use native mode or set workspace config

### Performance regression?

- Small problem sizes don't benefit from emulation overhead
- Memory-bound workloads see less speedup
- Try tuning `CUBLAS_WORKSPACE_CONFIG` for your workload
