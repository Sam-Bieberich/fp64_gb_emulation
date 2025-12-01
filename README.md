# FP64 Emulation Benchmark (Blackwell GB100/GB200)

This project benchmarks Double-Precision (FP64) performance on NVIDIA Blackwell GPUs, comparing:

- **Native FP64** baseline (cuBLAS Pedantic Math - no tensor core acceleration)
- **Emulated FP64** via ADP/Tensor Core acceleration (cuBLAS Default Math)

Two benchmarks are provided:

1. **DGEMM Benchmark** (`dgemm_benchmark`): Pure matrix multiplication (C = A × B)
2. **Linpack Benchmark** (`linpack_benchmark`): HPL-like LU solver (Ax = b)

Both measure GFLOP/s for `N=16384` and report the speedup of Emulated over Native.

## Build

Requirements:
- CUDA Toolkit 13.0+ installed and on PATH/CMake findable
- cuBLAS and cuSOLVER available (bundled with CUDA)

Commands (Windows bash):

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build . --config Release -j
```

Adjust `-DCMAKE_CUDA_ARCHITECTURES` if your device SM differs (e.g., `100` for GB100).

## Run

### DGEMM Benchmark

```bash
./dgemm_benchmark            # defaults to N=16384
./dgemm_benchmark 8192       # custom N
```

**Output example:**
```
Using device: NVIDIA GB100 (SM 110)
Native FP64 (Pedantic) GFLOP/s: 1250.5
Emulated FP64 (ADP/Default) GFLOP/s: 8750.2
Performance Speedup (Emulated / Native): 7.0x
```

### Linpack Benchmark (HPL-like)

```bash
./linpack_benchmark          # defaults to N=16384
./linpack_benchmark 8192     # custom N
```

**Output example:**
```
Using device: NVIDIA GB100 (SM 110)
Problem size N=16384 (matrix 16384x16384)
Approximate memory usage: 2.3 GiB

Native FP64 (Pedantic) GFLOP/s: 1100.3
Emulated FP64 (ADP/Default) GFLOP/s: 7650.8
Performance Speedup (Emulated / Native): 6.95x
```

## Notes

### Memory Requirements

| Benchmark | N=16384 Memory Usage | Description |
|-----------|---------------------|-------------|
| **DGEMM** | ~6 GiB | Three N×N matrices (A, B, C) |
| **Linpack** | ~2.3 GiB | One N×N matrix (A) + vectors + workspace |

Ensure sufficient free device memory before running.

### Performance Characteristics

- **DGEMM**: Pure compute-bound workload (BLAS Level 3)
  - FLOP count: 2N³ ≈ 8.79 TFLOP for N=16384
  - Uses 2 warm-up + 10 timed iterations
  
- **Linpack**: LU decomposition + solve (similar to HPL/TOP500)
  - FLOP count: (2/3)N³ + 2N² ≈ 2.93 TFLOP for N=16384
  - Uses 2 warm-up + 5 timed iterations (slower per iteration)
  - Includes pivoting and forward/backward substitution

### Why Two Benchmarks?

1. **DGEMM** is the simplest measure of FP64 compute throughput
2. **Linpack** represents realistic HPC workloads (linear system solving)
3. Both are dominated by the same cuBLAS DGEMM kernel internally
4. Linpack speedup validates emulation across a broader algorithm (not just one kernel)

### Math Modes

- **CUBLAS_PEDANTIC_MATH**: Forces native FP64 execution (no tensor cores)
- **CUBLAS_DEFAULT_MATH**: Enables ADP (Alternating Direction Preconditioner) on Blackwell
  - Uses FP32/TF32 tensor cores with iterative refinement
  - Achieves FP64 accuracy with ~7x throughput boost

## Research Documentation

See [`HPL_INTEGRATION_RESEARCH.md`](HPL_INTEGRATION_RESEARCH.md) for detailed analysis of:
- HPL benchmark integration options
- Why we chose a custom cuSOLVER-based approach
- Comparison with standard HPL
- Alternative implementations

## References

- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [NVIDIA cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [HPL (High-Performance Linpack)](https://www.netlib.org/benchmark/hpl/)
- [TOP500 Supercomputer List](https://www.top500.org/) (uses HPL as benchmark)


