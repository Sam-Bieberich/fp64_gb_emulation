# Quick Start Guide

## Building Both Benchmarks

```bash
cd /c/VSCode/fp64_gb_emulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build . --config Release -j
```

This will create two executables:
- `dgemm_benchmark.exe` (or `dgemm_benchmark` on Linux)
- `linpack_benchmark.exe` (or `linpack_benchmark` on Linux)

## Running the Benchmarks

### 1. DGEMM Benchmark (Fastest)

```bash
cd build/Release  # or just build/ on Linux
./dgemm_benchmark
```

**What it does:**
- Performs C = A × B matrix multiplication
- N=16384 (three 16384×16384 matrices)
- Compares Pedantic (native FP64) vs Default (emulated FP64)

**Expected output:**
```
Using device: NVIDIA GB100 (SM 110)
Native FP64 (Pedantic) GFLOP/s: 1200-1400
Emulated FP64 (ADP/Default) GFLOP/s: 8000-9000
Performance Speedup (Emulated / Native): 6-7x
```

### 2. Linpack Benchmark (HPL-like)

```bash
./linpack_benchmark
```

**What it does:**
- Solves Ax=b via LU decomposition
- N=16384 (one 16384×16384 matrix + vectors)
- More realistic HPC workload than pure GEMM

**Expected output:**
```
Using device: NVIDIA GB100 (SM 110)
Problem size N=16384 (matrix 16384x16384)
Approximate memory usage: 2.3 GiB

Native FP64 (Pedantic) GFLOP/s: 1000-1200
Emulated FP64 (ADP/Default) GFLOP/s: 7000-8500
Performance Speedup (Emulated / Native): 6-7x
```

## Testing Different Problem Sizes

Both benchmarks accept N as a command-line argument:

```bash
./dgemm_benchmark 8192      # Smaller problem (2 GiB memory)
./dgemm_benchmark 32768     # Larger problem (24 GiB memory)

./linpack_benchmark 8192    # Smaller problem (0.6 GiB memory)
./linpack_benchmark 32768   # Larger problem (8 GiB memory)
```

**Recommended sizes for testing:**
- **N=8192**: Quick test (~30 seconds each)
- **N=16384**: Standard benchmark (your default)
- **N=32768**: Large-scale test (requires >24 GiB for DGEMM, >8 GiB for Linpack)

## Interpreting Results

### Speedup Metric

The key metric is **Speedup = Emulated GFLOP/s / Native GFLOP/s**

**Expected ranges on Blackwell:**
- **6-8x**: Normal ADP speedup (Blackwell's emulation efficiency)
- **<5x**: Potential issue (check CUDA driver, clock throttling)
- **>8x**: Exceptional (verify correctness, may indicate native mode not fully native)

### Absolute Performance

**Native FP64 (Pedantic):**
- GB100: ~1.2-1.5 TFLOP/s (limited by FP64 units)
- GB200: Similar per-GPU

**Emulated FP64 (Default/ADP):**
- GB100: ~8-10 TFLOP/s (tensor core accelerated)
- GB200: Similar per-GPU

### Comparing DGEMM vs Linpack

| Metric | DGEMM | Linpack | Why Different? |
|--------|-------|---------|----------------|
| **FLOP Count** | 2N³ | (2/3)N³ | GEMM does more work |
| **Memory Access** | 3 matrices | 1 matrix + overhead | GEMM more memory intensive |
| **Algorithm** | Single GEMM call | LU + solve (multiple kernels) | Linpack has more overhead |
| **GFLOP/s** | Higher | ~70-85% of GEMM | Normal (Linpack has pivoting, solves) |

**Both should show similar speedup ratios** (within ~10%), validating that emulation works across different workloads.

## Troubleshooting

### Build Errors

**"Cannot find cusolver"**
```bash
# Verify CUDA Toolkit installation
nvcc --version

# Check if cusolver library exists
ls "$CUDA_PATH/lib/x64/cusolver.lib"  # Windows
ls "$CUDA_PATH/lib64/libcusolver.so"  # Linux
```

**"Unsupported GPU architecture"**
```bash
# Check your GPU's SM version
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<your_sm_version>
```

### Runtime Errors

**"Out of memory"**
- Reduce problem size: `./dgemm_benchmark 8192`
- Check available GPU memory: `nvidia-smi`

**"cuBLAS/cuSOLVER error"**
- Update GPU driver to latest version
- Verify CUDA 13.0+ is installed

**Very low performance (<100 GFLOP/s)**
- Check GPU clock throttling: `nvidia-smi -q -d CLOCK`
- Ensure GPU is in performance mode (not idle/low-power)
- Close other GPU applications

## Next Steps

1. **Run both benchmarks** with default N=16384
2. **Compare speedup ratios** between DGEMM and Linpack
3. **Test different sizes** to see scaling behavior
4. **Review** `HPL_INTEGRATION_RESEARCH.md` for background on implementation choices

## Files in This Repository

```
fp64_gb_emulation/
├── dgemm_benchmark.cu              # GEMM benchmark source
├── linpack_benchmark.cu            # Linpack benchmark source
├── CMakeLists.txt                  # Build configuration
├── README.md                       # Main documentation
├── HPL_INTEGRATION_RESEARCH.md    # Research on HPL options
└── QUICKSTART.md                   # This file
```
