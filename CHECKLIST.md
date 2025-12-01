# Build and Validation Checklist

## Pre-Build Verification

### 1. CUDA Toolkit Installation
```bash
# Verify CUDA is installed and in PATH
nvcc --version
# Expected: CUDA compilation tools, release 13.0 or higher

# Check for required libraries (Windows)
ls "$CUDA_PATH/lib/x64/cublas.lib"
ls "$CUDA_PATH/lib/x64/cusolver.lib"
ls "$CUDA_PATH/lib/x64/cudart.lib"

# Check for required libraries (Linux)
ls $CUDA_PATH/lib64/libcublas.so
ls $CUDA_PATH/lib64/libcusolver.so
ls $CUDA_PATH/lib64/libcudart.so
```

Expected output: All files exist ✓

### 2. CMake Version
```bash
cmake --version
# Expected: cmake version 3.25 or higher
```

### 3. GPU Architecture
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Expected for Blackwell: 10.0 (GB100) or 11.0 (Blackwell Ultra)
```

## Build Steps

### Step 1: Create Build Directory
```bash
cd /c/VSCode/fp64_gb_emulation
mkdir -p build
cd build
```

### Step 2: Configure with CMake
```bash
# For Blackwell Ultra (SM 11.0) - most common
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110

# For GB100 (SM 10.0)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100

# For older Blackwell variants, adjust as needed
```

Expected output:
```
-- The CXX compiler identification is ...
-- The CUDA compiler identification is NVIDIA 13.0...
-- CUDA Architectures: 110
-- Build Type: Release
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
```

### Step 3: Build Executables
```bash
cmake --build . --config Release -j
```

Expected output:
```
[ 25%] Building CUDA object CMakeFiles/dgemm_benchmark.dir/dgemm_benchmark.cu.o
[ 50%] Linking CUDA executable dgemm_benchmark
[ 50%] Built target dgemm_benchmark
[ 75%] Building CUDA object CMakeFiles/linpack_benchmark.dir/linpack_benchmark.cu.o
[100%] Linking CUDA executable linpack_benchmark
[100%] Built target linpack_benchmark
```

### Step 4: Verify Executables
```bash
# Windows
ls Release/dgemm_benchmark.exe
ls Release/linpack_benchmark.exe

# Linux
ls dgemm_benchmark
ls linpack_benchmark
```

Both files should exist ✓

## Runtime Validation

### Test 1: Check GPU Visibility
```bash
nvidia-smi
```

Expected: GPU is visible, not in use by other heavy processes

### Test 2: Run DGEMM Benchmark (Small Problem)
```bash
cd Release  # or stay in build/ on Linux
./dgemm_benchmark 4096
```

Expected output format:
```
Using device: NVIDIA ... (SM ...)
Native FP64 (Pedantic) GFLOP/s: <positive number>
Emulated FP64 (ADP/Default) GFLOP/s: <higher positive number>
Performance Speedup (Emulated / Native): <ratio>x
```

✓ No CUDA errors
✓ No cuBLAS errors
✓ Both GFLOP/s values are positive
✓ Speedup > 1.0

### Test 3: Run Linpack Benchmark (Small Problem)
```bash
./linpack_benchmark 4096
```

Expected output format:
```
Using device: NVIDIA ... (SM ...)
Problem size N=4096 (matrix 4096x4096)
Approximate memory usage: ... GiB

Native FP64 (Pedantic) GFLOP/s: <positive number>
Emulated FP64 (ADP/Default) GFLOP/s: <higher positive number>

Performance Speedup (Emulated / Native): <ratio>x
```

✓ No CUDA errors
✓ No cuBLAS/cuSOLVER errors
✓ Both GFLOP/s values are positive
✓ Speedup > 1.0

### Test 4: Run Full-Size Benchmarks (N=16384)
```bash
# Check available GPU memory first
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# DGEMM requires ~6 GiB free
./dgemm_benchmark

# Linpack requires ~2.3 GiB free
./linpack_benchmark
```

### Test 5: Validate Speedup Consistency
```bash
# Run both benchmarks and compare speedup ratios
./dgemm_benchmark > dgemm_result.txt
./linpack_benchmark > linpack_result.txt

# Extract speedup values and compare
grep "Speedup" dgemm_result.txt
grep "Speedup" linpack_result.txt
```

Expected:
- Both speedups in range 6-8x (typical for Blackwell ADP)
- Speedup ratios differ by <15% (validates emulation works across workloads)
- Example: DGEMM=7.2x, Linpack=6.9x → 4% difference ✓

## Common Issues and Solutions

### Issue 1: "Cannot find cublas/cusolver"
**Symptom:** CMake configure fails with library not found

**Solution:**
```bash
# Set CUDA_PATH environment variable
export CUDA_PATH=/usr/local/cuda-13.0  # Linux
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0  # Windows

# Re-run cmake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
```

### Issue 2: "Out of memory" Runtime Error
**Symptom:** CUDA error at runtime

**Solution:**
```bash
# Check available GPU memory
nvidia-smi

# Reduce problem size
./dgemm_benchmark 8192      # Half size
./linpack_benchmark 8192

# Or close other GPU applications
```

### Issue 3: Very Low Performance (<100 GFLOP/s)
**Symptom:** GFLOP/s values much lower than expected

**Solution:**
```bash
# Check GPU clocks are not throttled
nvidia-smi -q -d CLOCK

# Ensure GPU is in performance mode
nvidia-smi -i 0 -pm 1  # Enable persistence mode

# Set GPU to max clocks (if supported)
nvidia-smi -i 0 -lgc <max_graphics_clock>
```

### Issue 4: Speedup Ratio < 2x
**Symptom:** Emulated mode not much faster than native

**Solution:**
- Verify Blackwell GPU (SM 10.0 or 11.0): `nvidia-smi -q | grep "Compute Capability"`
- Update GPU driver to latest version
- Check CUDA Toolkit is 13.0+ (ADP support): `nvcc --version`
- Verify `CUBLAS_DEFAULT_MATH` enables tensor cores (code review)

### Issue 5: Compilation Warnings
**Symptom:** Build succeeds but with warnings

**Common warnings (safe to ignore):**
- `warning: variable ... set but not used` → Benign
- `warning: unused parameter` → Benign

**Warnings to investigate:**
- `warning: ... will be assumed to be ...` → May indicate incorrect CUDA architecture
- `error:` → Build failed, investigate specific error

## Expected Performance Baselines

### Blackwell GB100 (Single GPU)

| Benchmark | Native FP64 (GFLOP/s) | Emulated FP64 (GFLOP/s) | Speedup |
|-----------|-----------------------|-------------------------|---------|
| DGEMM N=16384 | 1200-1500 | 8000-10000 | 6-8x |
| Linpack N=16384 | 1000-1300 | 7000-9000 | 6-8x |

**If your results are within ±20% of these ranges, your implementation is correct.**

### Sanity Checks
1. Emulated GFLOP/s > Native GFLOP/s ✓
2. Speedup ratio 5-9x (typically 6-8x) ✓
3. Linpack GFLOP/s ≈ 70-85% of DGEMM GFLOP/s ✓ (due to LU overhead)
4. Both modes complete without errors ✓
5. Memory usage matches expectations ✓

## Performance Tuning (Optional)

### Experiment with Problem Sizes
```bash
# Test scaling behavior
for N in 4096 8192 16384 32768; do
    echo "Testing N=$N"
    ./dgemm_benchmark $N
    echo "---"
done
```

### Adjust Iteration Counts
Edit source files if needed:
- `dgemm_benchmark.cu`: Line ~62-63 (warmup_iters, iters)
- `linpack_benchmark.cu`: Line ~104-128 (warmup_iters, iters)

Increase iterations for more stable timing, decrease for faster runs.

### Memory Alignment
cuSOLVER and cuBLAS automatically handle memory alignment. No manual tuning needed for typical use.

## Documentation Review

After successful build and testing:

1. ✓ Read `README.md` for overview
2. ✓ Review `QUICKSTART.md` for usage examples
3. ✓ Consult `HPL_INTEGRATION_RESEARCH.md` for background
4. ✓ Check `SUMMARY.md` for project overview

## Final Checklist

- [ ] CUDA Toolkit 13.0+ installed
- [ ] CMake 3.25+ installed
- [ ] GPU detected by nvidia-smi
- [ ] Build directory created
- [ ] CMake configure successful
- [ ] Build completed without errors
- [ ] `dgemm_benchmark` executable exists
- [ ] `linpack_benchmark` executable exists
- [ ] Both benchmarks run with N=4096 (small test)
- [ ] Both benchmarks run with N=16384 (full test)
- [ ] Speedup ratios in expected range (6-8x)
- [ ] No CUDA/cuBLAS/cuSOLVER runtime errors
- [ ] Performance matches baselines (±20%)

**If all boxes checked: Your FP64 emulation benchmark suite is fully operational!** ✓

## Support and Further Development

### Extending the Benchmarks

**Add more problem sizes:**
```cpp
// In main(), loop over sizes
for (int N : {4096, 8192, 16384, 32768}) {
    run_dgemm_test(handle, N, "DGEMM");
}
```

**Add residual computation (Linpack):**
```cpp
// After solve, verify ||Ax - b|| is small
// See cuSolverDn_LinearSolver CUDA sample for example
```

**Add CSV output:**
```cpp
std::ofstream csv("results.csv");
csv << "N,Native_GFLOPS,Emulated_GFLOPS,Speedup\n";
csv << N << "," << native_gflops << "," << emu_gflops << "," << speedup << "\n";
```

### Contributing

If you enhance these benchmarks:
1. Test on multiple GPU architectures
2. Verify correctness (residual checks)
3. Document new features in README.md
4. Share results with NVIDIA developer forums

### Reporting Issues

If you encounter problems:
1. Check this checklist first
2. Review error messages carefully
3. Verify CUDA installation: `nvcc --version`, `nvidia-smi`
4. Test with smaller problem sizes (N=4096)
5. Consult NVIDIA CUDA documentation for specific API errors

---

**End of Checklist** — Happy Benchmarking! 🚀
