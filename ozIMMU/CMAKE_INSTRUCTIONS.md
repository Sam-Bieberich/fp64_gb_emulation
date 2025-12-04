# CMake & Run Instructions

Converted from `ozIMMU/cmake_instructions.txt`.

This file documents the CMake flags, build steps, environment variables, and run examples used on Vista and GB10 (CUDA 13.0) setups.

---
## Vista build & run notes (CUDA 13.0)

ml restore ozaki (personal module I saved with the 13.0 cuda setup required)

```bash
cd build
rm -rf *
cmake -DBUILD_TEST=ON \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    -DCUDAToolkit_ROOT=/home1/apps/nvidia/Linux_aarch64/25.9/cuda/13.0 \
    ..
make -j4
ls -lh main.test
```

This is as close as I got. No matter what I do, the make -j4 command fails, and I think it is because the CUDA installation either does not have cublas installed, or the version is incompatible. I am stopping testing since Todd was able to get some numbers with Vista and a GB200 already. 

## GB10 build & run notes (CUDA 13.0)

If Vista setup has issues, try building against the GB10 CUDA install instead. Example commands:

```bash
cd build
rm -rf *
cmake -DBUILD_TEST=ON \
      -DCMAKE_CUDA_COMPILER=$(which nvcc) \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      ..
make -j4
ls -lh main.test
```

```bash
# Set library path for cuBLAS (GB10)
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Set compute mode (default for tests)
export OZIMMU_COMPUTE_MODE=fp64_int8_9

# Now run the same tests as before. Example:
./main.test normal01 dgemm seq 1024 1024 1 fp64_int8_3 fp64_int8_9 fp64_int8_18 dgemm
```

Notes:
- On GB10 the performance may be lower but accuracy generally improves (except at very low slice counts, e.g., 3 slices).
- If you encounter unexpected runtime behaviour on Vista, verify that `CUDAToolkit_ROOT` and `LD_LIBRARY_PATH` point to the correct CUDA installation used to build and run.

---

## Quick checklist

- Ensure `nvcc` used by CMake matches the CUDA libraries in `LD_LIBRARY_PATH`.
- Use `-DCUDAToolkit_ROOT` to point CMake at the correct CUDA installation.
- After building, confirm `main.test` exists with `ls -lh main.test`.
- Set `OZIMMU_COMPUTE_MODE` to the desired mode before running tests.
- Use the example commands above to compare split levels, sizes, and precisions.

---

### More run examples

Compare all available split levels (3, 6, 9, 12, 15, 18):

```bash
./main.test normal01 dgemm seq 2048 2048 1 \
    fp64_int8_3 fp64_int8_6 fp64_int8_9 fp64_int8_12 fp64_int8_15 fp64_int8_18 dgemm
```

Compare across multiple matrix sizes (test sizes from 1024 to 4096 stepping by 1024):

```bash
./main.test normal01 dgemm seq 1024 4096 1024 fp64_int8_9 dgemm
```

Compare with FP32 (`sgemm`) as well:

```bash
./main.test normal01 dgemm seq 2048 2048 1 fp64_int8_9 dgemm sgemm
```

Test on ill-conditioned matrices (exponential distribution):

```bash
./main.test exp dgemm seq 1024 1024 1 fp64_int8_3 fp64_int8_9 fp64_int8_18 dgemm
```

Save results to CSV:

```bash
./main.test normal01 dgemm seq 1024 4096 1024 \
    fp64_int8_3 fp64_int8_9 fp64_int8_18 dgemm > comparison_results.csv
```

Test syntax reminder:

```text
./main.test <input_dist> <ref_impl> <size_mode> <m_start> <m_end> <m_interval> <mode1> [mode2 ...]
```
