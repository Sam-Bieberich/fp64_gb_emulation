# FP64 Emulation Benchmark (Blackwell GB100/GB200)

This project benchmarks Double-Precision General Matrix Multiplication (DGEMM) on NVIDIA Blackwell GPUs, comparing:

- Native FP64 baseline (cuBLAS Pedantic Math)
- Emulated FP64 via ADP/Tensor Core acceleration (cuBLAS Default Math)

It measures GFLOP/s for `N=16384` and reports the speedup of Emulated over Native.

## Build

Requirements:
- CUDA Toolkit 13.0+ installed and on PATH/CMake findable
- cuBLAS available (bundled with CUDA)

Commands (Windows bash):

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build . --config Release -j
```

Adjust `-DCMAKE_CUDA_ARCHITECTURES` if your device SM differs (e.g., `100`).

## Run

```bash
./dgemm_benchmark            # defaults to N=16384
./dgemm_benchmark 8192       # custom N
```

Output includes GFLOP/s for both modes and the speedup ratio.

## Notes

- Memory footprint for `N=16384` is ~6 GiB for A/B/C combined; ensure sufficient free device memory.
- Uses 2 warm-up iterations and 10 timed iterations, with `cudaEvent_t`.

