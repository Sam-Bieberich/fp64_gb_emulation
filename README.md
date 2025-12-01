# FP64 Emulation Workflow for Blackwell (GB100/GB200)

This repository provides a practical workflow to enable/disable **FP64 emulation via Tensor Cores** on NVIDIA Blackwell GPUs (CUDA 13.0+). Use this to accelerate double-precision computations in your existing applications.

> **Quick Start:** See [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) for a cheat sheet of environment variables, API calls, and common patterns.
---

## How to Activate FP64 Emulation

### Method 1: Environment Variables (Application-Wide)

Set these before running any CUDA application that uses cuBLAS:

```bash
export CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH           # Enable Tensor Core acceleration
export CUBLAS_EMULATION_STRATEGY=performant           # Use fastest emulation path
# Optional: configure workspace for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

To disable emulation (native FP64 only):

```bash
export CUBLAS_MATH_MODE=CUBLAS_PEDANTIC_MATH          # Force native FP64 units
```

### Method 2: Programmatic Control (In Your Code)

For applications where you control the source, set the math mode via cuBLAS API:

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Enable FP64 emulation (Tensor Cores + ADP)
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

// For explicit control with cublasGemmEx:
// Use computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
// and set cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);

// ... your cublasDgemm or cublasGemmEx calls ...

// To revert to native FP64:
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
```

### Method 3: Comparison Script (Black-Box Apps)

For applications you can't modify (precompiled binaries, scripts), use the included `compare_fp64.sh` wrapper:

```bash
./compare_fp64.sh your_application.sh [args...]
```

This runs your script twice:
1. **Native FP64** (pedantic math)
2. **Emulated FP64** (default math + ADP)

It reports execution time for both runs.

---

## Quick Example

A minimal example is provided in `example_simple.cu` demonstrating FP64 emulation toggle.

### Build the Example

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build . --config Release
```

### Run the Example Directly

```bash
./build/example_simple              # Linux
./build/Release/example_simple.exe  # Windows
```

### Compare with the Script Wrapper

```bash
chmod +x compare_fp64.sh example_app.sh
./compare_fp64.sh ./example_app.sh
```

This runs `example_app.sh` twice (which executes the compiled binary) - once with native FP64, once with emulated FP64 - and reports the speedup.

---

## Compatibility

- **GPU**: NVIDIA Blackwell (GB100, GB200) or newer with SM 100/110+
- **CUDA**: Toolkit 13.0 or later
- **cuBLAS**: Version 13.1+ (bundled with CUDA 13)
- **Driver**: r580+ on Linux/Windows

---

## Notes

- FP64 emulation trades slight numerical precision for significant performance gains (often 2-10x speedup depending on workload)
- Workspace configuration (`CUBLAS_WORKSPACE_CONFIG`) is optional but recommended for reproducible results
- Some workloads may not benefit; profile both modes to confirm speedup
- Check your application's tolerance for reduced precision before deploying emulation in production


