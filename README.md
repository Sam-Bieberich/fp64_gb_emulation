# FP64 Emulation Workflow for Blackwell (GB100/GB200)

This repository provides a practical workflow to enable/disable **FP64 emulation via Tensor Cores** on NVIDIA Blackwell GPUs (CUDA 13.0+). Use this to accelerate double-precision computations in your existing applications.

> **Works with any language:** Python (CuPy, PyTorch, TensorFlow), Julia, Fortran, MATLAB, C/C++, or any framework using cuBLAS.

> **Quick Start:** See [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) for a cheat sheet of environment variables and [`PYTHON_SETUP.md`](PYTHON_SETUP.md) for Python-specific installation and examples.

---

## How to Activate FP64 Emulation

### Method 1: Environment Variables (Universal - Works with Any Language)

**This method works with ALL languages** (Python, Julia, MATLAB, C/C++, etc.)

Set these before running any CUDA application that uses cuBLAS:

```bash
export CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH           # Enable Tensor Core acceleration
export CUBLAS_EMULATION_STRATEGY=performant           # Use fastest emulation path
# Optional: configure workspace for determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**Example with Python:**
```bash
export CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH
python your_script.py  # Automatically uses emulated FP64 in cuBLAS calls
### Method 2: Programmatic Control (Language-Specific)

**C/C++ (Direct cuBLAS API):**
```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // Enable emulation
// ... your cublasDgemm calls ...
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH); // Disable emulation
```

**Python (via environment - recommended):**
```python
import os
os.environ['CUBLAS_MATH_MODE'] = 'CUBLAS_DEFAULT_MATH'
import cupy as cp  # CuPy will use emulated FP64
# or PyTorch, TensorFlow, etc.
```

**Note:** For C/C++, you can also use explicit control with `cublasGemmEx`:
```cpp
// Use computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
// and set cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
```For explicit control with cublasGemmEx:
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
---

## Quick Examples

### Python Example (Recommended for Most Users)

**Environment Setup:**

> **Setting up a virtual environment (recommended):**
> ```bash
> # Create a new virtual environment
> python -m venv fp64_env
> 
> # Activate it (Linux/Mac)
> source fp64_env/bin/activate
> 
> # Activate it (Windows bash/Git Bash)
> source fp64_env/Scripts/activate
> 
> # Activate it (Windows CMD)
> fp64_env\Scripts\activate.bat
> 
> # Install CuPy (requires CUDA Toolkit 13.0+ already installed)
> pip install cupy-cuda13x
> 
> # Verify installation
> python -c "import cupy as cp; print(f'CuPy {cp.__version__}')"
> ```

**Or install globally:**
```bash
# Install CuPy (requires CUDA Toolkit 13.0+ already installed)
pip install cupy-cuda13x

# Verify installation
python -c "import cupy as cp; print(f'CuPy {cp.__version__}')"
```

**Run the example:**
```bash
# Test with native FP64
CUBLAS_MATH_MODE=CUBLAS_PEDANTIC_MATH python example_simple.py

# Test with emulated FP64
CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH python example_simple.py

# Or use the comparison script
chmod +x run_python.sh compare_fp64.sh
./compare_fp64.sh ./run_python.sh
```

The Python example (`example_simple.py`) uses CuPy for matrix multiplication. It automatically uses cuBLAS DGEMM internally, which respects the `CUBLAS_MATH_MODE` environment variable.

**See [`PYTHON_SETUP.md`](PYTHON_SETUP.md) for detailed installation, troubleshooting, and examples with PyTorch and TensorFlow.**

---

### C/C++ Example

A minimal C++ example is provided in `example_simple.cu` demonstrating FP64 emulation toggle.
## Compatibility

- **GPU**: NVIDIA Blackwell (GB100, GB200) or newer with SM 100/110+
- **CUDA**: Toolkit 13.0 or later
- **cuBLAS**: Version 13.1+ (bundled with CUDA 13)
- **Driver**: r580+ on Linux/Windows
- **Languages**: 
  - **Python**: CuPy 13.0+, PyTorch 2.2+, TensorFlow 2.15+
  - **Julia**: CUDA.jl with CUDA 13+
  - **C/C++**: Direct cuBLAS API
  - **MATLAB**: R2024a+ with Parallel Computing Toolbox
  - **Fortran**: Any version linking to cuBLAS 13+

---ke --build . --config Release
```

**Run the Example:**

```bash
./build/example_simple              # Linux
./build/Release/example_simple.exe  # Windows
```

**Compare with the Script Wrapper:**

```bash
chmod +x compare_fp64.sh example_app.sh
./compare_fp64.sh ./example_app.sh
```

---

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


