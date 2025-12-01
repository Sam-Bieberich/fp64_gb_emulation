# Python Environment Setup for FP64 Emulation

## Prerequisites

1. **CUDA Toolkit 13.0+** must be installed
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Verify: `nvcc --version`

2. **Python 3.8+**
   - Verify: `python --version`

## Installation

### Option 1: CuPy (Recommended)

CuPy is a NumPy-compatible library for GPU arrays:

```bash
# For CUDA 13.x
pip install cupy-cuda13x

# Verify installation
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}')"
python -c "import cupy as cp; print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')"
```

### Option 2: PyTorch

```bash
# For CUDA 13.x (check pytorch.org for latest)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Option 3: TensorFlow

```bash
pip install tensorflow[and-cuda]

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Running the Example

### Quick Test

```bash
# Native FP64
CUBLAS_MATH_MODE=CUBLAS_PEDANTIC_MATH python example_simple.py

# Emulated FP64
CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH python example_simple.py
```

### Full Comparison

```bash
chmod +x run_python.sh compare_fp64.sh
./compare_fp64.sh ./run_python.sh
```

Expected output:
```
========================================
FP64 Performance Comparison
========================================
Script: ./run_python.sh

----------------------------------------
Phase 1: Native FP64 (Pedantic Math)
----------------------------------------
FP64 Emulation Demo (Python/CuPy)
...
Performance: 1250.5 GFLOP/s

Native FP64 runtime: 5.234s

----------------------------------------
Phase 2: Emulated FP64 (ADP)
----------------------------------------
FP64 Emulation Demo (Python/CuPy)
...
Performance: 8750.2 GFLOP/s

Emulated FP64 runtime: 0.748s

========================================
Summary
========================================
Native FP64:   5.234s
Emulated FP64: 0.748s
Speedup:       7.00x
```

## Troubleshooting

### CuPy not finding CUDA

```bash
export CUDA_PATH=/usr/local/cuda-13.0  # Adjust path
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Wrong CUDA version

```bash
# Uninstall old version
pip uninstall cupy

# Install for your CUDA version
pip install cupy-cuda13x  # for CUDA 13.x
```

### Memory errors

Reduce matrix size in `example_simple.py`:
```python
N = 4096  # Instead of 8192
```

## Matrix Size Guide

| N | Memory per Matrix | Total (A+B+C) | Recommended GPU Memory |
|---|-------------------|---------------|------------------------|
| 2048 | 32 MB | 96 MB | 4 GB+ |
| 4096 | 128 MB | 384 MB | 4 GB+ |
| 8192 | 512 MB | 1.5 GB | 8 GB+ |
| 16384 | 2 GB | 6 GB | 16 GB+ |

Blackwell GB100/GB200 have 80-192 GB HBM, so large sizes are fine.

## Using with Other Python Libraries

### PyTorch Example

```python
import torch
import time

N = 8192
A = torch.rand(N, N, dtype=torch.float64, device='cuda')
B = torch.rand(N, N, dtype=torch.float64, device='cuda')

# Warm-up
for _ in range(2):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(10):
    C = torch.matmul(A, B)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 10

gflops = (2 * N**3) / (elapsed * 1e9)
print(f"Performance: {gflops:.2f} GFLOP/s")
```

Run with:
```bash
CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH python pytorch_example.py
```

### TensorFlow Example

```python
import tensorflow as tf
import time

N = 8192
A = tf.random.uniform([N, N], dtype=tf.float64)
B = tf.random.uniform([N, N], dtype=tf.float64)

# Move to GPU
with tf.device('/GPU:0'):
    A_gpu = tf.constant(A)
    B_gpu = tf.constant(B)
    
    # Warm-up
    for _ in range(2):
        C = tf.matmul(A_gpu, B_gpu)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        C = tf.matmul(A_gpu, B_gpu)
    elapsed = (time.perf_counter() - start) / 10

gflops = (2 * N**3) / (elapsed * 1e9)
print(f"Performance: {gflops:.2f} GFLOP/s")
```

Run with:
```bash
CUBLAS_MATH_MODE=CUBLAS_DEFAULT_MATH python tensorflow_example.py
```
