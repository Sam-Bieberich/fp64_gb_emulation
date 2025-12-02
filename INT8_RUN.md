# Running your program with INT8 (quantized) GEMM

This document explains whether and how you can run *any* program using INT8 (quantized) matrix multiply kernels on NVIDIA GPUs, what changes are required, and practical recipes you can follow. It references the `c_comparison.cu` example in this repo which performs a simple FP64 vs INT8 comparison using `cublasGemmEx`.

Short answer

- You cannot generally force an arbitrary program to use INT8 just by setting an environment variable: INT8 is a different data representation and the program must (1) quantize inputs, (2) call INT8-capable GEMM kernels (cuBLAS/cuBLASLt/TensorRT), and (3) dequantize results (or work with int32 outputs).
- If your program already supports INT8 (via a framework or a built-in option), enable it there (see Framework section). Otherwise you must modify the code or add a wrapper that performs quantize→GEMM(INT8)→dequantize.

When INT8 is appropriate

- INT8 gives the biggest speedups for very large GEMMs or neural-network style workloads with lots of matrix multiplies.
- Good when model/application tolerates reduced numeric precision or you can use calibration/iterative refinement.
- Not suitable for algorithms requiring exact FP64 results (scientific computing) without additional numerical correction.

Overview of approaches

1. Framework-level (minimal to no code changes)
   - If you use TensorRT, PyTorch (quantization toolchain), or other frameworks that support INT8, enable quantization or export an INT8 engine. The framework handles quantize/scale management and uses INT8 tensor-core kernels.
   - Pros: easiest, highest-quality autotuning and runtime support.
   - Cons: works only if your workload fits the framework.

2. Library-level (modify code to call INT8 kernels) — recommended for C/C++ programs
   - Modify the code to:
     1. Compute quantization scales for A and B (per-tensor or per-channel).
     2. Quantize floating data to int8 (host or device).
     3. Call an INT8 GEMM kernel:
        - `cublasGemmEx` (simple; `CUDA_R_8I` inputs, `CUBLAS_COMPUTE_32I` accumulation)
        - `cuBLASLt` (`cublasLtMatmul`) — preferred for performance/tuning and advanced features.
     4. Dequantize the int32 accumulations to float (apply `scaleA * scaleB`).
   - Pros: full control, portable to any C/C++ program.
   - Cons: you must implement quantization and scale handling.

3. Wrapper or pre/post-processing script (no code change to program logic, but pipeline change)
   - If your program reads/writes matrices from disk or accepts external inputs/outputs, you can add a pre-process step to quantize inputs and change the program to read quantized inputs and call INT8 kernels — but this still requires program support.
   - You can also run the program twice (FP64 baseline and an INT8-accelerated variant) if you have access to its source or a hook.

Can I just flip an env var and run any binary in INT8?

- No. Unlike the `CUBLAS_MATH_MODE` option used for FP64 emulation (where cuBLAS dynamically picks ADP paths), there is no single environment variable that converts floating-point GEMMs to INT8 for arbitrary binaries.
- INT8 requires explicit data format and API calls. If a binary was built specifically to use INT8 (or a framework inside it supports toggling to INT8), it can be enabled with framework-level flags. Otherwise, modify the program.

Practical recipe: minimal changes for a C/C++ program using cuBLAS for GEMM

This is the direct route if you control the source and it currently calls `cublasDgemm` or `cublasGemmEx` with FP64/FP32.

1. Add quantize/dequantize helpers (host or device):
   - Per-tensor symmetric quantization (fastest):
     - scale = max_abs(x) / 127
     - q = round(clamp(x / scale, -127, 127)) -> int8
     - dequantize: x_hat = q * scale
   - Per-channel quantization for weights: compute `scale` per output channel (reduces error)

2. Convert input matrices A and B to int8 buffers (host or device). Keep a copy of FP reference if you need accuracy checks.

3. Replace the FP GEMM call with INT8 GEMM API:
   - Using `cublasGemmEx` (simple):

```cpp
// Example prototype (after quantizing A and B)
int32_t alpha = 1; // integer scale handled separately
int32_t beta  = 0;
// A: CUDA_R_8I, B: CUDA_R_8I, C: CUDA_R_32I (accum)
cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A8, CUDA_R_8I, lda,
            d_B8, CUDA_R_8I, ldb,
            &beta,
            d_C32, CUDA_R_32I, ldc,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT);
```

   - Using cuBLASLt (recommended for best perf and autotuning): create `cublasLtMatmulDesc_t`, matrix layouts with `CUDA_R_8I` and compute type `CUBLAS_COMPUTE_32I`, query heuristics, allocate workspace, then call `cublasLtMatmul`.

4. Dequantize C (int32 accumulator) to float/double:

```cpp
// after copying d_C32 back to host or performing device-side scaling
double scale_out = scaleA * scaleB; // if inputs were scaled by scaleA and scaleB
C_fp[i] = (double)C_int32[i] * scale_out;
```

5. (Optional) If you need FP64 accuracy, you can implement iterative refinement: compute residual in FP, solve correction with lower precision and add back — this is algorithmically non-trivial and problem-dependent.

6. Timing and metrics
   - Time only the GEMM kernel using CUDA events for kernel-only performance (exclude quant/dequant if you want kernel throughput).
   - When reporting end-to-end performance include quant and dequant times.
   - Compute GOPS = (2*m*n*k) / (time_seconds * 1e9) for reporting.
   - Compute relative L2 error or other accuracy metrics vs FP baseline.

Example: use the included `c_comparison.cu`

- `c_comparison.cu` in this repo demonstrates the exact flow above: it initializes FP64 matrices, runs `cublasDgemm`, creates a per-tensor symmetric quantization for A and B, runs `cublasGemmEx` with `CUDA_R_8I` inputs and `CUBLAS_COMPUTE_32I` accumulation, dequantizes and computes a relative L2 error. The `compare_int8.sh` script compiles and runs it and prints parsed results.

Framework-specific options (no code change or small code change)

- **TensorRT**: convert your model to a TensorRT INT8 engine (calibrate with representative dataset). Best option for NN inference.
- **PyTorch**: use PyTorch quantization flow or `torch.compile`/fx + quantization-aware training / post-training static quantization. Some kernels will use optimized INT8 paths.
- **TensorFlow**: use TFLite/TF-TRT for INT8 conversions.

Python + CuPy

- CuPy does not automatically perform INT8 tensor-core GEMM the way cuBLASLt does for C++. You can quantize in Python and call `cupy.matmul` with int32/float buffers, but this will not use specialized INT8 tensor-core kernels in many cases.
- For reliable INT8 tensor-core usage from Python, use frameworks (PyTorch/TensorRT) or write a small C++ extension that calls cuBLASLt and expose it to Python.

Summary checklist to convert a program to INT8

- [ ] Decide granularity: per-tensor vs per-channel quantization
- [ ] Implement quantize on host/device
- [ ] Replace GEMM calls with INT8-capable calls (`cublasGemmEx` or `cublasLtMatmul`)
- [ ] Implement dequantize (and optional iterative refinement)
- [ ] Add timing for kernel-only and end-to-end
- [ ] Validate accuracy (relative L2, max-abs) vs FP baseline
- [ ] Benchmark for multiple sizes and adjust workspace/algorithms

If you want, I can:

- Generate a ready-to-build CMake target that builds `c_comparison.cu` and a small INT8 helper library using cuBLASLt and add a `compare_int8.sh` that prints accuracy automatically.
- Add per-channel quantization to `c_comparison.cu` (improves accuracy quickly).
- Provide a Python wrapper that calls a compiled cuBLASLt INT8 routine from Python (if you want to test from Python code).

Which of these do you want next? If you prefer, I can create a new `INT8_RUN.md` file in the repo now that captures the steps above (I can also add an example patch to `c_comparison.cu` for per-channel quantization).