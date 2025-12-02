import torch
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--size", type=int, default=8192)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--dtype", choices=["float32", "float64", "int8"], default="float32")
args = parser.parse_args()

# Set thread count for CPU
os.environ["OMP_NUM_THREADS"] = str(args.threads)
torch.set_num_threads(args.threads)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Running on {device} with {args.threads} threads, dtype={args.dtype}")

# Create matrices based on dtype
if args.dtype == "int8":
    if device.type == "cuda":
        print("Warning: PyTorch INT8 dynamic quantization is CPU-only.")
        print("For GPU INT8, use TensorRT, custom kernels, or torch.compile with quantization.")
        print("Falling back to FP32 on CUDA.")
        A = torch.randn(args.size, args.size, device=device)
        B = torch.randn(args.size, args.size, device=device)
        use_int8 = False
    else:
        # Create FP32 tensors first
        A_fp = torch.randn(args.size, args.size, device=device)
        B_fp = torch.randn(args.size, args.size, device=device)
        
        # Compute scales for symmetric quantization
        scale_A = A_fp.abs().max().item() / 127.0
        scale_B = B_fp.abs().max().item() / 127.0
        
        # Quantize to INT8
        A = torch.quantize_per_tensor(A_fp, scale=scale_A, zero_point=0, dtype=torch.qint8)
        B = torch.quantize_per_tensor(B_fp, scale=scale_B, zero_point=0, dtype=torch.qint8)
        
        print(f"Quantized A: {A.dtype}, scale={scale_A:.6f}")
        print(f"Quantized B: {B.dtype}, scale={scale_B:.6f}")
        use_int8 = True
else:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    A = torch.randn(args.size, args.size, device=device, dtype=dtype)
    B = torch.randn(args.size, args.size, device=device, dtype=dtype)
    use_int8 = False

# Warm-up
for _ in range(2):
    if use_int8:
        C = A.dequantize() @ B.dequantize()
    else:
        C = A @ B
if device.type == "cuda":
    torch.cuda.synchronize()

# Timed run
start = time.time()
for _ in range(args.iters):
    if use_int8:
        C = A.dequantize() @ B.dequantize()
    else:
        C = A @ B
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.time()

elapsed = end - start
gflops = (2 * args.size**3 * args.iters) / elapsed / 1e9
print(f"Elapsed time: {elapsed:.3f}s")
print(f"Approx throughput: {gflops:.2f} GFLOP/s")
