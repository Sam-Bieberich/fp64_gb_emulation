import torch
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--size", type=int, default=8192)
parser.add_argument("--iters", type=int, default=50)
args = parser.parse_args()

# Set thread count for CPU
os.environ["OMP_NUM_THREADS"] = str(args.threads)
torch.set_num_threads(args.threads)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Running on {device} with {args.threads} threads")

# Create matrices
A = torch.randn(args.size, args.size, device=device)
B = torch.randn(args.size, args.size, device=device)

# Warm-up
for _ in range(2):
    _ = A @ B
if device.type == "cuda":
    torch.cuda.synchronize()

# Timed run
start = time.time()
for _ in range(args.iters):
    _ = A @ B
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.time()

elapsed = end - start
gflops = (2 * args.size**3 * args.iters) / elapsed / 1e9
print(f"Elapsed time: {elapsed:.3f}s")
print(f"Approx throughput: {gflops:.2f} GFLOP/s")