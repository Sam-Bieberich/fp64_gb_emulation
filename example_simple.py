import cupy as cp
import time

print("=" * 50)
print("FP64 Emulation Demo (Python/CuPy)")
print("=" * 50)

# Get GPU info
device = cp.cuda.Device()
props = cp.cuda.runtime.getDeviceProperties(device.id)
print(f"Device: {props['name'].decode()}")
print(f"Compute Capability: {props['major']}.{props['minor']}")
print()

# Matrix size
N = 8192
print(f"Matrix size: {N}x{N}")
print(f"Memory per matrix: {N*N*8/1e9:.2f} GB")
print()

# Allocate matrices
print("Allocating matrices...")
A = cp.random.rand(N, N, dtype=cp.float64)
B = cp.random.rand(N, N, dtype=cp.float64)
print("Done.")
print()

# Warm-up
print("Warming up...")
for _ in range(2):
    C = cp.matmul(A, B)
cp.cuda.Device().synchronize()
print("Done.")
print()

# Benchmark
print("Running benchmark (10 iterations)...")
start = time.perf_counter()
for _ in range(10):
    C = cp.matmul(A, B)
cp.cuda.Device().synchronize()
elapsed = (time.perf_counter() - start) / 10

gflops = (2 * N**3) / (elapsed * 1e9)
print(f"Time per iteration: {elapsed*1000:.2f} ms")
print(f"Performance: {gflops:.2f} GFLOP/s")
print("=" * 50)
