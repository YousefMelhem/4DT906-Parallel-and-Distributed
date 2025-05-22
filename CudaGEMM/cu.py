import cupy as cp
import time

# Matrix size (must be power of two)
N = 1024*2

# Create random matrices
A = cp.random.rand(N, N).astype(cp.float32)
B = cp.random.rand(N, N).astype(cp.float32)

# Warm-up run (CUDA JIT compilation overhead)
A @ B

# Measure execution time for cuBLAS-based matrix multiplication
start = time.time()
C = A @ B 
cp.cuda.Device(0).synchronize()  # Ensure all operations are finished
end = time.time()

# Calculate GFLOPS
elapsed_time = end - start  # In seconds
flops = 2.0 * N**3  # FLOPs = 2 * N^3
gflops = (flops / (elapsed_time * 1e9))  # Convert to GFLOPS

print(f"CuPy/cuBLAS GEMM Performance: {gflops:.2f} GFLOPS")



