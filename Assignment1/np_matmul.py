import numpy as np
from time import monotonic

N = 2048*2

A = np.random.random((N, N)).astype(np.float32)
B = np.random.random((N, N)).astype(np.float32)

for i in range(10):
    start = monotonic()
    C = A @ B
    end = monotonic()
    s = end - start
    print(f"GFLOPS: {((N*N*2*N) / (s * 10**9)):.6f}")
