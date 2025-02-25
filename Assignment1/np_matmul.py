import numpy as np
from time import monotonic

N = 2048*2


A = np.random.random((N, N))
B = np.random.random((N, N))


itr = 10

avg = 0
for i in range(itr):
    start = monotonic()
    C = A @ B
    end = monotonic()
    s = end - start

    gflops = (N*N*2*N) / (s * 10**9)
    avg += gflops

    print(f"GFLOPS: {gflops:.6f}")

print(f"avg: {avg/itr:.2f}")
