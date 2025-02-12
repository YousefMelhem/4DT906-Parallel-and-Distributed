import numpy as np
from time import monotonic

N = 2048
# N = 256
gflops = []
times = []

for i in range(10):
    A = np.random.random((N, N))
    B = np.random.random((N, N))

    print("calculating ... \n")
    start = monotonic()

    C = A @ B

    end = monotonic()
    s = end - start

    print(f"{(N*N*2*N) / (s * 10**9)} GFLOPS")
    print(f"Time: {s}")

    gflops.append((N*N*2*N) / (s * 10**9))
    times.append(s)

print(f"{sum(gflops)/len(gflops):.4f} GFLOPS")
print(f"Time: {sum(times)/len(times):.4f}")
