import numpy as np
from time import monotonic
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    N = 4096
    # N = 256
    gflops = []
    times = []

    A = np.random.random((N, N))
    B = np.random.random((N, N))

    print("calculating ... \n")
    start = monotonic()

    C = A @ B

    end = monotonic()
    s = end - start

    gflops.append((N*N*2*N) / (s * 10**9))
    times.append(s)

    print(f"{sum(gflops)/len(gflops):.4f} GFLOPS")
    print(f"Time: {sum(times)/len(times):.4f}")
