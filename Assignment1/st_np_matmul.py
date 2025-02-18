import numpy as np
from time import monotonic
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    # N = 4096
    N = 1024

    A = np.random.random((N, N))
    B = np.random.random((N, N))

    start = monotonic()

    C = A @ B

    end = monotonic()
    s = end - start

    print(f"GFLPOS: {((N*N*2*N) / (s * 10**9)):.6f}")
    print("|")
    print(f"t: {s:.6f}")
    print("|")
    print(f"N: {N}")
