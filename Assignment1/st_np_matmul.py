import numpy as np
from time import monotonic
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    # N = 4096
    N = 1024*2

    A = np.random.random((N, N)).astype(np.float32)
    B = np.random.random((N, N)).astype(np.float32)

    avg = 0
    itr = 10
    for i in range(itr):
        start = monotonic()
        C = A @ B
        end = monotonic()
        s = end - start
        flops = N*N*2*N / (s * 10**9)
        print(f"GFLOPS: {flops:.2f}")
        avg += flops

    print(f"Average GFLOPS: {avg/itr:.2f}")
