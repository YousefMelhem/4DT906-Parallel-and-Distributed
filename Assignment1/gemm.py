#!/usr/bin/env python3
import numpy as np


N = 2048

if __name__ == "__main__":
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    Cvals = A @ B

    with open("/tmp/matmul", "wb") as f:
        f.write(A.data)
        f.write(B.data)
        f.write(Cvals.data)
