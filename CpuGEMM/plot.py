import numpy as np
from time import monotonic
import matplotlib.pyplot as plt


def calcnpflops(n):
    A = np.random.random((n, n)).astype(np.float32)
    B = np.random.random((n, n)).astype(np.float32)
    start = monotonic()
    C = A @ B
    end = monotonic()
    s = end - start

    gflops = (n*n*2*n) / (s * 10**9)
    return gflops


cpp_flops = [171.834152, 395.073334, 393.556854, 266.474579, 222.968567]
accelerate = [263.948334, 887.389954,
              1248.083496, 1293.433472, 1398.854248]

N = [512, 1024, 2048, 4096, 8192]

np_flops = [calcnpflops(i) for i in N]

plt.plot(N, cpp_flops, label="C++")
plt.plot(N, np_flops, label="Numpy")
plt.plot(N, accelerate, label="Accelerate")

plt.xlabel("N")

plt.ylabel("GFLOPS")

plt.legend()

plt.show()
