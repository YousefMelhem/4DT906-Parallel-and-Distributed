import numpy as np
from time import monotonic
import matplotlib.pyplot as plt


cpp_flops = [171.834152, 395.073334, 393.556854, 266.474579, 222.968567]
np_flops = []
n = [512, 1024, 2048, 4096, 8192]


for i in n:
    A = np.random.random((i, i)).astype(np.float32)
    B = np.random.random((i, i)).astype(np.float32)
    start = monotonic()
    C = A @ B
    end = monotonic()
    s = end - start

    gflops = (i*i*2*i) / (s * 10**9)
    np_flops.append(gflops)


plt.plot(n, cpp_flops, label="C++")

plt.plot(n, np_flops, label="Numpy")

plt.xlabel("N")

plt.ylabel("GFLOPS")

plt.legend()

plt.show()
