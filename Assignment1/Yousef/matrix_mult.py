import random
from tqdm import tqdm
from time import time
import numpy as np
# Numby with singe thread
import threadpoolctl as tpctl
tpctl.threadpool_limits(1)

# matrix multiplication
Gflop = []

n = 1024

# using numpy
def nympy_matrix_mult(A, B):
     return A @ B

A = np.random.rand(n, n)
B = np.random.rand(n, n)
for i in range(10):
     start = time()
     C = nympy_matrix_mult(A, B)
     end = time()
     flops = 2 * n**3
     gflops = flops / (end-start) / 10**9
     Gflop.append(gflops)

n = 0
for(gflops) in Gflop:
     n += gflops
     avgGflops = n / len(Gflop)

print("Gflops numpy: %0.6f" % avgGflops)
