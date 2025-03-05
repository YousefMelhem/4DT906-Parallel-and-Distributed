import random
from time import monotonic

# N = 2048
N = 256

A = [[random.random()
      for row in range(N)]
     for col in range(N)]

B = [[random.random()
      for row in range(N)]
     for col in range(N)]

C = [[0 for row in range(N)]
     for col in range(N)]


start = monotonic()
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]

end = monotonic()
s = end - start

print(f"GFLOPS: {((N*N*2*N) / (s * 10**9)):.6f}")
print("|")
print(f"t: {s:.6f}")
print("|")
print(f"N: {N}")
print("|")
