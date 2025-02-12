import random
from time import monotonic

# N = 2048
N = 256

gflops = []
times = []

for i in range(10):
    A = [[random.random()
          for row in range(N)]
         for col in range(N)]

    B = [[random.random()
          for row in range(N)]
         for col in range(N)]

    C = [[0 for row in range(N)]
         for col in range(N)]

    print("calculating ... \n")

    start = monotonic()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]

    end = monotonic()
    s = end - start
    gflops.append((N*N*2*N) / (s * 10**9))
    times.append(s)

    print(f"{sum(gflops)/len(gflops)} GFLOPS")
    print(f"Time: {sum(times)/len(times)}")


print(f"{sum(gflops)/len(gflops)} GFLOPS")
print(f"Time: {sum(times)/len(times)}")
