from mpi4py import MPI
import random
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix configuration (adjust as needed)
n = 512  # Matrix size (must be divisible by number of processes)
num_runs = 1  # Number of runs for averaging
verify = False  # Enable verification for small matrices

if rank == 0:
    # Generate random matrices A and B using list comprehensions
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    # Prepare row chunks for distribution
    chunk_size = n // size
    chunks = [A[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]

    # Start total timer (includes communication)
    total_start = time.time()
else:
    A, B, chunks = None, None, None

# Distribute work and broadcast matrix B
local_A = comm.scatter(chunks, root=0)
B = comm.bcast(B, root=0)

# Start computation timer
comp_start = time.time()

# Local matrix multiplication
local_C = []
for a_row in local_A:
    c_row = [0.0] * n
    for j in range(n):
        sum_val = 0.0
        for k in range(n):
            sum_val += a_row[k] * B[k][j]
        c_row[j] = sum_val
    local_C.append(c_row)

# End computation timer
comp_time = time.time() - comp_start

# Gather results and computation times
C_chunks = comm.gather(local_C, root=0)
comp_times = comm.gather(comp_time, root=0)

if rank == 0:
    # Calculate performance metrics
    total_time = time.time() - total_start
    max_comp_time = max(comp_times)
    total_ops = 2 * n**3  # 1 multiply + 1 add per inner iteration

    # Calculate GFLOPS rates
    total_gflops = (total_ops / 1e9) / total_time
    comp_gflops = (total_ops / 1e9) / max_comp_time

    # Combine final matrix
    C = []
    for chunk in C_chunks:
        C.extend(chunk)

    # Print performance information
    print(f"\nMatrix Multiplication Performance (n={n})")
    print("=========================================")
    print(f"MPI Processes: {size}")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Max computation time: {max_comp_time:.4f} sec")
    print(f"Communication overhead: {total_time - max_comp_time:.4f} sec")
    print(f"Total GFLOP/s: {total_gflops:.2f}")
    print(f"Computation GFLOP/s: {comp_gflops:.2f}")

    # Verification (for small matrices)
    if verify and n <= 10:
        print("\nVerification:")
        seq_C = [[sum(a * b for a, b in zip(A_row, B_col))
                  for B_col in zip(*B)] for A_row in A]
        diff = sum(abs(C[i][j] - seq_C[i][j])
                   for i in range(n) for j in range(n))
        print(f"Total difference: {diff:.6f}")

    # Memory usage estimation
    matrix_mb = 2 * 8 * n**2 / 1e6  # 8 bytes per float, 2 matrices
    print(f"\nApproximate memory usage (A+B): {matrix_mb:.2f} MB")
