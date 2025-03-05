from mpi4py import MPI
import random
import socket
import time


def calculate_pi(num_samples):
    inside_count = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x*x + y*y <= 1.0:
            inside_count += 1

    return inside_count


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

# Print basic information
print(f"Hello from rank {rank} out of {size} on host {hostname}")

num_samples_per_process = 1000000

start_time = time.time()

local_count = calculate_pi(num_samples_per_process)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Calculate Pi on the root process
if rank == 0:
    pi_estimate = 4.0 * total_count / (num_samples_per_process * size)
    end_time = time.time()
    print(f"Pi estimate: {pi_estimate}")
    print(f"True Pi value: {3.141592653589793}")
    print(f"Error: {abs(pi_estimate - 3.141592653589793)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(
        f"Using {size} processes with {num_samples_per_process} samples per process")
