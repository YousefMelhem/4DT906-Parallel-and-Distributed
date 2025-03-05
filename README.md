## THIS PROJECT IS STILL ONGOING
currently working on Optimizing Cuda implemntation to beat CuPy perfomance on 3080

# **Parallel and Distributed Computing**  

High-performance implementations of matrix multiplication using different parallelization techniques, including OpenMP, MPI, and CUDA. The project focuses on optimizing computational efficiency, memory usage, and scalability across single-threaded, multi-threaded, distributed, and GPU-accelerated environments.  

## **Features**  

- **Optimized Matrix Multiplication:**  
  - Baseline implementation using a standard three-loop algorithm.  
  - Optimized cache-aware single-threaded version.  
  - SIMD vectorization using **AVX (x86) and NEON (ARM) instructions** for efficient CPU execution.  
  - Parallelized versions leveraging OpenMP and multi-threading.  
  - Distributed computation using MPI.  
  - GPU acceleration with CUDA.  

- **Performance Benchmarks:**  
  - Detailed comparison of execution times and optimizations.  
  - Analysis of speedup across different architectures.  
  - Scaling behavior from single-core to multi-core and distributed execution.  

## **Technologies Used**  

- **C++** – Optimized CPU implementations  
- **Clang** – Recommended compiler (LLVM 18+)  
- **AVX & NEON** – SIMD optimizations for x86 and ARM architectures  
- **OpenMP** – Multithreading support  
- **MPI (MPICH/OpenMPI)** – Distributed computing across multiple nodes  
- **CUDA** – GPU-accelerated computation with NVIDIA GPUs  

## **Getting Started**  

### **Prerequisites**  

Ensure you have the following installed:  

- **Clang Compiler (LLVM 18+)**  
- **OpenMP** (for CPU parallelization)  
- **MPI (MPICH or OpenMPI)**  
- **CUDA Toolkit** (for GPU execution)  

### **Installation & Compilation**  

#### **Single-Threaded & OpenMP Version**  

```sh
clang++ -O3 -march=native -fopenmp gemm.cpp -o compiled/gemm
./gemm
````

MPI Version
```
mpic++ -O3 -march=native mpi_matrix_mul.cpp -o mpi_matmul
mpirun -np 4 ./mpi_matmul
```

CUDA Version

```
nvcc  gemm.cu -o compiled/gemm 
./cuda_matmul
```
