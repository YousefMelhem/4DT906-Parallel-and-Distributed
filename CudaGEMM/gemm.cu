// nvcc gemm.cu -o compiled/gemm -O3 && ./compiled/gemm

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 8192 // Matrix size (N x N, must be power of 2)
#define BLOCKSIZE                                                              \
  16          // Size of the thread block in each dimension (for basic kernels)
#define BM 64 // Size of the block tile in M dimension (for blocktiling)
#define BN 64 // Size of the block tile in N dimension (for blocktiling)
#define BK 8  // Size of the block tile in K dimension (for blocktiling)
#define TM                                                                     \
  8 // Number of output elements per thread in M dimension (for blocktiling)
#define CEIL_DIV(m, n) ((m) + (n) - 1) / (n)

// Utility function to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                                 \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Original kernel (non-optimized)
__global__ void gemm_original(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0;
    for (int k = 0; k < n; k++) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

// Kernel with global memory coalescing
__global__ void gemm_coalesced(float *A, float *B, float *C, int n) {
  // Change thread mapping to enable coalescing
  // Each thread processes one element of the result matrix
  // threadIdx.x is now used as a linear index within the block
  const int thread_id = threadIdx.x;

  // Calculate row and column indices for this thread
  // This mapping ensures threads in the same warp access consecutive memory
  const int row = blockIdx.y * BLOCKSIZE + (thread_id / BLOCKSIZE);
  const int col = blockIdx.x * BLOCKSIZE + (thread_id % BLOCKSIZE);

  if (row < n && col < n) {
    float sum = 0.0f;

    for (int k = 0; k < n; k++) {
      sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
  }
}

// Kernel with transposed B matrix for coalesced access
__global__ void gemm_transposed(float *A, float *B_transposed, float *C,
                                int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
#pragma unroll 16
    for (int k = 0; k < n; k++) {
      // A is accessed in a row-major fashion (coalesced)
      // B is transposed in memory layout for coalesced access
      sum += A[row * n + k] * B_transposed[col * n + k];
    }
    C[row * n + col] = sum;
  }
}

// Shared memory cache-blocked kernel
__global__ void gemm_shared(float *A, float *B, float *C, int n) {
  // Shared memory for the tiles
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // Thread indices within the block
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;

  // Block indices
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Global indices for the C matrix
  int row = blockRow * BLOCKSIZE + threadRow;
  int col = blockCol * BLOCKSIZE + threadCol;

  float sum = 0.0f;

  // Loop over all tiles required to compute the result
  for (int tileIdx = 0; tileIdx < n / BLOCKSIZE; tileIdx++) {
    // Each thread loads one element of A and B into shared memory
    As[threadRow][threadCol] = A[row * n + (tileIdx * BLOCKSIZE + threadCol)];
    Bs[threadRow][threadCol] = B[(tileIdx * BLOCKSIZE + threadRow) * n + col];

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

// Compute dot product for this tile
#pragma unroll 16
    for (int k = 0; k < BLOCKSIZE; k++) {
      sum += As[threadRow][k] * Bs[k][threadCol];
    }

    // Ensure computation is complete before loading next tile
    __syncthreads();
  }

  // Write the result to global memory
  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

// Shared memory with register blocking
// Fixed Shared Memory with Register Blocking kernel
__global__ void gemm_shared_register(float *A, float *B, float *C, int n) {
  // Shared memory for the tiles
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // Thread indices within the block
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;

  // Block indices
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Each thread computes a 2x2 submatrix of C
  int row1 = blockRow * BLOCKSIZE + threadRow;
  int col1 = blockCol * BLOCKSIZE + threadCol;
  int row2 = row1 + BLOCKSIZE / 2;
  int col2 = col1 + BLOCKSIZE / 2;

  // Bounds checking
  bool valid1 = row1 < n && col1 < n;
  bool valid2 = row2 < n && col2 < n;
  bool valid3 = row1 < n && col2 < n;
  bool valid4 = row2 < n && col1 < n;

  // Accumulate results in registers
  float sum11 = 0.0f;
  float sum12 = 0.0f;
  float sum21 = 0.0f;
  float sum22 = 0.0f;

  // Loop over all tiles required to compute the result
  for (int tileIdx = 0; tileIdx < (n + BLOCKSIZE - 1) / BLOCKSIZE; tileIdx++) {
    // Load A and B tiles into shared memory
    int globalIdxA = row1 * n + tileIdx * BLOCKSIZE + threadCol;
    int globalIdxB = (tileIdx * BLOCKSIZE + threadRow) * n + col1;

    // Add boundary checking for memory loads
    if (row1 < n && tileIdx * BLOCKSIZE + threadCol < n) {
      As[threadRow][threadCol] = A[globalIdxA];
    } else {
      As[threadRow][threadCol] = 0.0f;
    }

    if (tileIdx * BLOCKSIZE + threadRow < n && col1 < n) {
      Bs[threadRow][threadCol] = B[globalIdxB];
    } else {
      Bs[threadRow][threadCol] = 0.0f;
    }

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

// Compute dot products for this tile
#pragma unroll 8
    for (int k = 0; k < BLOCKSIZE; k++) {
      if (tileIdx * BLOCKSIZE + k >= n)
        break; // Boundary check

      // Load values into registers for the 4 combinations
      float aval1 = As[threadRow][k];
      float bval1 = Bs[k][threadCol];

      // Update accumulation
      sum11 += aval1 * bval1;

      // Only compute the other elements if they're valid
      if (valid3 && threadCol + BLOCKSIZE / 2 < BLOCKSIZE) {
        float bval2 = Bs[k][threadCol + BLOCKSIZE / 2];
        sum12 += aval1 * bval2;
      }

      if (valid4 && threadRow + BLOCKSIZE / 2 < BLOCKSIZE) {
        float aval2 = As[threadRow + BLOCKSIZE / 2][k];
        sum21 += aval2 * bval1;
      }

      if (valid2 && threadRow + BLOCKSIZE / 2 < BLOCKSIZE &&
          threadCol + BLOCKSIZE / 2 < BLOCKSIZE) {
        float aval2 = As[threadRow + BLOCKSIZE / 2][k];
        float bval2 = Bs[k][threadCol + BLOCKSIZE / 2];
        sum22 += aval2 * bval2;
      }
    }

    // Ensure computation is complete before loading next tile
    __syncthreads();
  }

  // Write results to global memory
  if (valid1)
    C[row1 * n + col1] = sum11;
  if (valid3)
    C[row1 * n + col2] = sum12;
  if (valid4)
    C[row2 * n + col1] = sum21;
  if (valid2)
    C[row2 * n + col2] = sum22;
}

// 1D Blocktiling kernel - each thread computes multiple results
__global__ void gemm_blocktiling(float *A, float *B, float *C, int n) {
  // Shared memory for the tiles
  __shared__ float As[BM][BK]; // BM x BK tile of A
  __shared__ float Bs[BK][BN]; // BK x BN tile of B

  // Block indices
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread indices
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;

  // Thread count per block
  int threadsPerBlockX = blockDim.x;
  int threadsPerBlockY = blockDim.y;

  // Calculate the starting positions
  int rowBegin = blockRow * BM;
  int colBegin = blockCol * BN;

  // Thread's inner position for loading data
  int innerRowA = threadRow;
  int innerColA = threadCol;
  int innerRowB = threadRow;
  int innerColB = threadCol;

  // Each thread computes TM results in a column
  int resultRow = threadRow * TM;
  int resultCol = threadCol;

  // Register array to accumulate results
  float threadResults[TM] = {0.0f};

  // Loop over all tiles required to compute the result
  for (int tileIdx = 0; tileIdx < n; tileIdx += BK) {
    // Load data into shared memory
    // Each thread loads multiple elements for As and one element for Bs
    for (int i = 0; i < BM; i += threadsPerBlockY) {
      for (int j = 0; j < BK; j += threadsPerBlockX) {
        if (innerRowA + i < BM && innerColA + j < BK) {
          int globalRow = rowBegin + innerRowA + i;
          int globalCol = tileIdx + innerColA + j;
          if (globalRow < n && globalCol < n) {
            As[innerRowA + i][innerColA + j] = A[globalRow * n + globalCol];
          } else {
            As[innerRowA + i][innerColA + j] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < BK; i += threadsPerBlockY) {
      for (int j = 0; j < BN; j += threadsPerBlockX) {
        if (innerRowB + i < BK && innerColB + j < BN) {
          int globalRow = tileIdx + innerRowB + i;
          int globalCol = colBegin + innerColB + j;
          if (globalRow < n && globalCol < n) {
            Bs[innerRowB + i][innerColB + j] = B[globalRow * n + globalCol];
          } else {
            Bs[innerRowB + i][innerColB + j] = 0.0f;
          }
        }
      }
    }

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

    // Compute dot products for this tile
    // Each thread computes TM results
    for (int k = 0; k < BK; k++) {
      // Cache the B element to reduce shared memory accesses
      float Btmp = Bs[k][resultCol];

// Compute multiple dot products
#pragma unroll
      for (int m = 0; m < TM; m++) {
        if (resultRow + m < BM) {
          threadResults[m] += As[resultRow + m][k] * Btmp;
        }
      }
    }

    // Ensure computation is complete before loading next tile
    __syncthreads();
  }

  // Write results to global memory
  for (int m = 0; m < TM; m++) {
    if (resultRow + m < BM && resultCol < BN) {
      int globalRow = rowBegin + resultRow + m;
      int globalCol = colBegin + resultCol;
      if (globalRow < n && globalCol < n) {
        C[globalRow * n + globalCol] = threadResults[m];
      }
    }
  }
}

// 2D Blocktiling kernel - each thread computes multiple results in both
// dimensions
__global__ void gemm_blocktiling_2d(float *A, float *B, float *C, int n) {
  // Shared memory for the tiles
  __shared__ float As[BM][BK]; // BM x BK tile of A
  __shared__ float Bs[BK][BN]; // BK x BN tile of B

  // Block indices
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread indices
  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;

  // Thread count per block
  int threadsPerBlockX = blockDim.x;
  int threadsPerBlockY = blockDim.y;

  // Calculate the starting positions
  int rowBegin = blockRow * BM;
  int colBegin = blockCol * BN;

  // Thread's inner position for loading data
  int innerRowA = threadRow;
  int innerColA = threadCol;
  int innerRowB = threadRow;
  int innerColB = threadCol;

  // Number of results each thread computes in each dimension
  const int TN = 4; // 4 columns of output per thread

  // Each thread computes a TM x TN block of results
  int resultRowStart = threadRow * TM;
  int resultColStart = threadCol * TN;

  // Register array to accumulate results (TM rows x TN columns)
  float threadResults[TM][TN];

  // Initialize all results to zero
  for (int m = 0; m < TM; m++) {
    for (int n = 0; n < TN; n++) {
      threadResults[m][n] = 0.0f;
    }
  }

  // Loop over all tiles required to compute the result
  for (int tileIdx = 0; tileIdx < n; tileIdx += BK) {
    // Load data into shared memory
    // Each thread loads multiple elements
    for (int i = 0; i < BM; i += threadsPerBlockY) {
      for (int j = 0; j < BK; j += threadsPerBlockX) {
        if (innerRowA + i < BM && innerColA + j < BK) {
          int globalRow = rowBegin + innerRowA + i;
          int globalCol = tileIdx + innerColA + j;
          if (globalRow < n && globalCol < n) {
            As[innerRowA + i][innerColA + j] = A[globalRow * n + globalCol];
          } else {
            As[innerRowA + i][innerColA + j] = 0.0f;
          }
        }
      }
    }

    for (int i = 0; i < BK; i += threadsPerBlockY) {
      for (int j = 0; j < BN; j += threadsPerBlockX) {
        if (innerRowB + i < BK && innerColB + j < BN) {
          int globalRow = tileIdx + innerRowB + i;
          int globalCol = colBegin + innerColB + j;
          if (globalRow < n && globalCol < n) {
            Bs[innerRowB + i][innerColB + j] = B[globalRow * n + globalCol];
          } else {
            Bs[innerRowB + i][innerColB + j] = 0.0f;
          }
        }
      }
    }

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

    // Compute dot products for this tile
    // Each thread computes TM x TN results
    for (int k = 0; k < BK; k++) {
// Compute multiple dot products in a 2D pattern
#pragma unroll
      for (int m = 0; m < TM; m++) {
        if (resultRowStart + m < BM) {
          float Aval = As[resultRowStart + m][k];

#pragma unroll
          for (int n = 0; n < TN; n++) {
            if (resultColStart + n < BN) {
              threadResults[m][n] += Aval * Bs[k][resultColStart + n];
            }
          }
        }
      }
    }

    // Ensure computation is complete before loading next tile
    __syncthreads();
  }

  // Write results to global memory
  for (int m = 0; m < TM; m++) {
    if (resultRowStart + m < BM) {
      for (int n = 0; n < TN; n++) {
        if (resultColStart + n < BN) {
          int globalRow = rowBegin + resultRowStart + m;
          int globalCol = colBegin + resultColStart + n;
          if (globalRow < n && globalCol < n) {
            C[globalRow * n + globalCol] = threadResults[m][n];
          }
        }
      }
    }
  }
}

// Function to transpose a matrix on CPU
void transpose_matrix(float *src, float *dst, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      dst[j * n + i] = src[i * n + j];
    }
  }
}

// Measure performance function
float measure_kernel(const char *kernel_name,
                     void (*kernel)(float *, float *, float *, int), float *d_A,
                     float *d_B, float *d_C, int n, int iterations) {
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  dim3 blockDim, gridDim;

  if (strcmp(kernel_name, "original") == 0) {
    blockDim = dim3(16, 16);
    gridDim = dim3(CEIL_DIV(n, blockDim.x), CEIL_DIV(n, blockDim.y));
  } else if (strcmp(kernel_name, "coalesced") == 0) {
    blockDim = dim3(BLOCKSIZE * BLOCKSIZE); // 1D block
    gridDim = dim3(CEIL_DIV(n, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  } else if (strcmp(kernel_name, "transposed") == 0) {
    blockDim = dim3(16, 16);
    gridDim = dim3(CEIL_DIV(n, blockDim.x), CEIL_DIV(n, blockDim.y));
  } else if (strcmp(kernel_name, "shared") == 0) {
    blockDim = dim3(BLOCKSIZE, BLOCKSIZE);
    gridDim = dim3(CEIL_DIV(n, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  } else if (strcmp(kernel_name, "shared_register") == 0) {
    blockDim = dim3(BLOCKSIZE / 2, BLOCKSIZE / 2);
    gridDim = dim3(CEIL_DIV(n, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  } else if (strcmp(kernel_name, "blocktiling") == 0) {
    blockDim =
        dim3(BN, BM / TM); // Each thread handles TM output elements in a column
    gridDim = dim3(CEIL_DIV(n, BN), CEIL_DIV(n, BM));
  } else if (strcmp(kernel_name, "blocktiling_2d") == 0) {
    blockDim =
        dim3(BN / 4, BM / TM); // Each thread handles a TM x 4 block of outputs
    gridDim = dim3(CEIL_DIV(n, BN), CEIL_DIV(n, BM));
  }

  printf("Kernel: %s\n", kernel_name);
  printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
  printf("Block dimensions: (%d, %d, %d)\n", blockDim.x, blockDim.y,
         blockDim.z);

  // Get occupancy
  int deviceId, maxBlocks;
  cudaGetDevice(&deviceId);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, kernel, blockDim.x * blockDim.y * blockDim.z, 0);
  printf("Occupancy: %d blocks per SM, %d SMs\n", maxBlocks,
         prop.multiProcessorCount);

  // Warm-up run
  kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  float total_ms = 0.0f;

  for (int i = 0; i < iterations; i++) {
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    total_ms += milliseconds;

    float iter_gflops = (2.0 * n * n * n) / (milliseconds * 1e6);
    printf("Iteration %d: %.2f ms (%.2f GFLOPS)\n", i + 1, milliseconds,
           iter_gflops);
  }

  float avg_ms = total_ms / iterations;
  float gflops = (2.0 * n * n * n) / (avg_ms * 1e6);

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  return gflops;
}

int main() {
  // Print device information
  int device_count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));

    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory per Block: %d KB\n",
           (int)(prop.sharedMemPerBlock / 1024));
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", (int)(prop.l2CacheSize / 1024));
    printf("\n");
  }

  const int iterations = 5;
  int size = N * N * sizeof(float);

  printf("Matrix size: %d x %d\n", N, N);
  printf("Memory required: %.2f GB\n", 3.0 * size / (1024.0 * 1024.0 * 1024.0));

  // Host memory allocation
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_B_transposed = (float *)malloc(size);
  float *h_C_original = (float *)malloc(size);
  float *h_C_coalesced = (float *)malloc(size);
  float *h_C_transposed = (float *)malloc(size);
  float *h_C_shared = (float *)malloc(size);
  float *h_C_shared_register = (float *)malloc(size);
  float *h_C_blocktiling = (float *)malloc(size);
  float *h_C_blocktiling_2d = (float *)malloc(size);

  if (!h_A || !h_B || !h_B_transposed || !h_C_original || !h_C_coalesced ||
      !h_C_transposed || !h_C_shared || !h_C_shared_register ||
      !h_C_blocktiling || !h_C_blocktiling_2d) {
    fprintf(stderr, "Failed to allocate host memory\n");
    exit(EXIT_FAILURE);
  }

  // Initialize matrices with realistic values
  srand(42);
  printf("Initializing matrices...\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_A[i * N + j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
      h_B[i * N + j] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
  }

  // Create transposed version of B for the transposed kernel
  printf("Transposing matrix B...\n");
  transpose_matrix(h_B, h_B_transposed, N);

  // Device memory allocation
  float *d_A, *d_B, *d_B_transposed, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B_transposed, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

  // Copy data to device
  printf("Copying data to device...\n");
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_B_transposed, h_B_transposed, size, cudaMemcpyHostToDevice));

  // Run original kernel
  printf("\n=== Running Original Kernel ===\n");
  float gflops_original =
      measure_kernel("original", gemm_original, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(cudaMemcpy(h_C_original, d_C, size, cudaMemcpyDeviceToHost));

  // Run coalesced kernel
  printf("\n=== Running Coalesced Kernel ===\n");
  float gflops_coalesced =
      measure_kernel("coalesced", gemm_coalesced, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(
      cudaMemcpy(h_C_coalesced, d_C, size, cudaMemcpyDeviceToHost));

  // Run transposed kernel
  printf("\n=== Running Transposed Kernel ===\n");
  float gflops_transposed = measure_kernel("transposed", gemm_transposed, d_A,
                                           d_B_transposed, d_C, N, iterations);
  CHECK_CUDA_ERROR(
      cudaMemcpy(h_C_transposed, d_C, size, cudaMemcpyDeviceToHost));

  // Run shared memory kernel
  printf("\n=== Running Shared Memory Kernel ===\n");
  float gflops_shared =
      measure_kernel("shared", gemm_shared, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost));

  // Run shared memory with register blocking kernel
  printf("\n=== Running Shared Memory with Register Blocking Kernel ===\n");
  float gflops_shared_register = measure_kernel(
      "shared_register", gemm_shared_register, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(
      cudaMemcpy(h_C_shared_register, d_C, size, cudaMemcpyDeviceToHost));

  // Run 1D blocktiling kernel
  printf("\n=== Running 1D Blocktiling Kernel ===\n");
  float gflops_blocktiling = measure_kernel("blocktiling", gemm_blocktiling,
                                            d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(
      cudaMemcpy(h_C_blocktiling, d_C, size, cudaMemcpyDeviceToHost));

  // Run 2D blocktiling kernel
  printf("\n=== Running 2D Blocktiling Kernel ===\n");
  float gflops_blocktiling_2d = measure_kernel(
      "blocktiling_2d", gemm_blocktiling_2d, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(
      cudaMemcpy(h_C_blocktiling_2d, d_C, size, cudaMemcpyDeviceToHost));

  // Verify results against original implementation
  printf("\n=== Verifying Results (vs Original) ===\n");

  printf("Coalesced vs Original:\n");
  int diff_count = 0;
  double max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_coalesced[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("Transposed vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_transposed[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("Shared Memory vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_shared[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("Shared Memory with Register Blocking vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff =
        fabs((double)h_C_original[i] - (double)h_C_shared_register[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("1D Blocktiling vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_blocktiling[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("2D Blocktiling vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_blocktiling_2d[i]);
    if (diff > 1e-3)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences > 1e-3: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  // Print performance summary
  printf("\n=== Performance Summary ===\n");
  printf("Original GEMM Performance:                      %.2f GFLOPS\n",
         gflops_original);
  printf("Coalesced GEMM Performance:                     %.2f GFLOPS\n",
         gflops_coalesced);
  printf("Transposed GEMM Performance:                    %.2f GFLOPS\n",
         gflops_transposed);
  printf("Shared Memory GEMM Performance:                 %.2f GFLOPS\n",
         gflops_shared);
  printf("Shared Memory with Register Blocking Performance: %.2f GFLOPS\n",
         gflops_shared_register);
  printf("1D Blocktiling GEMM Performance:                %.2f GFLOPS\n",
         gflops_blocktiling);
  printf("2D Blocktiling GEMM Performance:                %.2f GFLOPS\n",
         gflops_blocktiling_2d);

  printf("\n=== Speedup Relative to Original ===\n");
  printf("Coalesced:                     %.2fx\n",
         gflops_coalesced / gflops_original);
  printf("Transposed:                    %.2fx\n",
         gflops_transposed / gflops_original);
  printf("Shared Memory:                 %.2fx\n",
         gflops_shared / gflops_original);
  printf("Shared Memory with Register:   %.2fx\n",
         gflops_shared_register / gflops_original);
  printf("1D Blocktiling:                %.2fx\n",
         gflops_blocktiling / gflops_original);
  printf("2D Blocktiling:                %.2fx\n",
         gflops_blocktiling_2d / gflops_original);

  // Free memory
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_B_transposed));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_B_transposed);
  free(h_C_original);
  free(h_C_coalesced);
  free(h_C_transposed);
  free(h_C_shared);
  free(h_C_shared_register);
  free(h_C_blocktiling);
  free(h_C_blocktiling_2d);

  return 0;
}
