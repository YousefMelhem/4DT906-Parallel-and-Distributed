// nvcc gemm.cu -o compiled/gemm -O3 && ./compiled/gemm
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 8192       // Matrix size (N x N, must be power of 2)
#define BLOCKSIZE 32 // Size of the shared memory tile
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

// Original kernel for comparison
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

// Shared memory cache-blocked kernel with register blocking
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
  int row2 = row1 + BLOCKSIZE / 2; // Second row (if within bounds)
  int col2 = col1 + BLOCKSIZE / 2; // Second column (if within bounds)

  // Accumulate results in registers
  float sum11 = 0.0f;
  float sum12 = 0.0f;
  float sum21 = 0.0f;
  float sum22 = 0.0f;

  // Loop over all tiles required to compute the result
  for (int tileIdx = 0; tileIdx < n / BLOCKSIZE; tileIdx++) {
    // Each thread loads one element of A and B into shared memory
    As[threadRow][threadCol] = A[row1 * n + (tileIdx * BLOCKSIZE + threadCol)];
    Bs[threadRow][threadCol] = B[(tileIdx * BLOCKSIZE + threadRow) * n + col1];

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

// Compute dot products for this tile using registers
#pragma unroll 8
    for (int k = 0; k < BLOCKSIZE; k++) {
      // Load values into registers to reduce shared memory accesses
      float aval = As[threadRow][k];
      float bval = Bs[k][threadCol];

      // Compute partial sums for the 2x2 block
      sum11 += aval * bval;

      // Compute for additional elements if within bounds
      if (col2 < n && row2 < n) {
        float aval2 = As[threadRow + BLOCKSIZE / 2][k];
        float bval2 = Bs[k][threadCol + BLOCKSIZE / 2];

        sum12 += aval * bval2;
        sum21 += aval2 * bval;
        sum22 += aval2 * bval2;
      }
    }

    // Ensure computation is complete before loading next tile
    __syncthreads();
  }

  // Write results to global memory
  if (row1 < n && col1 < n) {
    C[row1 * n + col1] = sum11;

    // Write additional results if within bounds
    if (col2 < n && row1 < n)
      C[row1 * n + col2] = sum12;
    if (row2 < n && col1 < n)
      C[row2 * n + col1] = sum21;
    if (row2 < n && col2 < n)
      C[row2 * n + col2] = sum22;
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
  } else if (strcmp(kernel_name, "shared") == 0) {
    blockDim = dim3(BLOCKSIZE, BLOCKSIZE);
    gridDim = dim3(CEIL_DIV(n, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  } else if (strcmp(kernel_name, "shared_register") == 0) {
    blockDim = dim3(BLOCKSIZE / 2, BLOCKSIZE / 2);
    gridDim = dim3(CEIL_DIV(n, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
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
  float *h_C_original = (float *)malloc(size);
  float *h_C_shared = (float *)malloc(size);
  float *h_C_shared_register = (float *)malloc(size);

  if (!h_A || !h_B || !h_C_original || !h_C_shared || !h_C_shared_register) {
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

  // Device memory allocation
  float *d_A, *d_B, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

  // Copy data to device
  printf("Copying data to device...\n");
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Run original kernel
  printf("\n=== Running Original Kernel ===\n");
  float gflops_original =
      measure_kernel("original", gemm_original, d_A, d_B, d_C, N, iterations);
  CHECK_CUDA_ERROR(cudaMemcpy(h_C_original, d_C, size, cudaMemcpyDeviceToHost));

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

  // Verify results
  printf("\nVerifying Shared Memory vs Original:\n");
  int diff_count = 0;
  double max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs((double)h_C_original[i] - (double)h_C_shared[i]);
    if (diff > 1e-5)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  printf("\nVerifying Shared Memory with Register Blocking vs Original:\n");
  diff_count = 0;
  max_diff = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff =
        fabs((double)h_C_original[i] - (double)h_C_shared_register[i]);
    if (diff > 1e-5)
      diff_count++;
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  printf("  Max difference: %.6e\n", max_diff);
  printf("  Cells with differences: %d (%.2f%%)\n", diff_count,
         100.0 * diff_count / (N * N));

  // Print performance summary
  printf("\n=== Performance Summary ===\n");
  printf("Original GEMM Performance: %.2f GFLOPS\n", gflops_original);
  printf("Shared Memory GEMM Performance: %.2f GFLOPS\n", gflops_shared);
  printf("Shared Memory with Register Blocking GEMM Performance: %.2f GFLOPS\n",
         gflops_shared_register);
  printf("Speedup (Shared Memory vs Original): %.2fx\n",
         gflops_shared / gflops_original);
  printf("Speedup (Shared Memory with Register Blocking vs Original): %.2fx\n",
         gflops_shared_register / gflops_original);

  // Free memory
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C_original);
  free(h_C_shared);
  free(h_C_shared_register);

  return 0;
}
