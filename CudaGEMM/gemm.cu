#include <stdio.h>
#include <cuda_runtime.h>

#define N 16384  // Matrix size (N x N, must be power of 2)

// CUDA kernel for GEMM: C = A * B
__global__ void gemm(float *A, float *B, float *C, int n) {
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

// Function to measure GFLOPS
float measure_gemm(float *d_A, float *d_B, float *d_C, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(n / 16, n / 16);

    cudaEventRecord(start);
    gemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // FLOPs = 2 * N^3 (one multiply and one add per element)
    float gflops = (2.0 * n * n * n) / (milliseconds * 1e6);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return gflops;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    float gflops = measure_gemm(d_A, d_B, d_C, N);

    printf("GEMM Performance: %.2f GFLOPS\n", gflops);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

