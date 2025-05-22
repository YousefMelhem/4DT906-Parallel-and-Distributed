// This program computes matrix multiplication using shared memory tiling
// Optimized version targeting high GFLOPS

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Matrix size and tile size
const int N = 1 << 10;  // 1024 x 1024
const int TILE_SIZE = 32;
const int BLOCK_ROWS = 8;  // Number of rows each thread computes

// Optimized matrix multiplication kernel
__global__ void matrixMul(const float *__restrict__ a, 
                         const float *__restrict__ b,
                         float *__restrict__ c) {
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty * BLOCK_ROWS;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Shared memory tiles
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    // Accumulator registers
    float sum[BLOCK_ROWS] = {0.0f};

    // Loop over tiles
    for (int m = 0; m < N; m += TILE_SIZE) {
        // Load tiles into shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            // Load tile from matrix A
            if (row + i < N && m + tx < N) {
                s_a[ty * BLOCK_ROWS + i][tx] = a[(row + i) * N + m + tx];
            } else {
                s_a[ty * BLOCK_ROWS + i][tx] = 0.0f;
            }
            
            // Load tile from matrix B
            if (m + ty * BLOCK_ROWS + i < N && col < N) {
                s_b[ty * BLOCK_ROWS + i][tx] = b[(m + ty * BLOCK_ROWS + i) * N + col];
            } else {
                s_b[ty * BLOCK_ROWS + i][tx] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < BLOCK_ROWS; i++) {
                sum[i] += s_a[ty * BLOCK_ROWS + i][k] * s_b[k][tx];
            }
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; i++) {
        if (row + i < N && col < N) {
            c[(row + i) * N + col] = sum[i];
        }
    }
}

// Check result on the CPU
void verify_result(vector<float> &a, vector<float> &b, vector<float> &c) {
    const float epsilon = 1e-3;  // Increased epsilon for floating-point comparison
    float max_diff = 0.0f;
    
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            float tmp = 0;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Track maximum difference
            float diff = std::abs(tmp - c[i * N + j]);
            max_diff = std::max(max_diff, diff);
            
            // Check against the CPU result with epsilon
            if (diff > epsilon) {
                cout << "Mismatch at (" << i << "," << j << "): " 
                     << "CPU=" << tmp << " GPU=" << c[i * N + j] 
                     << " diff=" << diff << "\n";
                assert(false);
            }
        }
    }
    cout << "Maximum difference: " << max_diff << "\n";
}

int main() {
    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(float);

    // Host vectors
    vector<float> h_a(N * N);
    vector<float> h_b(N * N);
    vector<float> h_c(N * N);

    // Initialize matrices with random values
    generate(h_a.begin(), h_a.end(), []() { return (float)rand() / RAND_MAX; });
    generate(h_b.begin(), h_b.end(), []() { return (float)rand() / RAND_MAX; });

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    dim3 threads(TILE_SIZE, TILE_SIZE / BLOCK_ROWS);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Measure performance
    cudaEventRecord(start);
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate GFLOPS
    double total_ops = 2.0 * N * N * N;  // 2 operations per element (multiply and add)
    double gflops = (total_ops / (milliseconds / 1000.0)) / 1e9;

    cout << "Matrix size: " << N << "x" << N << "\n";
    cout << "Execution time: " << milliseconds << " ms\n";
    cout << "GFLOPS: " << gflops << "\n";

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}