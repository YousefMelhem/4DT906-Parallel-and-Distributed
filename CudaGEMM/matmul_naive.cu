// Naive matrix multiplication implementation (baseline)
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

const int N = 1 << 11;  // 1024 x 1024

__global__ void matrixMul(const float *__restrict__ a, 
                         const float *__restrict__ b,
                         float *__restrict__ c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

void verify_result(vector<float> &a, vector<float> &b, vector<float> &c) {
    const float epsilon = 1e-3;
    float max_diff = 0.0f;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            float diff = std::abs(tmp - c[i * N + j]);
            max_diff = std::max(max_diff, diff);
            
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
    size_t bytes = N * N * sizeof(float);

    vector<float> h_a(N * N);
    vector<float> h_b(N * N);
    vector<float> h_c(N * N);

    generate(h_a.begin(), h_a.end(), []() { return (float)rand() / RAND_MAX; });
    generate(h_b.begin(), h_b.end(), []() { return (float)rand() / RAND_MAX; });

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    //int THREADS = 32;
    //dim3 threads(THREADS, THREADS);
    //dim3 blocks(N / THREADS, N / THREADS);
    dim3 threadsPerBlock(32, 16); dim3 blocksPerGrid(N / 32, N / 16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matrixMul<<<threadsPerBlock, blocksPerGrid>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Measure performance
    cudaEventRecord(start);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double total_ops = 2.0 * N * N * N;
    double gflops = (total_ops / (milliseconds / 1000.0)) / 1e9;

    cout << "Naive Implementation:\n";
    cout << "Matrix size: " << N << "x" << N << "\n";
    cout << "Execution time: " << milliseconds << " ms\n";
    cout << "GFLOPS: " << gflops << "\n";

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    //verify_result(h_a, h_b, h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
} 