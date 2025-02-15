#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <arm_neon.h>
#include <ctime>
#include <omp.h>

using namespace std;

const int BLOCK_SIZE = 32;
const int N = 512*2;

void matrix_init(float32_t *M, uint32_t rows, uint32_t cols, float32_t val) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        M[i] = val;
    }
}

void matrix_init_rand(float32_t *M, uint32_t numvals) {
    for (uint32_t i = 0; i < numvals * numvals; i++) {
        M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void print_matrix(float32_t *M, uint32_t cols, uint32_t rows) {
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            cout << M[j * rows + i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void matmul(float32_t *A, float32_t *B, float32_t *C, uint32_t N) {
    int A_idx;
    int B_idx;
    int C_idx;
    
    float32x4_t A0, A1, A2, A3;
    float32x4_t B0, B1, B2, B3;
    float32x4_t C0, C1, C2, C3;
    
    clock_t t = clock();
    
    #pragma omp parallel for schedule(static)
    for (int i_idx = 0; i_idx < N; i_idx += BLOCK_SIZE) {
        for (int j_idx = 0; j_idx < N; j_idx += BLOCK_SIZE) {
            for (int k_idx = 0; k_idx < N; k_idx += BLOCK_SIZE) {
                for (int i = i_idx; i < i_idx + BLOCK_SIZE; i += 4) {
                    for (int j = j_idx; j < j_idx + BLOCK_SIZE; j += 4) {
                        float32x4_t C0 = vld1q_f32(C + i * N + j);
                        float32x4_t C1 = vld1q_f32(C + (i + 1) * N + j);
                        float32x4_t C2 = vld1q_f32(C + (i + 2) * N + j);
                        float32x4_t C3 = vld1q_f32(C + (i + 3) * N + j);
                        for (int k = k_idx; k < k_idx + BLOCK_SIZE; k += 4) {
                            float32x4_t A0 = vld1q_f32(A + i * N + k);
                            float32x4_t A1 = vld1q_f32(A + (i + 1) * N + k);
                            float32x4_t A2 = vld1q_f32(A + (i + 2) * N + k);
                            float32x4_t A3 = vld1q_f32(A + (i + 3) * N + k);

                            float32x4_t B0 = vld1q_f32(B + k * N + j);
                            C0 = vmlaq_laneq_f32(C0, A0, B0, 0);
                            C0 = vmlaq_laneq_f32(C0, A1, B0, 1);
                            C0 = vmlaq_laneq_f32(C0, A2, B0, 2);
                            C0 = vmlaq_laneq_f32(C0, A3, B0, 3);

                            float32x4_t B1 = vld1q_f32(B + (k + 1) * N + j);
                            C1 = vmlaq_laneq_f32(C1, A0, B1, 0);
                            C1 = vmlaq_laneq_f32(C1, A1, B1, 1);
                            C1 = vmlaq_laneq_f32(C1, A2, B1, 2);
                            C1 = vmlaq_laneq_f32(C1, A3, B1, 3);

                            float32x4_t B2 = vld1q_f32(B + (k + 2) * N + j);
                            C2 = vmlaq_laneq_f32(C2, A0, B2, 0);
                            C2 = vmlaq_laneq_f32(C2, A1, B2, 1);
                            C2 = vmlaq_laneq_f32(C2, A2, B2, 2);
                            C2 = vmlaq_laneq_f32(C2, A3, B2, 3);

                            float32x4_t B3 = vld1q_f32(B + (k + 3) * N + j);
                            C3 = vmlaq_laneq_f32(C3, A0, B3, 0);
                            C3 = vmlaq_laneq_f32(C3, A1, B3, 1);
                            C3 = vmlaq_laneq_f32(C3, A2, B3, 2);
                            C3 = vmlaq_laneq_f32(C3, A3, B3, 3);
                        }
                        vst1q_f32(C + i * N + j, C0);
                        vst1q_f32(C + (i + 1) * N + j, C1);
                        vst1q_f32(C + (i + 2) * N + j, C2);
                        vst1q_f32(C + (i + 3) * N + j, C3);
                    }
                }
            }
        }
    }



    t = clock() - t;
    double time_taken = static_cast<double>(t) / CLOCKS_PER_SEC;
    float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
    cout << "GFLOPS: " << gflops << endl;
}

int main() {
    // Allocate matrices dynamically for larger sizes
    float32_t *A = new float32_t[N * N];
    float32_t *B = new float32_t[N * N];
    float32_t *C = new float32_t[N * N];

    // Initialize matrices
    matrix_init_rand(A, N);
    matrix_init_rand(B, N);
    matrix_init(C, N, N, 0);

    // print_matrix(A, N, N);
    // print_matrix(B, N, N);
    // print_matrix(C, N, N);

    // Perform matrix multiplication
    matmul(A, B, C, N);

    // Print results (commented out for large matrices)
    // print_matrix(A, N, N);
    // print_matrix(B, N, N);
    // print_matrix(C, N, N);

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
