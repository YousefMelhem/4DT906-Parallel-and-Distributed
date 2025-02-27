#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstdint>
#include <omp.h>
#include <arm_neon.h>

using namespace std;


#ifdef DEBGU
cout << "DEBUG: " << endl;
return -1
#endifb

const int N = 1024*2;
const int BLOCK_SIZE = 4;

#define BLOCK_Y 8  // Process 4 rows at a time
#define BLOCK_X 2  // Process 2 vectors at a time (8 elements total with NEON)
#define BLOCK 4    // Elements per vector (for float32x4_t)
                  
float A[N*N], B[N*N], B_trans[N*N], C[N*N], C_trans[N*N], Cvals[N*N], val[N*N];

void print_matrix(float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            cout << matrix[i * cols + j] << " ";
        cout << endl;
    }
    cout << endl;
}

void fetchMat() {
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f == nullptr) {
        cout << "please pregenerate python /tmp/matmul file" << endl;
        exit(1);
    }
    // the saved matricies in the order of A, B, C (has to match the order in gemm.py)
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(Cvals, 1, sizeof(float)*N*N, f);
    fclose(f);
}

void gemm_omp() {
    int bi, bj, bk, i, j;
    #pragma omp parallel for private(bj, bk, i, j) shared(A, B_trans, C)
    for (bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (bk = 0; bk < N; bk += BLOCK_SIZE) {
                for (i = 0; i < BLOCK_SIZE; i++) {
                    int row = (bi + i) * N;
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int col = (bj + j);
                        float sum = 0.0f;
                        sum += A[row + (bk + 0)] * B_trans[col * N + (bk + 0)];
                        sum += A[row + (bk + 1)] * B_trans[col * N + (bk + 1)];
                        sum += A[row + (bk + 2)] * B_trans[col * N + (bk + 2)];
                        sum += A[row + (bk + 3)] * B_trans[col * N + (bk + 3)];
                        C[row + col] += sum;
                    }
                }
            }
        }
    }
}


void gemm_omp_neon() { 
    #pragma omp parallel for
    for (int y = 0; y < N; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * BLOCK_X) {
            float32x4_t acc[BLOCK_Y][BLOCK_X] = {};

            for (int k = 0; k < N; k++) {
                for (int iy = 0; iy < BLOCK_Y; iy++) {
                    float a_val = A[(y + iy) * N + k];
                    float32x4_t a_vec = vdupq_n_f32(a_val);

                    for (int ix = 0; ix < BLOCK_X; ix++) {
                        float32x4_t b_vec = vld1q_f32(&B[k * N + x + ix * BLOCK]);
                        acc[iy][ix] = vfmaq_f32(acc[iy][ix], a_vec, b_vec);
                    }
                }
            }

            for (int iy = 0; iy < BLOCK_Y; iy++) {
                for (int ix = 0; ix < BLOCK_X; ix++) {
                    vst1q_f32(&C[(y + iy) * N + x + ix * BLOCK], acc[iy][ix]);
                }
            }
        }
    }
}

void gemm_omp_neonx2() {
    #pragma omp parallel for
    for (int y = 0; y < N; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * BLOCK_X) {
            // Initialize accumulators for 4x2 blocks
            float32x4x2_t acc[BLOCK_Y] = {};

            for (int k = 0; k < N; k++) {
                for (int iy = 0; iy < BLOCK_Y; iy++) {
                    // Broadcast a single A value to all lanes
                    float32x4_t a_vec = vdupq_n_f32(A[(y + iy) * N + k]);

                    // Load 2 vectors of 4 elements each from B (8 elements total)
                    float32x4x2_t b_vec = vld1q_f32_x2(&B[k * N + x]);

                    // Multiply and accumulate
                    acc[iy].val[0] = vfmaq_f32(acc[iy].val[0], a_vec, b_vec.val[0]);
                    acc[iy].val[1] = vfmaq_f32(acc[iy].val[1], a_vec, b_vec.val[1]);
                }
            }

            // Store results back to C
            for (int iy = 0; iy < BLOCK_Y; iy++) {
                vst1q_f32_x2(&C[(y + iy) * N + x], acc[iy]);
            }
        }
    }
}

void gemm_omp_neonx4() {
    #pragma omp parallel for
    for (int y = 0; y < N; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * BLOCK_X) {
            // Initialize accumulators for 4x4 blocks
            float32x4x4_t acc[BLOCK_Y] = {};

            for (int k = 0; k < N; k++) {
                for (int iy = 0; iy < BLOCK_Y; iy++) {
                    // Broadcast a single A value to all lanes
                    float32x4_t a_vec = vdupq_n_f32(A[(y + iy) * N + k]);

                    // Load 4 vectors of 4 elements each from B (16 elements total)
                    float32x4x4_t b_vec = vld1q_f32_x4(&B_trans[x * N + k]);

                    // Multiply and accumulate
                    acc[iy].val[0] = vfmaq_f32(acc[iy].val[0], a_vec, b_vec.val[0]);
                    acc[iy].val[1] = vfmaq_f32(acc[iy].val[1], a_vec, b_vec.val[1]);
                    acc[iy].val[2] = vfmaq_f32(acc[iy].val[2], a_vec, b_vec.val[2]);
                    acc[iy].val[3] = vfmaq_f32(acc[iy].val[3], a_vec, b_vec.val[3]);
                }
            }

            // Store results back to C
            for (int iy = 0; iy < BLOCK_Y; iy++) {
                vst1q_f32_x4(&C[(y + iy) * N + x], acc[iy]);
            }
        }
    }
}

#define RUN_COUNT 15

int main() {

    fetchMat();
    
    struct timespec start, end;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_trans[i * N + j] = B[j * N + i];
        }
    }

    #if RUN_COUNT == 1
    clock_gettime(CLOCK_MONOTONIC, &start);
    gemm_omp();
    clock_gettime(CLOCK_MONOTONIC, &end);
    #else

    float avg = 0.0;
    for (int i = 0; i < RUN_COUNT; i++) {
        // Reset C to zero
        #pragma omp parallel for
        for (int y = 0; y < N*N; y++) {
            C[y] = 0.0f;
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        gemm_omp_neon();
        clock_gettime(CLOCK_MONOTONIC, &end);

        float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
        avg += gflops;
        cout << "GFLOPS: " << fixed << setprecision(6) << gflops << endl;

    }

    // // print all matricies
    // print_matrix(A, N, N);
    // print_matrix(B, N, N);
    // print_matrix(C, N, N);
    // print_matrix(Cvals, N, N);


    cout << "Average GFLOPS: " << fixed << setprecision(6) << avg / RUN_COUNT << endl;
    #endif

    #if RUN_COUNT == 1
    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
    cout << "GFLOPS: " << fixed << setprecision(6) << gflops << endl;
    cout << "|" << endl;
    cout << "t: " << fixed << setprecision(6) << time_taken << endl;
    cout << "|" << endl;
    cout << "N: " << N << endl;
    cout << "|" << endl;

    cout << "Output verified!" << endl;
    #endif


    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         C_trans[i * N + j] = C[j * N + i];
    //     }
    // }

    for (int y = 0; y < N*N; y++) {
        if (abs(Cvals[y] - C[y]) > 0.001) {
            cout << "mismatch at " << y / N << " " << y % N << endl;
            cout << Cvals[y] << " " << C[y] << endl;
            return -1;
        }
    }

    cout << "Output verified!" << endl;

    return 0;
}
