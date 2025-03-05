#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstdint>
#include <omp.h>
#include <arm_neon.h>
#include "amx.hpp"
#include <cstring>

using namespace std;


#ifdef DEBUG
  #define N 4
#endif

#ifndef N
  #define N 1024*2
#endif


#define BLOCK_SIZE 4

#define BLOCK 4   // Elements per vector (for float32x4_t)
#define BLOCK_Y 4 // rows at a time

#define BLOCK_X 8 // vectors at a time

                  

float A[N*N] __attribute__ ((aligned (32)));
float B[N*N] __attribute__ ((aligned (32)));
float BT[N*N] __attribute__ ((aligned (32)));
float C[N*N] __attribute__ ((aligned (32)));
float Cvals[N*N] __attribute__ ((aligned (32)));

void print_matrix(float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            cout << matrix[i * cols + j] << " ";
        cout << endl;
    }
    cout << endl;
}


void gemm_omp() {
    int bi, bj, bk, i, j;
    #pragma omp parallel for private(bj, bk, i, j) shared(A, BT, C)
    for (bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (bk = 0; bk < N; bk += BLOCK_SIZE) {

                for (i = 0; i < BLOCK_SIZE; i++) {
                    int row = (bi + i) * N;
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int col = (bj + j);
                        float sum = 0.0f;
                        sum += A[row + (bk + 0)] * BT[col * N + (bk + 0)];
                        sum += A[row + (bk + 1)] * BT[col * N + (bk + 1)];
                        sum += A[row + (bk + 2)] * BT[col * N + (bk + 2)];
                        sum += A[row + (bk + 3)] * BT[col * N + (bk + 3)];
                        C[row + col] += sum;
                    }
                }
            }
        }
    }
}


void gemm_omp_neon() { 
    #pragma omp parallel for
    for (int i = 0; i < N; i += BLOCK_Y) {
        for (int k = 0; k < N; k++) {
            for (int ii = 0; ii < BLOCK_Y; ii++) {
                float32x4_t a_vec = vdupq_n_f32(A[(i + ii) * N + k]);
                
                for (int j = 0; j < N; j += BLOCK * BLOCK_X) {
                    for (int jj = 0; jj < BLOCK_X; jj++) {
                        float32x4_t c_vec = vld1q_f32(&C[(i + ii) * N + j + jj * BLOCK]);
                        float32x4_t b_vec = vld1q_f32(&B[k * N + j + jj * BLOCK]);
                        c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
                        vst1q_f32(&C[(i + ii) * N + j + jj * BLOCK], c_vec);
                    }
                }
            }
        }
    }
}

void gemm_omp_neonx2() {
    //#pragma omp parallel for
    for (int y = 0; y < N; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * BLOCK_X) {
            // Initialize accumulators for 4x2 blocks
            float32x4x2_t acc[BLOCK_Y] = {};

            for (int k = 0; k < N; k++) {
                for (int iy = 0; iy < BLOCK_Y; iy++) {
                    float32x4_t a_vec = vdupq_n_f32(A[(y + iy) * N + k]);

                    float32x4x2_t b_vec = vld1q_f32_x2(&B[k * N + x]);

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

// does not work
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
                    float32x4x4_t b_vec = vld1q_f32_x4(&B[k * N + x]);

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

void gemm_omp_amx() {
    AMX_SET();
    memset(C, 0, sizeof(float) * N * N); // Initialize output matrix C to zero

    constexpr int TILE_SIZE = 16; // Use 16x16 tiles for AMX
    constexpr int NUM_X_REGS = 8; // Number of X registers
    constexpr int NUM_Y_REGS = 8; // Number of Y registers
    constexpr int Z_ROWS = 4;     // Use 4 z_rows for parallel accumulation

    float zero_buffer[16] = {0}; // Zero buffer for initializing Z registers

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < N; m += TILE_SIZE) {
        for (int n = 0; n < N; n += TILE_SIZE) {
            float z_buffer[TILE_SIZE][TILE_SIZE] __attribute__((aligned(64)));

            // Initialize Z registers to zero
            for (int i = 0; i < TILE_SIZE; ++i) {
                for (int z = 0; z < Z_ROWS; ++z) {
                    AMX_LDZ(ldstz().row_index(i * Z_ROWS + z).bind(zero_buffer));
                }
            }

            for (int k = 0; k < N; k += TILE_SIZE) {
                // Load A tile (m:TILE_SIZE, k:TILE_SIZE) into Y registers
                for (int y_reg = 0; y_reg < NUM_Y_REGS; ++y_reg) {
                    int row = m + y_reg;
                    if (row >= N) break;
                    float* a_ptr = &A[row * N + k];
                    AMX_LDY(ldxy().register_index(y_reg).multiple().multiple_four().bind(a_ptr));
                }

                // Load B tile (k:TILE_SIZE, n:TILE_SIZE) into X registers
                for (int x_reg = 0; x_reg < NUM_X_REGS; ++x_reg) {
                    int col = n + x_reg * (TILE_SIZE / NUM_X_REGS);
                    if (col >= N) break;
                    float* b_ptr = &B[k * N + col];
                    AMX_LDX(ldxy().register_index(x_reg).multiple().multiple_four().bind(b_ptr));
                }

                // Perform matrix multiplication with parallel accumulation
                for (int z = 0; z < Z_ROWS; ++z) {
                    AMX_MATFP(matfp()
                        .dtype_mode(matfp_dtype_t::f32f32)
                        .z_row(z)
                        .x_offset(0)
                        .y_offset(0));
                }
            }

            // Store Z registers to C
            for (int i = 0; i < TILE_SIZE; ++i) {
                for (int z = 0; z < Z_ROWS; ++z) {
                    AMX_STZ(ldstz().row_index(i * Z_ROWS + z).bind(z_buffer[i]));
                }
                for (int j = 0; j < TILE_SIZE; ++j) {
                    C[(m + i) * N + (n + j)] += z_buffer[i][j]; // Accumulate results
                }
            }
        }
    }

    AMX_CLR();
}

void gemm_omp_neon_advanced_lanes() {
    #pragma omp parallel for
    for (int y = 0; y < N; y += BLOCK_Y) {
        for (int x = 0; x < N; x += BLOCK * 2) {  // Process 8 elements (2 vectors) at once
            // Zero accumulators
            float32x4_t acc0[BLOCK_Y] = {};
            float32x4_t acc1[BLOCK_Y] = {};
            
            for (int k = 0; k < N; k++) {
                for (int iy = 0; iy < BLOCK_Y; iy++) {
                    // Load a single value from A to broadcast across lanes
                    float a_val = A[(y + iy) * N + k];
                    
                    // Load two groups of 4 elements (8 total) from B
                    float32x4_t b0 = vld1q_f32(&B[k * N + x]);
                    float32x4_t b1 = vld1q_f32(&B[k * N + x + 4]);
                    
                    // Broadcast A value for vector multiply-accumulate
                    acc0[iy] = vfmaq_n_f32(acc0[iy], b0, a_val);
                    acc1[iy] = vfmaq_n_f32(acc1[iy], b1, a_val);
                }
            }
            
            // Store accumulated results back to C
            for (int iy = 0; iy < BLOCK_Y; iy++) {
                // Load, add, and store back to C
                float32x4_t c0 = vld1q_f32(&C[(y + iy) * N + x]);
                float32x4_t c1 = vld1q_f32(&C[(y + iy) * N + x + 4]);
                
                vst1q_f32(&C[(y + iy) * N + x], vaddq_f32(c0, acc0[iy]));
                vst1q_f32(&C[(y + iy) * N + x + 4], vaddq_f32(c1, acc1[iy]));
            }
        }
    }
}

void initmat(){
    
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f == nullptr) {
        cout << "please pregenerate python /tmp/matmul file" << endl;
        exit(1);
    }

    #ifdef DEBUG
    for (int i = 0; i < N*N; i++) 
            A[i] = 1.0f;
        
    
    for (int i = 0; i < N*N; i++)
        B[i] = i;

    fread(Cvals, 1, sizeof(float)*N*N, f);

    #endif
    #ifndef DEBUG
        // the saved matricies in the order of A, B, C (has to match the order in gemm.py)
        fread(Cvals, 1, sizeof(float)*N*N, f);
        fread(A, 1, sizeof(float)*N*N, f);
        fread(B, 1, sizeof(float)*N*N, f);
    #endif

    fclose(f);

    
    
}

#define RUN_COUNT 15

int main() {

    initmat();
    
    struct timespec start, end;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[i * N + j] = B[j * N + i];
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

    #ifdef DEBUG
    print_matrix(A, N, N);
    print_matrix(B, N, N);
    print_matrix(C, N, N);
    #endif


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

    
    #ifndef DEBUG
    for (int y = 0; y < N*N; y++) {
        if (abs(Cvals[y] - C[y]) > 0.001) {
            cout << "mismatch at " << y / N << " " << y % N << endl;
            cout << Cvals[y] << " " << C[y] << endl;
            return -1;
        }
    }
    cout << "Output verified!" << endl;
    #endif

    return 0;
}
