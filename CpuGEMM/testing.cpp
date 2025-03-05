#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstdint>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int N = 2048;
const int blockSize = 256;

float A[N][N], B[N][N], BT[N][N], C[N][N], Cvals[N][N];
 
// float A[N][N] __attribute__((aligned(32)));;
// float B[N][N] __attribute__((aligned(32)));;
// float BT[N][N] __attribute__((aligned(32)));;
// float C[N][N] __attribute__((aligned(32)));;
// float Cvals[N][N] __attribute__((aligned(32)));;

void transpose(float* src, float* dst, const int rows, const int cols) {
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

// check if the matrix is transposed correctly
void print_matrix(float* matrix, const int rows, const int cols) {
    cout << " " << endl;
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
    fread(Cvals, 1, sizeof(float)*N*N, f);
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fclose(f);
}

void gemm_stupid(){
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * BT[j][k];
            }
        }
    }
}

void gemm_omp() {
    // Cache-based blocking sizes
    const int L2_BLOCK = 512;  // L2 cache blocking
    const int L1_BLOCK = 64;   // L1 cache blocking
    const int REG_BLOCK = 4;   // Register blocking
    
    // Cache line size is 128 bytes (from hw.cachelinesize)
    // Each float is 4 bytes, so 32 floats per cache line
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int bi=0; bi<N; bi+=L2_BLOCK) {
        for(int bj=0; bj<N; bj+=L2_BLOCK) {
            // Create local blocked buffers (can fit in L2)
            float Cblock[L2_BLOCK][L2_BLOCK] = {0};
            
            // Compute C[bi:bi+L2_BLOCK, bj:bj+L2_BLOCK]
            for(int bk=0; bk<N; bk+=L2_BLOCK) {
                // L1 cache blocking
                for(int i=0; i<min(L2_BLOCK, N-bi); i+=L1_BLOCK) {
                    for(int j=0; j<min(L2_BLOCK, N-bj); j+=L1_BLOCK) {
                        for(int k=0; k<min(L2_BLOCK, N-bk); k+=L1_BLOCK) {
                            // Register blocking - process 4x4 blocks at a time
                            for(int ii=0; ii<min(L1_BLOCK, N-bi-i); ii+=REG_BLOCK) {
                                for(int jj=0; jj<min(L1_BLOCK, N-bj-j); jj+=REG_BLOCK) {
                                    // Initialize register block
                                    float c00 = Cblock[i+ii][j+jj];
                                    float c01 = Cblock[i+ii][j+jj+1];
                                    float c02 = Cblock[i+ii][j+jj+2];
                                    float c03 = Cblock[i+ii][j+jj+3];
                                    
                                    float c10 = Cblock[i+ii+1][j+jj];
                                    float c11 = Cblock[i+ii+1][j+jj+1];
                                    float c12 = Cblock[i+ii+1][j+jj+2];
                                    float c13 = Cblock[i+ii+1][j+jj+3];
                                    
                                    float c20 = Cblock[i+ii+2][j+jj];
                                    float c21 = Cblock[i+ii+2][j+jj+1];
                                    float c22 = Cblock[i+ii+2][j+jj+2];
                                    float c23 = Cblock[i+ii+2][j+jj+3];
                                    
                                    float c30 = Cblock[i+ii+3][j+jj];
                                    float c31 = Cblock[i+ii+3][j+jj+1];
                                    float c32 = Cblock[i+ii+3][j+jj+2];
                                    float c33 = Cblock[i+ii+3][j+jj+3];
                                    
                                    // Inner product loop - good candidate for vectorization
                                    #pragma clang loop vectorize(enable)
                                    for(int kk=0; kk<min(L1_BLOCK, N-bk-k); kk++) {
                                        // Load elements from A
                                        float a0 = A[bi+i+ii][bk+k+kk];
                                        float a1 = A[bi+i+ii+1][bk+k+kk];
                                        float a2 = A[bi+i+ii+2][bk+k+kk];
                                        float a3 = A[bi+i+ii+3][bk+k+kk];
                                        
                                        // Load elements from B
                                        float b0 = B[bk+k+kk][bj+j+jj];
                                        float b1 = B[bk+k+kk][bj+j+jj+1];
                                        float b2 = B[bk+k+kk][bj+j+jj+2];
                                        float b3 = B[bk+k+kk][bj+j+jj+3];
                                        
                                        // Update C block
                                        c00 += a0 * b0;
                                        c01 += a0 * b1;
                                        c02 += a0 * b2;
                                        c03 += a0 * b3;
                                        
                                        c10 += a1 * b0;
                                        c11 += a1 * b1;
                                        c12 += a1 * b2;
                                        c13 += a1 * b3;
                                        
                                        c20 += a2 * b0;
                                        c21 += a2 * b1;
                                        c22 += a2 * b2;
                                        c23 += a2 * b3;
                                        
                                        c30 += a3 * b0;
                                        c31 += a3 * b1;
                                        c32 += a3 * b2;
                                        c33 += a3 * b3;
                                    }
                                    
                                    // Store results back
                                    Cblock[i+ii][j+jj] = c00;
                                    Cblock[i+ii][j+jj+1] = c01;
                                    Cblock[i+ii][j+jj+2] = c02;
                                    Cblock[i+ii][j+jj+3] = c03;
                                    
                                    Cblock[i+ii+1][j+jj] = c10;
                                    Cblock[i+ii+1][j+jj+1] = c11;
                                    Cblock[i+ii+1][j+jj+2] = c12;
                                    Cblock[i+ii+1][j+jj+3] = c13;
                                    
                                    Cblock[i+ii+2][j+jj] = c20;
                                    Cblock[i+ii+2][j+jj+1] = c21;
                                    Cblock[i+ii+2][j+jj+2] = c22;
                                    Cblock[i+ii+2][j+jj+3] = c23;
                                    
                                    Cblock[i+ii+3][j+jj] = c30;
                                    Cblock[i+ii+3][j+jj+1] = c31;
                                    Cblock[i+ii+3][j+jj+2] = c32;
                                    Cblock[i+ii+3][j+jj+3] = c33;
                                }
                            }
                        }
                    }
                }
            }
            
            // Copy the computed block back to C
            for(int i=0; i<min(L2_BLOCK, N-bi); i++) {
                for(int j=0; j<min(L2_BLOCK, N-bj); j++) {
                    C[bi+i][bj+j] = Cblock[i][j];
                }
            }
        }
    }
}



#define RUN_COUNT 15

int main() {

    fetchMat();
    struct timespec start, end;

    transpose(&B[0][0], &BT[0][0], N, N);

    #if RUN_COUNT == 1
    clock_gettime(CLOCK_MONOTONIC, &start);
    gemm_omp();
    clock_gettime(CLOCK_MONOTONIC, &end);
    #else

    float avg = 0.0;
    for (int i = 0; i < RUN_COUNT; i++) {
        // Reset C to zero
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                C[y][x] = 0.0f;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &start);
        gemm_stupid();
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
        avg += gflops;
        cout << "GFLOPS: " << fixed << setprecision(6) << gflops << endl;

    }

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

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (abs(Cvals[y][x] - C[y][x]) > 0.001) {
                cout << "mismatch at " << y << " " << x << endl;
                cout << Cvals[y][x] << " " << C[y][x] << endl;
                return -1;
            }
        }
    }

    cout << "Output verified!" << endl;

    return 0;
}