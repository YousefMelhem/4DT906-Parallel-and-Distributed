#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstdint>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int N = 2048;

// float A[N][N], B[N][N], C[N][N], Cvals[N][N];
 
float A[N][N] __attribute__((aligned(64)));;
float B[N][N] __attribute__((aligned(64)));;
float BT[N][N] __attribute__((aligned(64)));;
float C[N][N] __attribute__((aligned(64)));;
float Cvals[N][N] __attribute__((aligned(64)));;


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


#define BLOCKSIZE 4

void gemm_omp_ikj_neon() {
    int bi, bk, bj, i, k, j;

    #pragma omp parallel for private(bk, bj, i, k, j) shared(A, B, C)
    for(bi = 0; bi < N; bi += BLOCKSIZE)
        for(bk = 0; bk < N; bk += BLOCKSIZE)
            for(bj = 0; bj < N; bj += BLOCKSIZE)

                for(i = 0; i < BLOCKSIZE; i++)
                    for(k = 0; k < BLOCKSIZE; k++) {
                        float a_val = A[bi + i][bk + k];

                        float32x4_t a_vec = vdupq_n_f32(a_val);
                        
                        for(j = 0; j < BLOCKSIZE; j+=4) {
                            float32x4_t b_vec = vld1q_f32(&B[bk + k][bj + j]);
                            float32x4_t c_vec = vld1q_f32(&C[bi + i][bj + j]);
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                            
                            vst1q_f32(&C[bi + i][bj + j], c_vec);
                        }

                    }
}


void gemm_omp_ikj() {
    int bi, bk, bj, i, k, j;

    #pragma omp parallel for private(bk, bj, i, k, j) shared(A, B, C)
    for(bi = 0; bi < N; bi += BLOCKSIZE)
        for(bk = 0; bk < N; bk += BLOCKSIZE)
            for(bj = 0; bj < N; bj += BLOCKSIZE)

                for(i = 0; i < BLOCKSIZE; i++)
                    for(k = 0; k < BLOCKSIZE; k++) {
                        float a_val = A[bi + i][bk + k];
                        for(j = 0; j < BLOCKSIZE; j+=2) {
                            C[bi + i][bj + j + 0] += a_val * B[bk + k][bj + j + 0];
                            C[bi + i][bj + j + 1] += a_val * B[bk + k][bj + j + 1];
                        }
                    }
}

#define RUN_COUNT 15

int main() {

    fetchMat();
    struct timespec start, end;

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
        gemm_omp_ikj_neon();
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
