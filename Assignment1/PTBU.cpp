#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstdint>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int N = 1024*2;
const int blockSize = 4;

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

void gemm_omp(){
    int bi, bk, bj, i, k, j;
    #pragma omp parallel for private(bk, bj, i, k, j) shared(A, B, C)
    for(bi=0; bi<N; bi+=blockSize)
        for(bk=0; bk<N; bk+=blockSize)
            for(bj=0; bj<N; bj+=blockSize)
                for(i=0; i<blockSize; i++)
                    for(k=0; k<blockSize; k++) {
                        float a_val = A[bi + i][bk + k];
                        #pragma vector always
                        for(j=0; j<blockSize; j++) {
                            C[bi + i][bj + j] += a_val * B[bk + k][bj + j];
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
        gemm_omp();
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