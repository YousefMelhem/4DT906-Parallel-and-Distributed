#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

const int N = 1024;
const int blockSize = 16;

float A[N][N], B[N][N], C[N][N], Cvals[N][N], B_trans[N][N];

void transpose(float* src, float* dst, const int rows, const int cols) {
    #pragma omp parallel for
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

void gemm_omp() {
    int bi, bj, bk, i, j, k;
    #pragma omp parallel for private(bj, bk, i, j, k) shared(A, B_trans, C)
    for (bi = 0; bi < N; bi += blockSize)
        for (bj = 0; bj < N; bj += blockSize)
            for (bk = 0; bk < N; bk += blockSize)
                for (i = 0; i < blockSize; i++)
                    for (j = 0; j < blockSize; j++) {
                        float sum = 0.0f;
                        float* A_ptr = &A[bi + i][bk];
                        float* B_ptr = &B_trans[bk][bj + j];
                        for (k = 0; k < blockSize; k += 4) {
                            sum += A_ptr[k] * B_ptr[k];
                            sum += A_ptr[k + 1] * B_ptr[k + 1];
                            sum += A_ptr[k + 2] * B_ptr[k + 2];
                            sum += A_ptr[k + 3] * B_ptr[k + 3];
                        }
                        C[bi + i][bj + j] += sum;
                    }
}

int main() {
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f == nullptr) {
        cout << "please pregenerate python /tmp/matmul file" << endl;
        return -1;
    }
    fread(A, 1, sizeof(float) * N * N, f);
    fread(B, 1, sizeof(float) * N * N, f);
    fread(Cvals, 1, sizeof(float) * N * N, f);
    fclose(f);

    struct timespec start, end;

    transpose(&B[0][0], &B_trans[0][0], N, N);

    clock_gettime(CLOCK_MONOTONIC, &start);
    gemm_omp();
    clock_gettime(CLOCK_MONOTONIC, &end);

    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
    cout << "GFLOPS: " << fixed << setprecision(6) << gflops << endl;
    cout << "|" << endl;
    cout << "t: " << fixed << setprecision(6) << time_taken << endl;
    cout << "|" << endl;
    cout << "N: " << N << endl;
    cout << "|" << endl;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (abs(Cvals[y][x] - C[y][x]) > 0.001) {
                cout << "mismatch at " << y << " " << x << endl;
                cout << Cvals[y][x] << " " << C[y][x] << endl;
                return -1;
            }
        }
    }
    return 0;
}