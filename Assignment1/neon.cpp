#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include <arm_neon.h>



using namespace std;

const int N = 1024;
const int blockSize = 16;

float A[N][N], B[N][N], C[N][N], Cvals[N][N], B_trans[N][N];

void transpose(float* src, float* dst, const int rows, const int cols) {
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

void gemm(float* A, float* B, float* C, int N, int blockSize) {
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                gemm_neon_block(A, B, C, N, blockSize, i, j, k);
            }
        }
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
    gemm();
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