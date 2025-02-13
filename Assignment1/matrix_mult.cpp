#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

const int N = 1024;
float A[N][N], B[N][N], B_trans[N][N], C[N][N];

void transpose(float* src, float* dst, const int rows, const int cols) {
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

int main() {
    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }

    struct timespec start, end;

    // Transpose B into B_trans
    transpose(&B[0][0], &B_trans[0][0], N, N);

    clock_gettime(CLOCK_MONOTONIC, &start);
    ios_base::sync_with_stdio(false);

    // Matrix multiplication using transposed B
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B_trans[j][k];
            C[i][j] = sum;
        }

    clock_gettime(CLOCK_MONOTONIC, &end);

    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // gflops
    float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);

    cout << "Time taken by program is : " << fixed << time_taken << setprecision(6) << " sec" << endl;
    cout << "GFLOPS: " << fixed << gflops << setprecision(6) << endl;

    return 0;
}