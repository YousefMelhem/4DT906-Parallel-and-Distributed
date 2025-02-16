#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <Accelerate/Accelerate.h>

using namespace std;

const int N = 8192;
static float *A, *B, *B_trans, *C;

void transpose(float* src, float* dst, const int rows, const int cols) {
    // Use vDSP for transpose
    vDSP_mtrans(src,         // Source matrix
                1,           // Source matrix row stride
                dst,         // Destination matrix
                1,           // Destination matrix row stride
                cols,        // Number of columns in source/rows in destination
                rows);       // Number of rows in source/columns in destination
}

int main() {
    // Dynamically allocate memory with 16-byte alignment for better performance
    posix_memalign((void**)&A, 16, N * N * sizeof(float));
    posix_memalign((void**)&B, 16, N * N * sizeof(float));
    posix_memalign((void**)&B_trans, 16, N * N * sizeof(float));
    posix_memalign((void**)&C, 16, N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)rand() / RAND_MAX;
            B[i * N + j] = (float)rand() / RAND_MAX;
            C[i * N + j] = 0.0;
        }
    }

    struct timespec start, end;

    // Transpose B into B_trans
    // transpose(B, B_trans, N, N);

    clock_gettime(CLOCK_MONOTONIC, &start);
    ios_base::sync_with_stdio(false);

    // Matrix multiplication using vDSP
    // https://developer.apple.com/documentation/accelerate/1449984-vdsp_mmul
    vDSP_mmul(A,
              1,
              B_trans,
              1, 
              C,
              1,
              N, N, N);

    clock_gettime(CLOCK_MONOTONIC, &end);

    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    float tflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);

    cout << "GFLOPS: " << fixed << tflops << setprecision(6) << endl;
    cout << "|" << endl;
    cout << "s: " << fixed << time_taken << setprecision(6) << " sec" << endl;
    cout << "|" << endl;
    cout << "N: " << N << endl;
    cout << "|" << endl;

    // Cleanup
    free(A);
    free(B);
    free(B_trans);
    free(C);

    return 0;
}