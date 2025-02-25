#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

const int N = 1024*2;
const int blockSize = 4;
                  
float A[N][N], B[N][N], B_trans[N][N], C[N][N], Cvals[N][N];

void transpose(float* src, float* dst, const int rows, const int cols) {
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

// check if the matrix is transposed correctly
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

void gemm_omp(){
    int bi, bj, bk, i, j, k;
    // Matrix multiplication using transposed B
    // ading bi in the private loses me 3gflops
    #pragma omp parallel for private(bj, bk, i, j) shared(A, B_trans, C)
    for(bi=0; bi<N; bi+=blockSize)
        for(bj=0; bj<N; bj+=blockSize)
            for(bk=0; bk<N; bk+=blockSize)

                for (i = 0; i < blockSize; i++) {
                    for (j = 0; j < blockSize; j++) {
                        // Fully unrolled for blockSize = 4
                        C[bi + i][bj + j] += A[bi + i][bk + 0] * B_trans[bj + j][bk + 0];
                        C[bi + i][bj + j] += A[bi + i][bk + 1] * B_trans[bj + j][bk + 1];
                        C[bi + i][bj + j] += A[bi + i][bk + 2] * B_trans[bj + j][bk + 2];
                        C[bi + i][bj + j] += A[bi + i][bk + 3] * B_trans[bj + j][bk + 3];
       
                    }
                }
}


#define RUN_COUNT 10

int main() {
    fetchMat();
    struct timespec start, end;

    transpose(&B[0][0], &B_trans[0][0], N, N);

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

    return 0;
}