#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>
using namespace std;

// A[y][x] 
// A[x * N + y]

const int N = 1024;
const int blockSize=16; 

// will making these 1D arrays make it faster?  
float A[N*N], B[N*N], C[N*N], Cvals[N*N], B_trans[N*N];

void transpose1D(float* src, float* dst, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// check if the matrix is transposed correctly by printing
// used to debug with small matrices
void print_matrix(float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            cout << matrix[i * cols + j] << " ";
        cout << endl;
    }
    cout << endl;
}


int main() {
    // stop this idea from George
    // Initialize matrices by reading A and B from /tmp/matmul
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f == nullptr) {
        cout << "please pregenerate python /tmp/matmul file" << endl;
        return -1;
    }
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(Cvals, 1, sizeof(float)*N*N, f);
    fclose(f);
    
    struct timespec start, end;

    // transpose B
    transpose1D(&B[0], &B_trans[0], N, N);

    clock_gettime(CLOCK_MONOTONIC, &start);
    ios_base::sync_with_stdio(false);
    
    int bi, bj, bk, i, j, k;
    // Matrix multiplication using transposed B
    // ading bi in the private loses me 3gflops
    #pragma omp parallel for private(bj, bk, i, j, k) shared(A, B_trans, C)
    for(bi=0; bi<N; bi+=blockSize)
        for(bj=0; bj<N; bj+=blockSize)
            for(bk=0; bk<N; bk+=blockSize)

                for (i = 0; i < blockSize; i++){
                    for (j = 0; j < blockSize; j++) {
                        for (k = 0; k < blockSize; k++){
                            // C[bi + i][bj + j] += A[bi + i][bk + k] * B_trans[bk + k][bj + j];
                            C[(bi + i) * N + (bj + j)] += A[(bi + i) * N + (bk + k)] * B_trans[(bk + k) * N + (bj + j)];
                        }
                    }
                }
                
    clock_gettime(CLOCK_MONOTONIC, &end);

    // monitic time holds two values, tv_sec and tv_nsec, must add both
    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // gflops
    float gflops = (2.0 * N * N * N) / (1000000000.0 * time_taken);
    cout << "GFLOPS: " << fixed << gflops << setprecision(6) << endl;
    cout << "|" << endl;
    cout << "t: " << fixed << time_taken << setprecision(6)  << endl;
    cout << "|" << endl;
    cout << "N: " << N << endl;
    cout << "|" << endl;
    
    // validation
    // read actuall Cvals from /tmp/matmul and compare with C
    for (int y = 0; y < N; y++) {  // Fixed loop bounds
        for (int x = 0; x < N; x++) {  // Fixed loop bounds
            if (abs(C[y * N + x] - Cvals[y * N + x]) > 0.001) {
                cout << "mismatch at " << y << " " << x << endl;
                return -1;
            }
            // if (abs(C[y][x] - Cvals[y][x]) > 0.001) {  // Fixed if statement and variable name
            //     cout << "mismatch at " << y << " " << x << endl;
            //     return -1;
            // }
        }
    }
    return 0;
}