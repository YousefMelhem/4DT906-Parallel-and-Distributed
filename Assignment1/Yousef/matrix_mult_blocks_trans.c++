#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

const int N = 1024*2;
const int TILE_SIZE = 64;  // Choose an optimal block size (typically 16, 32, or 64)

float A[N][N], B[N][N], B_T[N][N], C[N][N];

// Function to transpose matrix B
void transpose_matrix(float src[N][N], float dest[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dest[j][i] = src[i][j];
        }
    }
}

void matmul_blocked() {
    for(int jj = 0; jj < N; jj += TILE_SIZE) {
        for(int kk = 0; kk < N; kk += TILE_SIZE) {
            for(int i = 0; i < N; i++) {
                for(int j = jj; j < ((jj + TILE_SIZE) > N ? N : (jj + TILE_SIZE)); j++) {
                    float temp = 0;
                    for(int k = kk; k < ((kk + TILE_SIZE) > N ? N : (kk + TILE_SIZE)); k++) {
                        temp += A[i][k] * B_T[j][k];  // Use transposed matrix B_T
                    }
                    C[i][j] += temp;
                }
            }
        }
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

    // Transpose matrix B
    transpose_matrix(B, B_T);

    auto start = chrono::high_resolution_clock::now();
    ios_base::sync_with_stdio(false);

    // Perform matrix multiplication using blocking
    matmul_blocked();

    auto stop = chrono::high_resolution_clock::now();
    auto time_taken = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << "Time taken by program is : " << fixed << time_taken.count() << setprecision(6) << " sec" << endl;

    // Calculate GFLOPS
    float flop = 2.0 * N * N * N;
    float gflops = flop / time_taken.count() / 1e9;
    cout << "GFLOPS: " << gflops << endl;

    return 0;
}