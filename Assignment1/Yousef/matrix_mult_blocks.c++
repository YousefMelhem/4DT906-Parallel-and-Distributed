#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

const int N = 1024;
const int TILE_SIZE = 64;  // Choose an optimal block size (typically 16, 32, or 64)

float A[N][N], B[N][N], C[N][N];

void matmul_blocked() {
    for(int jj = 0; jj < N; jj += TILE_SIZE) {
        for(int kk = 0; kk < N; kk += TILE_SIZE) {
            for(int i = 0; i < N; i++) {
                for(int j = jj; j < ((jj + TILE_SIZE) > N ? N : (jj + TILE_SIZE)); j++) {
                    float temp = 0;
                    for(int k = kk; k < ((kk + TILE_SIZE) > N ? N : (kk + TILE_SIZE)); k++) {
                        temp += A[i][k] * B[k][j];
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

    chrono::high_resolution_clock::time_point start, end;
    start = chrono::high_resolution_clock::now();
    ios_base::sync_with_stdio(false);

    // Perform matrix multiplication using blocking
    matmul_blocked();

    end = chrono::high_resolution_clock::now();
    float time_taken = chrono::duration_cast<chrono::duration<float>>(end - start).count();
    cout << "Time taken by program is : " << fixed << time_taken << setprecision(6) << " sec" << endl;

    // Calculate GFLOPS
    float flops = 2.0 * N * N * N;
    float gflops = flops / time_taken / 1e9;
    cout << "GFLOPS: " << gflops << endl;

    return 0;
}