#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

const int N = 1024;

float A[N][N], B[N][N], C[N][N];


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

    // Perform matrix multiplication 
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];


    end = chrono::high_resolution_clock::now();
    float time_taken = chrono::duration_cast<chrono::duration<float>>(end - start).count();
    cout << "Time taken by program is : " << fixed << time_taken << setprecision(6) << " sec" << endl;

    // Calculate GFLOPS
    float flops = 2.0 * N * N * N;
    float gflops = flops / time_taken / 1e9;
    cout << "GFLOPS: " << gflops << endl;

    return 0;
}