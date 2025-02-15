#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

const int N = 1024;

float A[N][N], B[N][N], B_trans[N][N], C[N][N];

void transpose(float* src, float* dst, const int rows, const int cols)
{
    for (int idx = 0; idx < rows * cols; idx++)
    {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

int main()
{
    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
            C[i][j] = 0.0;
        }

    chrono::high_resolution_clock::time_point start, end;
    // Transpose B into B_trans
    transpose(&B[0][0], &B_trans[0][0], N, N);

    start = chrono::high_resolution_clock::now();
    ios_base::sync_with_stdio(false);


    // Matrix multiplication using transposed B
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B_trans[j][k];
            C[i][j] = sum;
        }

    end = chrono::high_resolution_clock::now();

    float time_taken = chrono::duration_cast<chrono::duration<float>>(end - start).count();
    cout << "Time taken by program is : " << fixed << time_taken << setprecision(6) << " sec" << endl;

    //calculate the GFLOPS of the matrix multiplication

    float flops = 2.0 * N * N * N;
    float gflops = flops / time_taken / 1e9;
    cout << "GFLOPS: " << gflops << endl;

    return 0;
}
