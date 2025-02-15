#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

const int N = 512;

static float *A, *B, *C, *C_reference;

void transpose(float* src, float* dst, const int rows, const int cols) {
    for (int idx = 0; idx < rows * cols; idx++) {
        int i = idx / cols;
        int j = idx % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

void prefetch_block(float* addr) {
    // readme explains the below
    asm volatile("prfm pldl1keep, [%0]" : : "r" (addr));
}

// compute_reference_multiplication and validate_result <- THESE METHODS ARE AI GENERATED
// Add reference implementation for validation
void compute_reference_multiplication() {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_reference[i * N + j] = sum;
        }
    }
}

bool validate_result() {
    const float epsilon = 1e-3; // Tolerance for floating-point comparison
    int errors = 0;
    float max_diff = 0.0f;
    int max_diff_i = 0, max_diff_j = 0;
    
    #pragma omp parallel for collapse(2) reduction(+:errors) reduction(max:max_diff)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float diff = std::abs(C[i * N + j] - C_reference[i * N + j]);
            if(diff > epsilon) {
                #pragma omp atomic
                errors++;
                if(diff > max_diff) {
                    max_diff = diff;
                    max_diff_i = i;
                    max_diff_j = j;
                }
            }
        }
    }
    
    if(errors > 0) {
        cout << "Validation FAILED!" << endl;
        cout << "Number of errors: " << errors << endl;
        cout << "Maximum difference: " << max_diff << " at position [" << max_diff_i << "][" << max_diff_j << "]" << endl;
        cout << "Expected: " << C_reference[max_diff_i * N + max_diff_j] << endl;
        cout << "Got: " << C[max_diff_i * N + max_diff_j] << endl;
        return false;
    }
    
    cout << "Validation PASSED!" << endl;
    return true;
}


int main() {

    const int blockSize=32; 

    // dynamically alocatte memory
    // 1D matrix
    A = new float[N * N];
    B = new float[N * N];
    C = new float[N * N];
    C_reference = new float[N * N];

    // to further explain the above
    // instead of using new float[N][N]'
    // we isnetead use float[N]
    // this is because 1D arrays are stored in memory more effecientrly
    // and closer together (contiguous blocks of memory for he full matrix)
    // So basically accesing 1D memory is generally faster

    // to access A[y][x] in 1D matrix you would do A[x * N + y]
    // x indexing the row, and y the column

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
    //transpose(B, B_trans, N, N);

    clock_gettime(CLOCK_MONOTONIC, &start);
    ios_base::sync_with_stdio(false);

    int bi, bj, bk, i, j, k;

    // Mult
    
    // some important links for the ARM NEON instructions (vld1q_f32, vdupq_n_f32, vld1q_f32)
    //  https://developer.arm.com/architectures/instruction-sets/intrinsics/
    // should bi be in private or not?
    #pragma omp parallel for private(bj, bk, i, j, k) shared(A, B, C)
    for(bi = 0; bi < N; bi += blockSize) {
        for(bj = 0; bj < N; bj += blockSize) {
            prefetch_block(&A[(bi + blockSize) * N]);
            prefetch_block(&B[(bj + blockSize) * N]);
            for(bk = 0; bk < N; bk += blockSize) {
                for(i = 0; i < blockSize; i++) {
                    for(j = 0; j < blockSize; j += 4) {
                        float32x4_t sum = vld1q_f32(&C[(bi + i) * N + (bj + j)]);
                        for(k = 0; k < blockSize; k++) {
                            float32x4_t a = vdupq_n_f32(A[(bi + i) * N + (bk + k)]);
                            // Load 4 consecutive values from B at once
                            float32x4_t b = vld1q_f32(&B[(bk + k) * N + (bj + j)]);
                            sum = vfmaq_f32(sum, a, b);
                        }
                        vst1q_f32(&C[(bi + i) * N + (bj + j)], sum);
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    // monitic time holds two values, tv_sec and tv_nsec, must add both
    float time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    //cout << "Computing reference result for validation..." << endl;
    //compute_reference_multiplication();
    
    // Validate results
    //cout << "Validating results..." << endl;
    //bool is_valid = validate_result();
    

    // gflops
    float gflops = (2.0 * N * N * N) / (1000000000000.0 * time_taken);

    cout << "N: " << N << endl;
    cout << "BLOCKSIZE: " << blockSize << endl;
    cout << "Time taken by program is : " << fixed << time_taken << setprecision(6) << " sec" << endl;
    cout << "TFLOPS: " << fixed << gflops << setprecision(6) << endl;

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;


    return 0;
}
