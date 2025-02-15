#include <cstdint>
#include <_time.h>
#include <arm_neon.h>
#include <iostream>

using namespace std;


const int N = 512;
const int BLOCK = 8;


// align fits the data to 32 bytes
float A[N*N] __attribute__((aligned(32)));
float B[N*N] __attribute__((aligned(32)));
float C[N*N] __attribute__((aligned(32)));



uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

// https://developer.arm.com/documentation/102467/0201/Example---matrix-multiplication
// https://developer.arm.com/documentation/102159/0400/Matrix-multiplication


void matrix_multiply_c(float32_t *A, float32_t *B, float32_t *C, uint32_t N, uint32_t k) {
    for (int i_idx=0; i_idx < N; i_idx++) {
        for (int j_idx=0; j_idx < N; j_idx++) {
            C[N*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx < k; k_idx++) {
                C[N*j_idx + i_idx] += A[N*k_idx + i_idx]*B[k*j_idx + k_idx];
            }
        }
    }
}


void main(){

}
