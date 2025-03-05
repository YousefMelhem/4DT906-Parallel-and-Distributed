#include <arm_neon.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <chrono>

// Simple NxN matrix multiplication using ARM NEON
void matrix_multiply_neon(const float* A, const float* B, float* C, int N) {
    // For each row in A
    for (int i = 0; i < N; i++) {
        // For each column in B
        for (int j = 0; j < N; j += 4) {
            // Process 4 elements at a time if possible
            if (j + 3 < N) {
                float32x4_t acc = vmovq_n_f32(0.0f); // Initialize accumulator to zeros
                
                // Multiply and accumulate
                for (int k = 0; k < N; k++) {
                    // Load A[i][k]
                    float32x4_t a = vdupq_n_f32(A[i * N + k]);
                    
                    // Load B[k][j:j+3]
                    float32x4_t b = vld1q_f32(&B[k * N + j]);
                    
                    // Multiply and accumulate
                    acc = vmlaq_f32(acc, a, b);
                }
                
                // Store result to C[i][j:j+3]
                vst1q_f32(&C[i * N + j], acc);
            } else {
                // Handle remaining columns (if N is not a multiple of 4)
                for (; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }
}

// For comparison - standard NxN matrix multiplication without NEON
void matrix_multiply_standard(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Helper function to print a matrix
void print_matrix(const char* label, const float* matrix, int N) {
    std::cout << label << " (" << N << "×" << N << "):" << std::endl;
    
    // Only print the full matrix if it's small
    if (N <= 10) {
        for (int i = 0; i < N; i++) {
            std::cout << "  ";
            for (int j = 0; j < N; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * N + j];
            }
            std::cout << std::endl;
        }
    } else {
        // For large matrices, just print the top-left corner
        for (int i = 0; i < 4; i++) {
            std::cout << "  ";
            for (int j = 0; j < 4; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * N + j];
            }
            std::cout << " ...";
            std::cout << std::endl;
        }
        std::cout << "  ..." << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to initialize a matrix with sequential values
void initialize_matrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = static_cast<float>((i * N + j) % 10 + 1); // Use modulo to keep values reasonable
        }
    }
}

// Helper function to verify results match between the two implementations
bool verify_results(const float* C_neon, const float* C_standard, int N) {
    for (int i = 0; i < N * N; i++) {
        if (std::abs(C_neon[i] - C_standard[i]) > 0.001f) {
            std::cout << "Mismatch at index " << i << ": " 
                      << C_neon[i] << " vs " << C_standard[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Function to benchmark matrix multiplication
void benchmark(int N) {
    // Allocate matrices
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_neon = new float[N * N];
    float* C_standard = new float[N * N];
    
    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    std::memset(C_neon, 0, N * N * sizeof(float));
    std::memset(C_standard, 0, N * N * sizeof(float));
    
    // Print input matrices (only for small N)
    if (N <= 10) {
        print_matrix("Matrix A", A, N);
        print_matrix("Matrix B", B, N);
    } else {
        std::cout << "Matrix A and B initialized with size " << N << "×" << N << std::endl;
    }
    
    // Benchmark NEON multiplication
    auto start_neon = std::chrono::high_resolution_clock::now();
    matrix_multiply_neon(A, B, C_neon, N);
    auto end_neon = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_neon = end_neon - start_neon;
    
    // Benchmark standard multiplication
    auto start_std = std::chrono::high_resolution_clock::now();
    matrix_multiply_standard(A, B, C_standard, N);
    auto end_std = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_std = end_std - start_std;
    
    // Print results (only for small N)
    if (N <= 10) {
        print_matrix("Result (NEON)", C_neon, N);
        print_matrix("Result (Standard)", C_standard, N);
    }
    
    // Verify results
    bool results_match = verify_results(C_neon, C_standard, N);
    
    // Print timing results
    std::cout << "Matrix size: " << N << "×" << N << std::endl;
    std::cout << "NEON time: " << duration_neon.count() << " ms" << std::endl;
    std::cout << "Standard time: " << duration_std.count() << " ms" << std::endl;
    std::cout << "Speedup: " << duration_std.count() / duration_neon.count() << "x" << std::endl;
    std::cout << "Results match: " << (results_match ? "Yes ✓" : "No ✗") << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    
    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_neon;
    delete[] C_standard;
}

int main(int argc, char* argv[]) {
    std::cout << "NxN Matrix Multiplication with ARM NEON" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    #ifdef __ARM_NEON__
    std::cout << "ARM NEON is supported (__ARM_NEON__ is defined)" << std::endl;
    #else
    std::cout << "WARNING: ARM NEON may not be supported (__ARM_NEON__ is not defined)" << std::endl;
    #endif
    std::cout << std::endl;
    
    // Default matrix size
    int N = 4;
    
    // If size is provided as command line argument, use that
    if (argc > 1) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Invalid matrix size. Using default size of 4." << std::endl;
            N = 4;
        }
    }
    
    // Run benchmark for specified size
    benchmark(N);
    
    // Additional sizes for comparison if not too large
    if (N < 100 && argc == 1) {
        benchmark(8);
        benchmark(16);
        benchmark(32);
        if (N < 50) {
            benchmark(64);
            benchmark(128);
        }
    }
    
    return 0;
}