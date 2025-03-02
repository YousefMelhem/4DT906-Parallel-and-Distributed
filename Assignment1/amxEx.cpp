#include "amx.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring> // for memset

// Proper GEMM implementation with AMX
void gemm_amx(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    // Note: This implementation doesn't use AMX instructions directly for matrix multiplication,
    // but rather uses them as accelerated operations within a standard GEMM algorithm.
    
    // Enable AMX
    AMX_SET();
    
    // Clear result matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i*ldc + j] = 0.0f;
        }
    }
    
    // Basic matrix multiplication algorithm (ijk form)
    // C = A * B
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            const float a_val = A[i*lda + k];
            
            // Use AMX to multiply a row of A by a row of B
            // We'll set up vectors for this dot product
            float a_vec[16] = {0}; // AMX register size (16 floats)
            float b_vec[16] = {0};
            float z_vec[16] = {0};
            
            // Fill with data - broadcast a_val to all elements of a_vec
            for (int t = 0; t < 16; t++) {
                a_vec[t] = a_val;
            }
            
            // We'll process B in chunks of 16 elements at a time
            for (int j = 0; j < N; j += 16) {
                // Fill b_vec with a chunk of row k from matrix B
                int remaining = (N - j < 16) ? (N - j) : 16;
                for (int t = 0; t < remaining; t++) {
                    b_vec[t] = B[k*ldb + (j + t)];
                    // Get current value from C for accumulation
                    z_vec[t] = C[i*ldc + (j + t)];
                }
                
                // Load into AMX registers
                AMX_LDX(a_vec);
                AMX_LDY(b_vec);
                AMX_LDZ(z_vec);
                
                // Perform FMA: Z += X * Y
                AMX_FMA32(0);
                
                // Store result back to C
                AMX_STZ(z_vec);
                for (int t = 0; t < remaining; t++) {
                    C[i*ldc + (j + t)] = z_vec[t];
                }
            }
        }
    }
    
    // Disable AMX
    AMX_CLR();
}

// Simple test for the GEMM function
void test_gemm() {
    std::cout << "=== TESTING AMX GEMM IMPLEMENTATION ===\n" << std::endl;
    
    const int M = 4;
    const int N = 4;
    const int K = 4;
    
    // Test matrices
    float A[M*K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float B[K*N] = {
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2
    };
    
    float C[M*N] = {0};
    
    // Display input matrices
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << std::setw(4) << A[i*K + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMatrix B:" << std::endl;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << B[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Perform matrix multiplication using our AMX GEMM function
    gemm_amx(M, N, K, A, K, B, N, C, N);
    
    // Display result
    std::cout << "\nResult Matrix C (A*B):" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << C[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compute expected result
    float expected[M*N] = {0};
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                expected[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
    
    // Display expected result
    std::cout << "\nExpected Result:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << expected[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Verify results
    bool match = true;
    for (int i = 0; i < M*N; i++) {
        if (std::abs(C[i] - expected[i]) > 0.001f) {
            match = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "\n✓ GEMM result matches expected result" << std::endl;
    } else {
        std::cout << "\n✗ GEMM result does NOT match expected result" << std::endl;
        
        // Show detailed differences for debugging
        std::cout << "\nDetailed differences:" << std::endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float diff = C[i*N + j] - expected[i*N + j];
                if (std::abs(diff) > 0.001f) {
                    std::cout << "C[" << i << "][" << j << "] = " << C[i*N + j] 
                              << ", Expected: " << expected[i*N + j]
                              << ", Diff: " << diff << std::endl;
                }
            }
        }
    }
}

// Original GEMM function from NEON for reference
void gemm_ref(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] = sum;
        }
    }
}

// Test with diagonal matrix to isolate AMX functionality
void test_diagonal_matrix() {
    std::cout << "\n=== TESTING WITH DIAGONAL MATRIX ===\n" << std::endl;
    
    const int N = 4;
    
    // Create identity matrix
    float I[N*N] = {0};
    for (int i = 0; i < N; i++) {
        I[i*N + i] = 1.0f;
    }
    
    // Create diagonal matrix with value 2
    float D[N*N] = {0};
    for (int i = 0; i < N; i++) {
        D[i*N + i] = 2.0f;
    }
    
    // Test matrix
    float A[N*N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float C[N*N] = {0};
    
    // Test A * D (should multiply each element by 2)
    gemm_amx(N, N, N, A, N, D, N, C, N);
    
    // Display matrices
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << A[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nDiagonal Matrix D:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << D[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nResult Matrix C (A*D):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << C[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Compute expected result
    float expected[N*N] = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                expected[i*N + j] += A[i*N + k] * D[k*N + j];
            }
        }
    }
    
    std::cout << "\nExpected Result:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(4) << expected[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {a
    std::cout << "=== PROPER AMX MATRIX MULTIPLICATION ===\n" << std::endl;
    
    // Test the GEMM implementation
    test_gemm();
    
    // Test with diagonal matrix
    test_diagonal_matrix();
    
    return 0;
}