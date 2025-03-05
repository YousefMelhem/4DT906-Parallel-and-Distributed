#include <arm_neon.h>
#include <iostream>
#include <iomanip>
#include <vector>

// Helper function to print a float32x4_t vector
void print_vector(const char* label, float32x4_t vec) {
    float result[4];
    vst1q_f32(result, vec);
    
    std::cout << std::left << std::setw(35) << label << "{ ";
    for (int i = 0; i < 4; i++) {
        std::cout << result[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << " }" << std::endl;
}

// Helper function to print a float value
void print_value(const char* label, float val) {
    std::cout << std::left << std::setw(35) << label << val << std::endl;
}

// Test FMA (Fused Multiply-Add) operations
void test_fma_operations() {
    std::cout << "\n=== FUSED MULTIPLY-ADD (FMA) OPERATIONS ===\n" << std::endl;
    
    // Standard FMA operation: a * b + c
    {
        float32x4_t a = { 2.0f, 3.0f, 4.0f, 5.0f };
        float32x4_t b = { 3.0f, 4.0f, 5.0f, 6.0f };
        float32x4_t c = { 1.0f, 1.0f, 1.0f, 1.0f };
        float32x4_t result = vfmaq_f32(c, a, b);
        
        print_vector("a:", a);
        print_vector("b:", b);
        print_vector("c:", c);
        print_vector("FMA (vfmaq_f32): a * b + c:", result);
        
        // For comparison, show the same using separate mul/add
        float32x4_t mul_result = vmulq_f32(a, b);
        float32x4_t add_result = vaddq_f32(mul_result, c);
        print_vector("Separate mul+add: (a * b) + c:", add_result);
    }
    
    // FMA with scalar: a * b + c (where b is scalar)
    {
        float32x4_t a = { 2.0f, 3.0f, 4.0f, 5.0f };
        float scalar_b = 3.0f;
        float32x4_t c = { 1.0f, 1.0f, 1.0f, 1.0f };
        float32x4_t result = vfmaq_n_f32(c, a, scalar_b);
        
        print_vector("a:", a);
        print_value("scalar b:", scalar_b);
        print_vector("c:", c);
        print_vector("FMA (vfmaq_n_f32): a * b + c:", result);
    }
    
    // Negated FMA: -(a * b) + c
    {
        float32x4_t a = { 2.0f, 3.0f, 4.0f, 5.0f };
        float32x4_t b = { 3.0f, 4.0f, 5.0f, 6.0f };
        float32x4_t c = { 10.0f, 20.0f, 30.0f, 40.0f };
        float32x4_t result = vfmsq_f32(c, a, b);
        
        print_vector("a:", a);
        print_vector("b:", b);
        print_vector("c:", c);
        print_vector("FNMA (vfmsq_f32): -(a * b) + c:", result);
        
        // For comparison, show the same using separate mul/sub
        float32x4_t mul_result = vmulq_f32(a, b);
        float32x4_t sub_result = vsubq_f32(c, mul_result);
        print_vector("Separate mul+sub: c - (a * b):", sub_result);
    }
    
    // Negated FMA with scalar: -(a * b) + c (where b is scalar)
    {
        float32x4_t a = { 2.0f, 3.0f, 4.0f, 5.0f };
        float scalar_b = 3.0f;
        float32x4_t c = { 10.0f, 20.0f, 30.0f, 40.0f };
        float32x4_t result = vfmsq_n_f32(c, a, scalar_b);
        
        print_vector("a:", a);
        print_value("scalar b:", scalar_b);
        print_vector("c:", c);
        print_vector("FNMA (vfmsq_n_f32): -(a * b) + c:", result);
    }
    
    // Precision comparison demonstration
    {
        float a_val = 1e-8f;
        float b_val = 1e8f;
        float c_val = 1.0f;
        
        // Create vectors with these values
        float32x4_t a = vmovq_n_f32(a_val);
        float32x4_t b = vmovq_n_f32(b_val);
        float32x4_t c = vmovq_n_f32(c_val);
        
        // Calculate using FMA
        float32x4_t fma_result = vfmaq_f32(c, a, b);
        
        // Calculate using separate mul+add
        float32x4_t mul_result = vmulq_f32(a, b);
        float32x4_t sep_result = vaddq_f32(mul_result, c);
        
        // Calculate the exact result for comparison (using doubles for higher precision)
        double exact = (double)a_val * (double)b_val + (double)c_val;
        
        print_value("a (small value):", a_val);
        print_value("b (large value):", b_val);
        print_value("c:", c_val);
        print_vector("Using FMA (vfmaq_f32):", fma_result);
        print_vector("Using separate mul+add:", sep_result);
        print_value("Exact result (double precision):", (float)exact);
        
        std::cout << "Note: In some cases, FMA may provide better precision" << std::endl;
        std::cout << "due to avoiding intermediate rounding errors." << std::endl;
    }
    
    // Practical example: Dot product calculation using FMA
    {
        float32x4_t a = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t b = { 5.0f, 6.0f, 7.0f, 8.0f };
        float32x4_t accum = vmovq_n_f32(0.0f);
        
        // Accumulate dot product using FMA
        accum = vfmaq_f32(accum, a, b);
        
        print_vector("Vector a:", a);
        print_vector("Vector b:", b);
        print_vector("a * b (partial dot product):", accum);
        
        // Sum the lanes to get the complete dot product
        float32x2_t sum2 = vadd_f32(vget_low_f32(accum), vget_high_f32(accum));
        float32x2_t sum1 = vpadd_f32(sum2, sum2);
        float dot_product = vget_lane_f32(sum1, 0);
        
        print_value("Complete dot product:", dot_product);
        std::cout << "This is a common pattern in optimized linear algebra." << std::endl;
    }
}

// Demonstrate practical use case: Improved dot product using FMA
float dot_product_fma_neon(const float* a, const float* b, size_t length) {
    float result = 0.0f;
    size_t i = 0;
    
    // Process 4 elements at a time using NEON with FMA
    float32x4_t sum = vmovq_n_f32(0.0f);
    for (; i + 3 < length; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);  // sum += va * vb (using FMA)
    }
    
    // Sum up the four partial sums
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float32x2_t sum1 = vpadd_f32(sum2, sum2);
    result = vget_lane_f32(sum1, 0);
    
    // Process remaining elements
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}


// Test float arithmetic operations
void test_float_arithmetic() {
    std::cout << "\n=== FLOAT ARITHMETIC OPERATIONS ===\n" << std::endl;
    
    // Addition
    {
        float32x4_t v1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v2 = { 1.0f, 1.0f, 1.0f, 1.0f };
        float32x4_t sum = vaddq_f32(v1, v2);
        
        print_vector("Original v1:", v1);
        print_vector("Original v2:", v2);
        print_vector("Addition (vaddq_f32):", sum);
    }
    
    // Multiplication
    {
        float32x4_t v1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v2 = { 1.0f, 1.0f, 1.0f, 1.0f };
        float32x4_t prod = vmulq_f32(v1, v2);
        
        print_vector("Original v1:", v1);
        print_vector("Original v2:", v2);
        print_vector("Multiplication (vmulq_f32):", prod);
    }
    
    // Multiply and accumulate
    {
        float32x4_t v1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v2 = { 2.0f, 2.0f, 2.0f, 2.0f };
        float32x4_t v3 = { 3.0f, 3.0f, 3.0f, 3.0f };
        float32x4_t acc = vmlaq_f32(v3, v1, v2);
        
        print_vector("Original v1:", v1);
        print_vector("Original v2:", v2);
        print_vector("Original v3:", v3);
        print_vector("Multiply-Accumulate (vmlaq_f32):", acc);
    }
    
    // Multiply by scalar
    {
        float32x4_t v = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32_t s = 3.0f;
        float32x4_t prod = vmulq_n_f32(v, s);
        
        print_vector("Original v:", v);
        print_value("Scalar value:", s);
        print_vector("Scalar Multiplication (vmulq_n_f32):", prod);
    }
    
    // Multiply by scalar and accumulate
    {
        float32x4_t v1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v2 = { 1.0f, 1.0f, 1.0f, 1.0f };
        float32_t s = 3.0f;
        float32x4_t acc = vmlaq_n_f32(v1, v2, s);
        
        print_vector("Original v1:", v1);
        print_vector("Original v2:", v2);
        print_value("Scalar value:", s);
        print_vector("Scalar Multiply-Accumulate (vmlaq_n_f32):", acc);
    }
    
    // Invert (reciprocal)
    {
        float32x4_t v = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t reciprocal = vrecpeq_f32(v);
        
        print_vector("Original v:", v);
        print_vector("Reciprocal (vrecpeq_f32):", reciprocal);
    }
    
    // Invert with Newton-Raphson refinement
    {
        float32x4_t v = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t reciprocal = vrecpeq_f32(v);
        float32x4_t inverse = vmulq_f32(vrecpsq_f32(v, reciprocal), reciprocal);
        
        print_vector("Original v:", v);
        print_vector("Refined Reciprocal:", inverse);
    }
}

// Test load operations
void test_load_operations() {
    std::cout << "\n=== LOAD OPERATIONS ===\n" << std::endl;
    
    // Load vector
    {
        float values[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float32x4_t v = vld1q_f32(values);
        
        std::cout << "Original array: { 1.0, 2.0, 3.0, 4.0, 5.0 }" << std::endl;
        print_vector("Loaded Vector (vld1q_f32):", v);
    }
    
    // Load same value for all lanes
    {
        float val = 3.0f;
        float32x4_t v = vld1q_dup_f32(&val);
        
        print_value("Value to duplicate:", val);
        print_vector("Duplicated Vector (vld1q_dup_f32):", v);
    }
    
    // Set all lanes to a hardcoded value
    {
        float32x4_t v = vmovq_n_f32(1.5f);
        
        print_value("Value to set:", 1.5f);
        print_vector("Set Vector (vmovq_n_f32):", v);
    }
}

// Test store operations
void test_store_operations() {
    std::cout << "\n=== STORE OPERATIONS ===\n" << std::endl;
    
    // Store vector
    {
        float32x4_t v = { 1.0f, 2.0f, 3.0f, 4.0f };
        float values[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        
        print_vector("Original vector:", v);
        std::cout << "Original array: { 0.0, 0.0, 0.0, 0.0, 0.0 }" << std::endl;
        
        vst1q_f32(values, v);
        
        std::cout << "After vst1q_f32 array: { ";
        for (int i = 0; i < 5; i++) {
            std::cout << values[i];
            if (i < 4) std::cout << ", ";
        }
        std::cout << " }" << std::endl;
    }
    
    // Store lane of array of vectors
    {
        float32x4_t v0 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v1 = { 5.0f, 6.0f, 7.0f, 8.0f };
        float32x4_t v2 = { 9.0f, 10.0f, 11.0f, 12.0f };
        float32x4_t v3 = { 13.0f, 14.0f, 15.0f, 16.0f };
        float32x4x4_t u = { v0, v1, v2, v3 };
        float buff[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        
        print_vector("v0:", v0);
        print_vector("v1:", v1);
        print_vector("v2:", v2);
        print_vector("v3:", v3);
        std::cout << "Original buff: { 0.0, 0.0, 0.0, 0.0 }" << std::endl;
        
        vst4q_lane_f32(buff, u, 0);
        
        std::cout << "After vst4q_lane_f32 buff: { ";
        for (int i = 0; i < 4; i++) {
            std::cout << buff[i];
            if (i < 3) std::cout << ", ";
        }
        std::cout << " }" << std::endl;
    }
}

// Test array operations
void test_array_operations() {
    std::cout << "\n=== ARRAY OPERATIONS ===\n" << std::endl;
    
    // Access to values
    {
        float32x4_t v0 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v1 = { 5.0f, 6.0f, 7.0f, 8.0f };
        float32x4_t v2 = { 9.0f, 10.0f, 11.0f, 12.0f };
        float32x4_t v3 = { 13.0f, 14.0f, 15.0f, 16.0f };
        float32x4x4_t ary = { v0, v1, v2, v3 };
        
        print_vector("ary.val[0]:", ary.val[0]);
        print_vector("ary.val[1]:", ary.val[1]);
        print_vector("ary.val[2]:", ary.val[2]);
        print_vector("ary.val[3]:", ary.val[3]);
    }
}

// Test min and max operations
void test_min_max_operations() {
    std::cout << "\n=== MIN AND MAX OPERATIONS ===\n" << std::endl;
    
    // Max of two vectors
    {
        float32x4_t v0 = { 5.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v1 = { 1.0f, 6.0f, 7.0f, 8.0f };
        float32x4_t v2 = vmaxq_f32(v0, v1);
        
        print_vector("v0:", v0);
        print_vector("v1:", v1);
        print_vector("Max (vmaxq_f32):", v2);
    }
    
    // Max of vector elements
    {
        float32x4_t v0 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x2_t maxOfHalfs = vpmax_f32(vget_low_f32(v0), vget_high_f32(v0));
        float32x2_t maxOfMaxOfHalfs = vpmax_f32(maxOfHalfs, maxOfHalfs);
        float maxValue = vget_lane_f32(maxOfMaxOfHalfs, 0);
        
        print_vector("Original vector:", v0);
        print_value("Max value (using vpmax_f32):", maxValue);
    }
    
    // Min of two vectors
    {
        float32x4_t v0 = { 5.0f, 2.0f, 3.0f, 4.0f };
        float32x4_t v1 = { 1.0f, 6.0f, 7.0f, 8.0f };
        float32x4_t v2 = vminq_f32(v0, v1);
        
        print_vector("v0:", v0);
        print_vector("v1:", v1);
        print_vector("Min (vminq_f32):", v2);
    }
    
    // Min of vector elements
    {
        float32x4_t v0 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float32x2_t minOfHalfs = vpmin_f32(vget_low_f32(v0), vget_high_f32(v0));
        float32x2_t minOfMinOfHalfs = vpmin_f32(minOfHalfs, minOfHalfs);
        float minValue = vget_lane_f32(minOfMinOfHalfs, 0);
        
        print_vector("Original vector:", v0);
        print_value("Min value (using vpmin_f32):", minValue);
    }
}

// Test conditional operations
void test_conditional_operations() {
    std::cout << "\n=== CONDITIONAL OPERATIONS ===\n" << std::endl;
    
    // Ternary operator
    {
        float32x4_t v1 = { 1.0f, 0.0f, 1.0f, 0.0f };
        float32x4_t v2 = { 0.0f, 1.0f, 1.0f, 0.0f };
        uint32x4_t mask = vcltq_f32(v1, v2);  // v1 < v2
        float32x4_t ones = vmovq_n_f32(10.0f);
        float32x4_t twos = vmovq_n_f32(20.0f);
        float32x4_t v3 = vbslq_f32(mask, ones, twos);
        
        print_vector("v1:", v1);
        print_vector("v2:", v2);
        
        // Print the mask (this is a bit-mask)
        uint32_t mask_values[4];
        vst1q_u32(mask_values, mask);
        std::cout << std::left << std::setw(35) << "Mask (vcltq_f32 - v1 < v2):" << "{ ";
        for (int i = 0; i < 4; i++) {
            std::cout << (mask_values[i] ? "true" : "false");
            if (i < 3) std::cout << ", ";
        }
        std::cout << " }" << std::endl;
        
        print_vector("If true value:", ones);
        print_vector("If false value:", twos);
        print_vector("Conditional result (vbslq_f32):", v3);
    }
}

// Demonstrate practical use case: vector dot product
float dot_product_neon(const float* a, const float* b, size_t length) {
    float result = 0.0f;
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    float32x4_t sum = vmovq_n_f32(0.0f);
    for (; i + 3 < length; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);  // sum += va * vb
    }
    
    // Sum up the four partial sums
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float32x2_t sum1 = vpadd_f32(sum2, sum2);
    result = vget_lane_f32(sum1, 0);
    
    // Process remaining elements
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Demonstrate a practical use case
void test_practical_usage() {
    std::cout << "\n=== PRACTICAL USAGE: DOT PRODUCT ===\n" << std::endl;
    
    const size_t length = 1000;
    std::vector<float> a(length, 1.0f);
    std::vector<float> b(length, 2.0f);
    
    // Calculate the dot product using NEON
    float neon_result = dot_product_neon(a.data(), b.data(), length);
    
    // Calculate the dot product using a standard loop for verification
    float standard_result = 0.0f;
    for (size_t i = 0; i < length; i++) {
        standard_result += a[i] * b[i];
    }
    
    std::cout << "Vector length: " << length << std::endl;
    std::cout << "Dot product (NEON): " << neon_result << std::endl;
    std::cout << "Dot product (Standard): " << standard_result << std::endl;
    
    if (std::abs(neon_result - standard_result) < 0.001f) {
        std::cout << "Results match! ✓" << std::endl;
    } else {
        std::cout << "Results do not match! ✗" << std::endl;
    }
}


int main() {
    std::cout << "=== ARM NEON INTRINSICS TEST SUITE ===" << std::endl;
    std::cout << "Testing ARM NEON intrinsics on " 
              << (sizeof(void*) == 8 ? "64-bit" : "32-bit") << " platform" << std::endl;
    
    #ifdef __ARM_NEON__
    std::cout << "ARM NEON is supported at build time (__ARM_NEON__ is defined)" << std::endl;
    #else
    std::cout << "WARNING: ARM NEON may not be supported at build time (__ARM_NEON__ is not defined)" << std::endl;
    #endif
    
    // Run all tests
    test_float_arithmetic();
    test_load_operations();
    test_store_operations();
    test_array_operations();
    test_min_max_operations();
    test_conditional_operations();
    
    // Run our new FMA test
    test_fma_operations();
    
    test_practical_usage();
    
    // Test improved dot product using FMA
    const size_t length = 1000;
    std::vector<float> a(length, 1.0f);
    std::vector<float> b(length, 2.0f);
    float fma_result = dot_product_fma_neon(a.data(), b.data(), length);
    std::cout << "\nDot product using FMA: " << fma_result << std::endl;
    
    return 0;
}