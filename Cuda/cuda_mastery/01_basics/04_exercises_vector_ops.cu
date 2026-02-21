// ============================================================================
// Exercise 1.4: Vector Operations - Practice Your CUDA Skills
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections below to implement vector operations.
//   Each exercise builds on the previous one.
//   Compile with: nvcc -o ex1.4 04_exercises_vector_ops.cu
//   Run with: ./ex1.4
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// EXERCISE 1: Vector Subtraction
// Implement: C[i] = A[i] - B[i]
// ============================================================================
// TODO: Complete the kernel below
__global__ void vectorSubtract(const float *A, const float *B, float *C, int n) {
    // TODO: Calculate global thread index
    int i = 0;  // FIXME: Replace with correct formula
    
    // TODO: Add bounds check
    
    // TODO: Implement subtraction: C[i] = A[i] - B[i]
}

// ============================================================================
// EXERCISE 2: Element-wise Multiplication
// Implement: C[i] = A[i] * B[i]
// ============================================================================
// TODO: Complete the kernel below
__global__ void vectorMultiply(const float *A, const float *B, float *C, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Add bounds check
    
    // TODO: Implement multiplication: C[i] = A[i] * B[i]
}

// ============================================================================
// EXERCISE 3: Vector Scaling
// Implement: C[i] = alpha * A[i]
// ============================================================================
// TODO: Complete the kernel below
__global__ void vectorScale(const float *A, float *C, int n, float alpha) {
    // TODO: Calculate global thread index
    
    // TODO: Add bounds check
    
    // TODO: Implement scaling: C[i] = alpha * A[i]
}

// ============================================================================
// EXERCISE 4: Combined Operation (Challenge!)
// Implement: C[i] = alpha * A[i] + beta * B[i]
// This is the SAXPY operation (Single-precision A*X Plus Y)
// ============================================================================
// TODO: Complete the kernel below
__global__ void saxpy(const float *A, const float *B, float *C, int n, 
                      float alpha, float beta) {
    // TODO: Calculate global thread index
    
    // TODO: Add bounds check
    
    // TODO: Implement: C[i] = alpha * A[i] + beta * B[i]
}

// ============================================================================
// VERIFICATION FUNCTION
// Checks if GPU results match CPU results
// ============================================================================
bool verifyResults(const float *gpu, const float *cpu, int n, const char *testName) {
    float tolerance = 1e-5;
    for (int i = 0; i < n; i++) {
        float diff = fabs(gpu[i] - cpu[i]);
        if (diff > tolerance) {
            printf("  [FAIL] %s: Mismatch at index %d: GPU=%.4f, CPU=%.4f\n", 
                   testName, i, gpu[i], cpu[i]);
            return false;
        }
    }
    printf("  [PASS] %s: All %d elements match!\n", testName, n);
    return true;
}

// ============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// Used to verify GPU results
// ============================================================================
void cpuVectorSubtract(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }
}

void cpuVectorMultiply(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

void cpuVectorScale(const float *A, float *C, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        C[i] = alpha * A[i];
    }
}

void cpuSaxpy(const float *A, const float *B, float *C, int n, float alpha, float beta) {
    for (int i = 0; i < n; i++) {
        C[i] = alpha * A[i] + beta * B[i];
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    printf("=== Vector Operations Exercises ===\n");
    printf("Vector size: %d elements\n\n", n);
    
    // Host arrays
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    
    // Initialize arrays with known values
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 2.0f;      // A = [0, 2, 4, 6, ...]
        h_B[i] = i * 3.0f;      // B = [0, 3, 6, 9, ...]
    }
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Execution config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // ========================================================================
    // TEST 1: Vector Subtraction
    // ========================================================================
    printf("Exercise 1: Vector Subtraction (C = A - B)\n");
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch GPU kernel
    vectorSubtract<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    cpuVectorSubtract(h_A, h_B, h_C_cpu, n);
    
    // Verify
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Subtraction");
    printf("  Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 5.0f);
    
    // ========================================================================
    // TEST 2: Element-wise Multiplication
    // ========================================================================
    printf("Exercise 2: Element-wise Multiplication (C = A * B)\n");
    
    // Launch GPU kernel
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    cpuVectorMultiply(h_A, h_B, h_C_cpu, n);
    
    // Verify
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Multiply");
    printf("  Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 30.0f);
    
    // ========================================================================
    // TEST 3: Vector Scaling
    // ========================================================================
    printf("Exercise 3: Vector Scaling (C = alpha * A)\n");
    float alpha = 2.5f;
    
    // Launch GPU kernel
    vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, n, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    cpuVectorScale(h_A, h_C_cpu, n, alpha);
    
    // Verify
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Scale");
    printf("  Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 25.0f);
    
    // ========================================================================
    // TEST 4: SAXPY (Challenge!)
    // ========================================================================
    printf("Exercise 4: SAXPY (C = alpha*A + beta*B)\n");
    float beta = 0.5f;
    
    // Launch GPU kernel
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    cpuSaxpy(h_A, h_B, h_C_cpu, n, alpha, beta);
    
    // Verify
    verifyResults(h_C_gpu, h_C_cpu, n, "SAXPY");
    printf("  Sample: C[5] = %.2f (expected: %.2f)\n\n", h_C_gpu[5], 27.5f);
    
    // ========================================================================
    // BONUS: Grid-Stride Loop Implementation
    // ========================================================================
    printf("=== BONUS: Grid-Stride Loop ===\n");
    printf("Modify vectorAdd to handle arrays larger than total threads!\n");
    printf("Hint: Each thread processes multiple elements spaced by grid size.\n\n");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    printf("=== Exercises Complete! ===\n");
    printf("Next: Try 05_exercises_thread_indexing.cu\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Global thread index formula:
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
// 2. Always check bounds before accessing arrays:
//    if (idx < n) { ... }
//
// 3. For grid-stride loop:
//    int stride = gridDim.x * blockDim.x;
//    for (int i = idx; i < n; i += stride) { ... }
//
// 4. Compile and test incrementally:
//    - Complete one exercise at a time
//    - Verify it works before moving on
// ============================================================================
