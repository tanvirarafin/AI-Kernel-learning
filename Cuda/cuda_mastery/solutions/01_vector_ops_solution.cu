// ============================================================================
// SOLUTION: Exercise 1.4 - Vector Operations
// ============================================================================
// Complete working solutions for all vector operation exercises.
// Compile with: nvcc -o sol1.4 solutions/01_vector_ops_solution.cu
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
// SOLUTION 1: Vector Subtraction
// C[i] = A[i] - B[i]
// ============================================================================
__global__ void vectorSubtract(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}

// ============================================================================
// SOLUTION 2: Element-wise Multiplication
// C[i] = A[i] * B[i]
// ============================================================================
__global__ void vectorMultiply(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}

// ============================================================================
// SOLUTION 3: Vector Scaling
// C[i] = alpha * A[i]
// ============================================================================
__global__ void vectorScale(const float *A, float *C, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        C[i] = alpha * A[i];
    }
}

// ============================================================================
// SOLUTION 4: SAXPY Operation
// C[i] = alpha * A[i] + beta * B[i]
// ============================================================================
__global__ void saxpy(const float *A, const float *B, float *C, int n, 
                      float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        C[i] = alpha * A[i] + beta * B[i];
    }
}

// ============================================================================
// BONUS: Grid-Stride Loop Implementation
// Handles arrays larger than total thread count
// ============================================================================
__global__ void vectorAddGridStride(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Verification function
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

// CPU reference implementations
void cpuVectorSubtract(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] - B[i];
}

void cpuVectorMultiply(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] * B[i];
}

void cpuVectorScale(const float *A, float *C, int n, float alpha) {
    for (int i = 0; i < n; i++) C[i] = alpha * A[i];
}

void cpuSaxpy(const float *A, const float *B, float *C, int n, float alpha, float beta) {
    for (int i = 0; i < n; i++) C[i] = alpha * A[i] + beta * B[i];
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    printf("=== Vector Operations - SOLUTIONS ===\n\n");
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 2.0f;
        h_B[i] = i * 3.0f;
    }
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Test 1: Vector Subtraction
    printf("1. Vector Subtraction (C = A - B)\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    vectorSubtract<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    cpuVectorSubtract(h_A, h_B, h_C_cpu, n);
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Subtraction");
    printf("   Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 5.0f);
    
    // Test 2: Element-wise Multiplication
    printf("2. Element-wise Multiplication (C = A * B)\n");
    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    cpuVectorMultiply(h_A, h_B, h_C_cpu, n);
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Multiply");
    printf("   Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 30.0f);
    
    // Test 3: Vector Scaling
    printf("3. Vector Scaling (C = alpha * A), alpha = 2.5\n");
    float alpha = 2.5f;
    vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, n, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    cpuVectorScale(h_A, h_C_cpu, n, alpha);
    verifyResults(h_C_gpu, h_C_cpu, n, "Vector Scale");
    printf("   Sample: C[5] = %.1f (expected: %.1f)\n\n", h_C_gpu[5], 25.0f);
    
    // Test 4: SAXPY
    printf("4. SAXPY (C = alpha*A + beta*B), alpha=2.5, beta=0.5\n");
    float beta = 0.5f;
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    cpuSaxpy(h_A, h_B, h_C_cpu, n, alpha, beta);
    verifyResults(h_C_gpu, h_C_cpu, n, "SAXPY");
    printf("   Sample: C[5] = %.2f (expected: %.2f)\n\n", h_C_gpu[5], 27.5f);
    
    // Bonus: Grid-stride loop
    printf("5. BONUS: Grid-Stride Loop (handles large arrays)\n");
    int largeN = 100000;
    size_t largeSize = largeN * sizeof(float);
    float *h_A_large = (float *)malloc(largeSize);
    float *h_B_large = (float *)malloc(largeSize);
    float *h_C_large = (float *)malloc(largeSize);
    for (int i = 0; i < largeN; i++) {
        h_A_large[i] = 1.0f;
        h_B_large[i] = 2.0f;
    }
    float *d_A_large, *d_B_large, *d_C_large;
    CUDA_CHECK(cudaMalloc(&d_A_large, largeSize));
    CUDA_CHECK(cudaMalloc(&d_B_large, largeSize));
    CUDA_CHECK(cudaMalloc(&d_C_large, largeSize));
    CUDA_CHECK(cudaMemcpy(d_A_large, h_A_large, largeSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_large, h_B_large, largeSize, cudaMemcpyHostToDevice));
    
    // Use small configuration to force grid-stride
    vectorAddGridStride<<<10, 32>>>(d_A_large, d_B_large, d_C_large, largeN);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_large, d_C_large, largeSize, cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < largeN; i++) {
        if (h_C_large[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("   [%s] Grid-stride loop: All %d elements = 3.0\n\n",
           correct ? "PASS" : "FAIL", largeN);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_large); cudaFree(d_B_large); cudaFree(d_C_large);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    free(h_A_large); free(h_B_large); free(h_C_large);
    
    printf("=== All Solutions Complete! ===\n");
    
    return 0;
}
