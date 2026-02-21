/**
 * Level 1: Global Memory Basics
 * 
 * GOAL: Complete the kernel to copy and transform data in global memory.
 * 
 * CONCEPTS:
 * - Global memory allocation (cudaMalloc)
 * - Memory transfers (cudaMemcpy)
 * - Thread indexing for global memory access
 * - Basic pointer arithmetic
 * 
 * EXERCISE:
 * 1. Complete the kernel to copy input to output with a scale factor
 * 2. Implement proper global thread indexing
 * 3. Add bounds checking to prevent out-of-bounds access
 * 
 * HINTS:
 * - Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 * - Always check bounds before accessing memory
 * - Use the scale_factor parameter in your computation
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define SCALE_FACTOR 2.5f

// TODO: Complete this kernel
// Task: Copy input to output, multiplying each element by scale_factor
// Requirements:
//   1. Calculate global thread ID
//   2. Add bounds check (thread ID < n)
//   3. Perform: output[i] = input[i] * scale_factor
__global__ void globalMemoryTransform(float *input, float *output, int n, float scale_factor) {
    // TODO: Calculate global thread index
    int idx = /* YOUR CODE HERE */;
    
    // TODO: Add bounds check and perform the transformation
    /* YOUR CODE HERE */
}

// Alternative kernel with strided access pattern
// TODO: Complete this kernel to handle cases where N > total threads
__global__ void globalMemoryTransformStrided(float *input, float *output, int n, float scale_factor) {
    // TODO: Calculate global thread index
    int idx = /* YOUR CODE HERE */;
    
    // TODO: Calculate total grid stride
    int stride = /* YOUR CODE HERE */;
    
    // TODO: Implement strided loop to handle all elements
    // Hint: for (int i = idx; i < n; i += stride) { ... }
    /* YOUR CODE HERE */
}

void verifyResults(float *output, int n, float expected) {
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (output[i] != expected) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, output[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("✓ Verification passed!\n");
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    // TODO: Allocate device memory using cudaMalloc
    // Hint: cudaMalloc(&pointer, size_in_bytes)
    /* YOUR CODE HERE */
    
    // TODO: Copy input data to device
    // Hint: cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice)
    /* YOUR CODE HERE */
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    globalMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, SCALE_FACTOR);
    
    // TODO: Check for kernel launch errors
    // Hint: cudaGetLastError()
    /* YOUR CODE HERE */
    
    // TODO: Copy results back to host
    // Hint: cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost)
    /* YOUR CODE HERE */
    
    // Verify results (first 10 elements)
    printf("Checking first 10 results (expected: %f):\n", SCALE_FACTOR);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");
    
    // TODO: Free device memory
    // Hint: cudaFree(pointer)
    /* YOUR CODE HERE */
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    printf("Global memory exercise completed!\n");
    return 0;
}
