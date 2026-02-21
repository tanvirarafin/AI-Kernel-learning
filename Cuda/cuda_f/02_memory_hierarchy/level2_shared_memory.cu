/**
 * Level 2: Shared Memory Introduction
 * 
 * GOAL: Learn to use shared memory for efficient data sharing within a block.
 * 
 * CONCEPTS:
 * - Shared memory declaration (__shared__)
 * - Cooperative thread loading
 * - Thread synchronization (__syncthreads())
 * - Reducing global memory accesses
 * 
 * EXERCISE:
 * 1. Load data from global to shared memory cooperatively
 * 2. Synchronize threads before using shared data
 * 3. Process data from shared memory
 * 
 * HINTS:
 * - Shared memory is declared with __shared__ keyword
 * - __syncthreads() ensures all threads reach the barrier
 * - Shared memory is much faster than global memory
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define BLOCK_SIZE 256

// TODO: Complete this kernel using shared memory
// Task: 
//   1. Load data cooperatively from global to shared memory
//   2. Synchronize threads
//   3. Process data from shared memory (multiply by 2)
//   4. Write results back to global memory
__global__ void sharedMemoryExample(float *input, float *output, int n) {
    // TODO: Declare shared memory array
    // Hint: __shared__ float sharedData[BLOCK_SIZE];
    /* YOUR CODE HERE */
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Load data from global to shared memory cooperatively
    // Each thread loads one element
    // Don't forget bounds checking!
    /* YOUR CODE HERE */
    
    // TODO: Synchronize all threads in the block
    // Hint: __syncthreads();
    /* YOUR CODE HERE */
    
    // TODO: Process data from shared memory
    // Multiply the shared memory value by 2 and store to output
    /* YOUR CODE HERE */
}

// Advanced: Multi-pass shared memory usage
// TODO: Complete this kernel that processes data in multiple passes
__global__ void sharedMemoryMultiPass(float *input, float *output, int n, int passes) {
    // TODO: Declare shared memory
    __shared__ float sharedData[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load initial data
    if (idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 0.0f;
    }
    
    // TODO: Process for 'passes' number of iterations
    // In each pass:
    //   1. Synchronize threads
    //   2. Each thread adds its left neighbor's value (if exists)
    //   3. Synchronize again before next pass
    // Hint: Be careful of boundary conditions!
    for (int p = 0; p < passes; p++) {
        /* YOUR CODE HERE - Add synchronization and neighbor addition */
    }
    
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    // Allocate and initialize
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch configuration
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch shared memory kernel
    printf("Launching shared memory kernel...\n");
    sharedMemoryExample<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy results back
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify (should be 2.0 for all elements)
    printFirstElements(h_output, 10, "Results (expected 2.00)");
    
    // Test multi-pass version
    printf("\nTesting multi-pass shared memory (3 passes)...\n");
    sharedMemoryMultiPass<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, 3);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 10, "Multi-pass results");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    printf("\nShared memory exercise completed!\n");
    printf("Key takeaway: Shared memory reduces global memory accesses!\n");
    
    return 0;
}
