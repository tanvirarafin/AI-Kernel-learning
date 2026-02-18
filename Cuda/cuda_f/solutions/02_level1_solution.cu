/*
 * Solution: Shared Memory Level 1 - Basics
 * 
 * This is a REFERENCE SOLUTION. Try the exercise first!
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// SOLUTION: Basic shared memory load and store
__global__ void basicSharedLoad(float *input, float *output, int n) {
    // Declare shared memory
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    // Synchronize before reading shared memory
    __syncthreads();
    
    // Process and store
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f;
    }
}

// SOLUTION: Shared memory reverse
__global__ void sharedReverse(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    
    __syncthreads();
    
    // Read in reverse order within block
    if (idx < n) {
        int reverseIdx = BLOCK_SIZE - 1 - tid;
        // Handle boundary case
        int globalReverseIdx = blockIdx.x * BLOCK_SIZE + reverseIdx;
        if (globalReverseIdx < n) {
            output[idx] = sdata[reverseIdx];
        } else {
            output[idx] = sdata[tid];  // Keep original if out of bounds
        }
    }
}

// SOLUTION: Shared memory swap
__global__ void sharedSwap(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    
    __syncthreads();
    
    // Swap with adjacent element
    if (idx < n) {
        int swapIdx = (tid % 2 == 0) ? tid + 1 : tid - 1;
        
        // Handle boundary - last element in odd-sized block
        if (swapIdx < BLOCK_SIZE) {
            int globalSwapIdx = blockIdx.x * BLOCK_SIZE + swapIdx;
            if (globalSwapIdx < n) {
                output[idx] = sdata[swapIdx];
            } else {
                output[idx] = sdata[tid];  // Keep original
            }
        } else {
            output[idx] = sdata[tid];  // Keep original
        }
    }
}

// SOLUTION: Multi-stage shared memory processing
__global__ void multiStageShared(float *input, float *output, int n, float multiplier) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Load
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();
    
    // Stage 2: Multiply
    if (idx < n) {
        sdata[tid] = sdata[tid] * multiplier;
    }
    __syncthreads();
    
    // Stage 3: Add thread index
    if (idx < n) {
        sdata[tid] = sdata[tid] + tid;
    }
    __syncthreads();
    
    // Stage 4: Store
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

int main() {
    printf("=== Solution: Shared Memory Level 1 ===\n\n");
    
    const int N = 2048;
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));
    
    initArray(h_in, N);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test 1: Basic shared memory load
    printf("Test 1: Basic shared memory load\n");
    basicSharedLoad<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) h_expected[i] = h_in[i] * 2.0f;
    if (verifyArray(h_out, h_expected, N)) printf("  ✓ PASSED\n");
    else printf("  ✗ FAILED\n");
    
    // Test 2: Shared memory reverse
    printf("\nTest 2: Shared memory reverse\n");
    sharedReverse<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.2f ", h_out[i]);
    printf("\n");
    
    // Test 3: Shared memory swap
    printf("\nTest 3: Shared memory swap\n");
    sharedSwap<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.2f ", h_out[i]);
    printf("\n");
    
    // Test 4: Multi-stage processing
    printf("\nTest 4: Multi-stage shared processing\n");
    const float MULT = 3.0f;
    multiStageShared<<<gridSize, blockSize>>>(d_in, d_out, N, MULT);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.2f ", h_out[i]);
    printf("\n");
    
    // Cleanup
    free(h_in); free(h_out); free(h_expected);
    cudaFree(d_in); cudaFree(d_out);
    
    printf("\n=== Key Points ===\n");
    printf("1. __shared__ declares block-shared memory\n");
    printf("2. __syncthreads() synchronizes all threads in block\n");
    printf("3. Shared memory is much faster than global memory\n");
    printf("4. All threads must reach __syncthreads() (no conditional sync)\n");
    
    return 0;
}
