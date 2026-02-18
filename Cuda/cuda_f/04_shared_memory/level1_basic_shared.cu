/*
 * Shared Memory Level 1: Basics
 * 
 * EXERCISE: Learn shared memory fundamentals including declaration,
 * data loading, and synchronization.
 * 
 * SKILLS PRACTICED:
 * - Shared memory declaration
 * - Cooperative data loading
 * - Thread synchronization
 * - Block-level communication
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// ============================================================================
// KERNEL 1: Basic Shared Memory Load and Store
 * Load data to shared memory, synchronize, then process
// ============================================================================
__global__ void basicSharedLoad(float *input, float *output, int n) {
    // TODO: Declare shared memory array of size BLOCK_SIZE
    // __shared__ float sdata[/* YOUR CODE HERE */];
    /* YOUR DECLARATION HERE */;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Load data from global to shared memory
    // Each thread loads one element
    if (idx < n) {
        // sdata[tid] = /* YOUR CODE HERE */;
    } else {
        sdata[tid] = 0.0f;
    }
    
    // TODO: Synchronize threads before reading shared memory
    // __syncthreads();
    
    // Process: multiply by 2 after synchronization
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f;
    }
}

// ============================================================================
// KERNEL 2: Shared Memory Reverse
 * Load data, reverse within block using shared memory
// ============================================================================
__global__ void sharedReverse(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Load data into shared memory
    if (idx < n) {
        sdata[tid] = /* YOUR CODE HERE */;
    }
    
    __syncthreads();
    
    // TODO: Read from shared memory in reverse order within block
    // Thread 0 reads from sdata[BLOCK_SIZE-1], thread 1 from sdata[BLOCK_SIZE-2], etc.
    if (idx < n) {
        int reverseIdx = BLOCK_SIZE - 1 - tid;
        // TODO: Handle boundary case where block is not full
        if (/* YOUR CODE HERE */) {
            output[idx] = sdata[reverseIdx];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// ============================================================================
// KERNEL 3: Shared Memory Swap
 * Swap adjacent elements using shared memory
// ============================================================================
__global__ void sharedSwap(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    
    __syncthreads();
    
    // TODO: Swap with adjacent element
    // Even threads swap with next odd thread
    // Odd threads swap with previous even thread
    if (idx < n) {
        int swapIdx = (tid % 2 == 0) ? tid + 1 : tid - 1;
        
        // TODO: Handle boundary - last element in odd-sized block
        if (/* YOUR CODE HERE */) {
            output[idx] = sdata[swapIdx];
        } else {
            output[idx] = sdata[tid];  // Keep original if no swap partner
        }
    }
}

// ============================================================================
// KERNEL 4: Multi-Stage Shared Memory Processing
 * Load -> Transform -> Sync -> Transform -> Store
// ============================================================================
__global__ void multiStageShared(float *input, float *output, int n, float multiplier) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Load
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    
    __syncthreads();
    
    // Stage 2: First transformation (multiply)
    if (idx < n) {
        sdata[tid] = sdata[tid] * multiplier;
    }
    
    __syncthreads();
    
    // Stage 3: Second transformation (add thread index)
    if (idx < n) {
        // TODO: Add thread index to the value
        sdata[tid] = /* YOUR CODE HERE */;
    }
    
    __syncthreads();
    
    // Stage 4: Store result
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

bool verifyReverse(float *result, float *input, int n, int blockSize) {
    for (int block = 0; block * blockSize < n; block++) {
        int blockStart = block * blockSize;
        int blockEnd = min(blockStart + blockSize, n);
        
        for (int i = blockStart; i < blockEnd; i++) {
            int reverseIdx = blockStart + (blockEnd - 1 - i);
            if (fabsf(result[i] - input[reverseIdx]) > 1e-5f) return false;
        }
    }
    return true;
}

bool verifySwap(float *result, float *input, int n, int blockSize) {
    for (int block = 0; block * blockSize < n; block++) {
        int blockStart = block * blockSize;
        int blockEnd = min(blockStart + blockSize, n);
        
        for (int i = blockStart; i < blockEnd; i++) {
            int localIdx = i - blockStart;
            int swapLocalIdx = (localIdx % 2 == 0) ? localIdx + 1 : localIdx - 1;
            int swapIdx = (swapLocalIdx >= 0 && swapLocalIdx < (blockEnd - blockStart)) 
                          ? blockStart + swapLocalIdx : i;
            if (fabsf(result[i] - input[swapIdx]) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Shared Memory Level 1: Basics ===\n\n");
    
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
    printf("Testing basic shared memory load...\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    basicSharedLoad<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute expected
    for (int i = 0; i < N; i++) h_expected[i] = h_in[i] * 2.0f;
    
    if (verifyArray(h_out, h_expected, N)) {
        printf("✓ Basic shared memory PASSED\n");
    } else {
        printf("✗ Basic shared memory FAILED - Check declaration and synchronization\n");
    }
    
    // Test 2: Shared memory reverse
    printf("\nTesting shared memory reverse...\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    sharedReverse<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyReverse(h_out, h_in, N, blockSize)) {
        printf("✓ Shared memory reverse PASSED\n");
    } else {
        printf("✗ Shared memory reverse FAILED - Check reverse indexing\n");
    }
    
    // Test 3: Shared memory swap
    printf("\nTesting shared memory swap...\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    sharedSwap<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifySwap(h_out, h_in, N, blockSize)) {
        printf("✓ Shared memory swap PASSED\n");
    } else {
        printf("✗ Shared memory swap FAILED - Check swap logic\n");
    }
    
    // Test 4: Multi-stage processing
    printf("\nTesting multi-stage shared processing...\n");
    const float MULT = 3.0f;
    cudaMemset(d_out, 0, N * sizeof(float));
    multiStageShared<<<gridSize, blockSize>>>(d_in, d_out, N, MULT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Expected: (input * MULT) + threadIdx.x
    bool passMulti = true;
    for (int block = 0; block < gridSize && passMulti; block++) {
        for (int t = 0; t < blockSize; t++) {
            int idx = block * blockSize + t;
            if (idx < N) {
                float expected = (h_in[idx] * MULT) + t;
                if (fabsf(h_out[idx] - expected) > 1e-5f) passMulti = false;
            }
        }
    }
    
    if (passMulti) {
        printf("✓ Multi-stage processing PASSED\n");
    } else {
        printf("✗ Multi-stage processing FAILED - Complete the transformation\n");
    }
    
    // Cleanup
    free(h_in);
    free(h_out);
    free(h_expected);
    cudaFree(d_in);
    cudaFree(d_out);
    
    printf("\n=== Level 1 Complete ===\n");
    printf("Next: Try level2_tiled_matmul.cu for tiled algorithms\n");
    
    return 0;
}
