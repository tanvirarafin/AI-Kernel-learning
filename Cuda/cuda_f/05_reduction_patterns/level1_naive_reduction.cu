/*
 * Reduction Level 1: Naive Reduction
 *
 * EXERCISE: Learn the basic tree reduction pattern.
 *
 * CONCEPTS:
 * - Tree-based parallel reduction
 * - Sequential addressing in reduction
 * - Synchronization requirements
 * - Global memory reduction (slow baseline)
 *
 * SKILLS PRACTICED:
 * - Reduction loop pattern
 * - Power-of-2 stride handling
 * - Thread synchronization
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// ============================================================================
// KERNEL 1: Naive Global Memory Reduction (Incomplete)
 * Each block reduces a portion, then atomic add to global result
 * TODO: Complete the tree reduction pattern
// ============================================================================
__global__ void naiveReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread starts with one element (or 0 if out of bounds)
    float partialSum = (idx < n) ? input[idx] : 0.0f;

    // TODO: Implement strided reduction loop
    // Pattern: for (int stride = 1; stride < blockDim.x; stride *= 2) {
    //   if (threadIdx.x % (2*stride) == 0) {
    //     partialSum += input[idx + stride];  // Need to handle bounds
    //   }
    //   __syncthreads();
    // }
    // 
    // IMPORTANT: This naive version reads from global memory in the loop
    // (Later levels will optimize using shared memory)

    /* YOUR CODE HERE - Implement the reduction loop */

    // Thread 0 of each block writes partial sum to output
    if (tid == 0) {
        // TODO: Use atomicAdd to combine results from all blocks
        // atomicAdd(output, partialSum);
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 2: Sequential Addressing Reduction (Better Pattern)
 * Uses sequential addressing for better memory access pattern
 * TODO: Complete the implementation
// ============================================================================
__global__ void sequentialReduction(float *input, float *output, int n) {
    __shared__ float sharedData[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // TODO: Implement sequential addressing reduction
    // Pattern: for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //   if (tid < stride) {
    //     sharedData[tid] += sharedData[tid + stride];
    //   }
    //   __syncthreads();
    // }
    //
    // This pattern ensures threads with consecutive IDs are active together
    
    /* YOUR CODE HERE - Implement sequential reduction loop */
    
    // Thread 0 writes block's result
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// ============================================================================
// KERNEL 3: Unrolled Reduction (Optimization)
 * Manually unroll the last few iterations for better performance
 * TODO: Complete the unrolled reduction
// ============================================================================
__global__ void unrolledReduction(float *input, float *output, int n) {
    __shared__ float sharedData[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // TODO: Implement reduction with manual unrolling
    // For block size 256:
    // - Reduce from 256 to 32 normally
    // - Unroll the last 5 iterations (32->16->8->4->2->1)
    // - No sync needed in unrolled portion (all threads in warp)
    
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// ============================================================================
// KERNEL 4: Grid-Stride Reduction Loop
 * Handle inputs larger than total thread count
 * TODO: Complete the grid-stride accumulation
// ============================================================================
__global__ void gridStrideReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // TODO: Each thread accumulates multiple elements using grid-stride
    // Pattern: for (int i = idx; i < n; i += stride) {
    //   sum += input[i];
    // }
    float threadSum = 0.0f;
    /* YOUR CODE HERE - Implement grid-stride loop */
    
    // Store thread's partial sum in shared memory
    __shared__ float sharedData[256];
    sharedData[tid] = threadSum;
    __syncthreads();
    
    // TODO: Reduce within block using sequential addressing
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

float reduceCPU(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    printf("=== Reduction Level 1: Naive Reduction ===\n\n");

    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));
    
    // Initialize: all 1s, so sum should equal N
    initArray(h_input, N, 1.0f);
    float expected = (float)N;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Input size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);
    
    // Test 1: Naive reduction
    printf("Test 1: Naive global memory reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    naiveReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED (within 1%% tolerance)\n");
    } else {
        printf("  ✗ FAILED - Complete the reduction loop and atomicAdd\n");
    }
    
    // Test 2: Sequential addressing reduction
    printf("\nTest 2: Sequential addressing reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    sequentialReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the sequential reduction loop\n");
    }
    
    // Test 3: Unrolled reduction
    printf("\nTest 3: Unrolled reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    unrolledReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Implement the unrolled reduction\n");
    }
    
    // Test 4: Grid-stride reduction
    printf("\nTest 4: Grid-stride reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    // Use fewer blocks to test grid-stride
    int gsGridSize = 64;
    gridStrideReduction<<<gsGridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the grid-stride loop\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Tree reduction: halve working set each iteration\n");
    printf("- Sequential addressing: better thread utilization\n");
    printf("- Unrolling: reduces loop overhead for last iterations\n");
    printf("- Grid-stride: handles inputs larger than thread count\n");
    printf("- AtomicAdd: combines partial results from blocks\n");
    printf("\nNext: Try level2_shared_memory_reduction.cu for optimization\n");
    
    return 0;
}
