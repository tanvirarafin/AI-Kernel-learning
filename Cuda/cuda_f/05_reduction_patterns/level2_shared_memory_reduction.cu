/*
 * Reduction Level 2: Shared Memory Optimization
 *
 * EXERCISE: Optimize reduction using shared memory to minimize
 * global memory accesses.
 *
 * CONCEPTS:
 * - Load once to shared memory
 * - Reduce in shared memory (fast)
 * - Single global write per block
 * - Avoiding bank conflicts in reduction
 *
 * SKILLS PRACTICED:
 * - Shared memory loading
 * - In-place reduction
 * - Bank conflict avoidance
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define BLOCK_SIZE 256

// ============================================================================
// KERNEL 1: Basic Shared Memory Reduction
 * Load to shared memory, reduce there, write once to global
 * TODO: Complete the shared memory reduction
// ============================================================================
__global__ void sharedMemReduction(float *input, float *output, int n) {
    // TODO: Declare shared memory
    __shared__ float sharedData[/* YOUR CODE HERE */];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Load data from global to shared memory
    // Use grid-stride loop to handle large inputs
    float sum = 0.0f;
    /* YOUR CODE HERE - Grid-stride load */
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // TODO: Implement tree reduction in shared memory
    // Use sequential addressing pattern
    /* YOUR CODE HERE - Reduction loop */
    
    // Thread 0 writes block's result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 2: Bank Conflict-Free Reduction
 * Add padding to avoid bank conflicts during reduction
 * TODO: Complete the padded shared memory version
// ============================================================================
__global__ void sharedMemReductionPadded(float *input, float *output, int n) {
    // TODO: Add padding to avoid bank conflicts
    // With 32 banks and 256 threads, add 1 element padding per 32
    // Example: __shared__ float sharedData[BLOCK_SIZE + BLOCK_SIZE/32];
    __shared__ float sharedData[/* YOUR CODE HERE - Add padding */];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load with grid-stride
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // TODO: Store with padded index
    sharedData[/* YOUR CODE HERE */] = sum;
    __syncthreads();
    
    // TODO: Reduce with padded indexing
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //   if (tid < stride) {
    //     sharedData[tid + padded_offset] += sharedData[tid + stride + padded_offset];
    //   }
    //   __syncthreads();
    // }
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 3: Multi-Stage Reduction
 * First pass: block-level reduction to shared memory
 * Second pass: atomic add to final result
 * TODO: Complete both stages
// ============================================================================
__global__ void multiStageReduction(float *input, float *partialSums, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Grid-stride accumulation
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // Stage 2: Block-level reduction
    // TODO: Implement reduction with sequential addressing
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// Second kernel: combine partial sums
__global__ void combinePartialSums(float *partialSums, float *output, int numBlocks) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Load partial sums (fewer than BLOCK_SIZE typically)
    if (tid < numBlocks) {
        sharedData[tid] = partialSums[tid];
    } else {
        sharedData[tid] = 0.0f;
    }
    __syncthreads();
    
    // TODO: Reduce partial sums
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        *output = sharedData[0];
    }
}

// ============================================================================
// KERNEL 4: Interleaved Addressing (Alternative Pattern)
 * Uses interleaved addressing instead of sequential
 * TODO: Complete and compare with sequential
// ============================================================================
__global__ void interleavedReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sharedData[tid] = sum;
    __syncthreads();
    
    // TODO: Implement interleaved addressing reduction
    // Pattern: for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //   if (tid < stride) {
    //     sharedData[tid] += sharedData[tid + stride];
    //   }
    //   __syncthreads();
    // }
    // Note: This has more bank conflicts than sequential!
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
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
    printf("=== Reduction Level 2: Shared Memory Optimization ===\n\n");
    
    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));
    float *h_partial = (float*)malloc(100 * sizeof(float));
    
    initArray(h_input, N, 1.0f);
    float expected = (float)N;
    
    float *d_input, *d_output, *d_partial;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 100 * sizeof(float));
    cudaMalloc(&d_partial, 100 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = 64;  // Use fixed grid size
    
    printf("Input size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);
    
    // Test 1: Basic shared memory reduction
    printf("Test 1: Basic shared memory reduction\n");
    sharedMemReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Combine results on CPU for now
    float *h_temp = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the shared memory reduction\n");
    }
    
    // Test 2: Padded version
    printf("\nTest 2: Bank conflict-free (padded) reduction\n");
    sharedMemReductionPadded<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Fix the padded indexing\n");
    }
    
    // Test 3: Multi-stage reduction
    printf("\nTest 3: Multi-stage reduction\n");
    multiStageReduction<<<gridSize, blockSize>>>(d_input, d_partial, N);
    cudaDeviceSynchronize();
    
    combinePartialSums<<<1, blockSize>>>(d_partial, d_output, gridSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete both kernel stages\n");
    }
    
    // Test 4: Interleaved addressing
    printf("\nTest 4: Interleaved addressing reduction\n");
    interleavedReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the interleaved reduction\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_partial);
    free(h_temp);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory reduces global memory traffic significantly\n");
    printf("- Sequential addressing has fewer bank conflicts than interleaved\n");
    printf("- Padding eliminates bank conflicts but uses more memory\n");
    printf("- Multi-stage: reduce in blocks, then combine\n");
    printf("\nNext: Try level3_warp_level_reduction.cu for warp shuffle ops\n");
    
    return 0;
}
