/*
 * Reduction Level 4: Multi-Block Reduction
 *
 * EXERCISE: Handle large datasets that span multiple blocks.
 *
 * CONCEPTS:
 * - Two-pass reduction
 * - Grid-stride loops
 * - Atomic operations for final combine
 * - Multi-kernel reduction pipeline
 *
 * SKILLS PRACTICED:
 * - Multi-block coordination
 * - Partial sum management
 * - Atomic operations
 * - Scalable reduction
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 10000000  // 10 million elements
#define BLOCK_SIZE 256

// ============================================================================
// KERNEL 1: Two-Pass Reduction - First Pass
 * Each block reduces its portion to a partial sum
 * TODO: Complete the first pass reduction
// ============================================================================
__global__ void reductionPass1(float *input, float *partialSums, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Grid-stride accumulation
    // Each thread accumulates multiple elements across the input
    float threadSum = 0.0f;
    /* YOUR CODE HERE - Grid-stride loop */
    
    sharedData[tid] = threadSum;
    __syncthreads();
    
    // TODO: Block-level reduction using sequential addressing
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         sharedData[tid] += sharedData[tid + stride];
    //     }
    //     __syncthreads();
    // }
    /* YOUR CODE HERE */
    
    // Each block writes its partial sum
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 2: Two-Pass Reduction - Second Pass
 * Combine partial sums from first pass
 * TODO: Complete the second pass
// ============================================================================
__global__ void reductionPass2(float *partialSums, float *output, int numPartialSums) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // TODO: Load partial sums (there are numPartialSums of them)
    // Use grid-stride if numPartialSums > BLOCK_SIZE
    float sum = 0.0f;
    /* YOUR CODE HERE */
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // TODO: Reduce within block
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        *output = sharedData[0];
    }
}

// ============================================================================
// KERNEL 3: Single-Pass Atomic Reduction
 * Use atomicAdd for single-pass reduction (simpler but slower)
 * TODO: Complete the atomic reduction
// ============================================================================
__global__ void atomicReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread accumulates a portion
    float threadSum = 0.0f;
    
    // TODO: Grid-stride accumulation
    /* YOUR CODE HERE */
    
    // Reduce within block using shared memory
    __shared__ float sharedData[BLOCK_SIZE];
    sharedData[tid] = threadSum;
    __syncthreads();
    
    // TODO: Block-level reduction
    /* YOUR CODE HERE */
    
    // TODO: Use atomicAdd to combine results from all blocks
    // Only thread 0 from each block does the atomic add
    if (tid == 0) {
        /* YOUR CODE HERE - atomicAdd to output */
    }
}

// ============================================================================
// KERNEL 4: Recursive Reduction (Advanced)
 * Launch second pass from within first pass kernel
 * Uses CUDA dynamic parallelism (requires compute capability 3.5+)
 * TODO: Complete the recursive approach
// ============================================================================
__global__ void recursiveReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First pass: reduce to partial sums
    float threadSum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        threadSum += input[i];
    }
    sharedData[tid] = threadSum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    // Launch second pass if we have multiple blocks
    if (tid == 0 && gridDim.x > 1) {
        // Store partial sum
        input[blockIdx.x] = sharedData[0];
        
        // TODO: Launch second pass kernel recursively
        // This requires dynamic parallelism
        // reductionPass2<<<1, BLOCK_SIZE>>>(input, output, gridDim.x);
        
        // For now, just mark that we're done with first pass
        // The host will launch the second pass
    }
}

// ============================================================================
// KERNEL 5: Segmented Reduction
 * Reduce multiple segments independently in one kernel
 * TODO: Complete segmented reduction
// ============================================================================
__global__ void segmentedReduction(float *input, float *output, 
                                   int n, int numSegments) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int segmentId = blockIdx.x;
    
    // Calculate segment boundaries
    int segmentSize = (n + numSegments - 1) / numSegments;
    int segmentStart = segmentId * segmentSize;
    int segmentEnd = min(segmentStart + segmentSize, n);
    
    // TODO: Reduce this segment
    float segmentSum = 0.0f;
    for (int i = segmentStart + tid; i < segmentEnd; i += blockDim.x) {
        segmentSum += input[i];
    }
    
    sharedData[tid] = segmentSum;
    __syncthreads();
    
    // TODO: Block-level reduction
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        output[segmentId] = sharedData[0];
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
    // Process in chunks to avoid overflow in loop counter
    const int CHUNK = 1000000;
    for (int start = 0; start < n; start += CHUNK) {
        int end = min(start + CHUNK, n);
        for (int i = start; i < end; i++) {
            sum += arr[i];
        }
    }
    return sum;
}

int main() {
    printf("=== Reduction Level 4: Multi-Block Reduction ===\n\n");
    
    const int N = 10000000;  // 10 million
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));
    float *h_partial = (float*)malloc(1000 * sizeof(float));
    
    initArray(h_input, N, 1.0f);
    float expected = (float)N;
    
    float *d_input, *d_output, *d_partial;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_partial, 1000 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = 256;  // 256 blocks
    
    printf("Input size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);
    
    // Test 1: Two-pass reduction
    printf("Test 1: Two-pass reduction\n");
    printf("  Pass 1: Reducing %d elements to %d partial sums...\n", N, gridSize);
    reductionPass1<<<gridSize, blockSize>>>(d_input, d_partial, N);
    cudaDeviceSynchronize();
    
    printf("  Pass 2: Combining %d partial sums...\n", gridSize);
    reductionPass2<<<1, blockSize>>>(d_partial, d_output, gridSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete both reduction passes\n");
    }
    
    // Test 2: Atomic reduction
    printf("\nTest 2: Single-pass atomic reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    atomicReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the atomic reduction\n");
    }
    
    // Test 3: Segmented reduction
    printf("\nTest 3: Segmented reduction\n");
    const int NUM_SEGMENTS = 100;
    float *h_segOutput = (float*)malloc(NUM_SEGMENTS * sizeof(float));
    float *d_segOutput;
    cudaMalloc(&d_segOutput, NUM_SEGMENTS * sizeof(float));
    
    segmentedReduction<<<NUM_SEGMENTS, blockSize>>>(d_input, d_segOutput, N, NUM_SEGMENTS);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_segOutput, d_segOutput, NUM_SEGMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sum all segment results
    float segmentTotal = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        segmentTotal += h_segOutput[i];
    }
    
    printf("  Result: %.0f (from %d segments)\n", segmentTotal, NUM_SEGMENTS);
    if (fabsf(segmentTotal - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the segmented reduction\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_partial);
    free(h_segOutput);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    cudaFree(d_segOutput);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Two-pass reduction: better performance than atomic for large data\n");
    printf("- Atomic reduction: simpler code, good for small datasets\n");
    printf("- Segmented reduction: process multiple segments independently\n");
    printf("- Grid-stride loops handle any input size\n");
    printf("\nNext: Try level5_advanced_patterns.cu for custom reductions\n");
    
    return 0;
}
