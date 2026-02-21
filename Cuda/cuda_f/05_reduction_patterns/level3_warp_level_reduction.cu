/*
 * Reduction Level 3: Warp-Level Primitives
 *
 * EXERCISE: Use warp shuffle instructions for the fastest reduction.
 *
 * CONCEPTS:
 * - Warp shuffle instructions (__shfl_down_sync)
 * - No shared memory needed for warp-level reduction
 * - Hybrid: shared memory + warp shuffle
 * - Volta+ tensor core considerations
 *
 * SKILLS PRACTICED:
 * - Warp shuffle operations
 * - Lane ID extraction
 * - Warp-synchronous code
 * - Hybrid reduction patterns
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Warp Shuffle Reduction (Final Warp Only)
 * Use shuffle instructions for the final warp of reduction
 * TODO: Complete the warp-level reduction
// ============================================================================
__global__ void warpShuffleReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stage 1: Grid-stride accumulation to shared memory
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sharedData[tid] = sum;
    __syncthreads();
    
    // Stage 2: Reduce in shared memory to warp size
    // TODO: Reduce from BLOCK_SIZE to WARP_SIZE using shared memory
    /* YOUR CODE HERE */
    
    __syncthreads();
    
    // Stage 3: Warp-level reduction using shuffle
    // Only first warp participates
    if (tid < WARP_SIZE) {
        sum = sharedData[tid];
        
        // TODO: Use __shfl_down_sync to reduce within warp
        // Pattern:
        // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        //     sum += __shfl_down_sync(0xffffffff, sum, offset);
        // }
        // Note: 0xffffffff is the mask for all threads in warp
        /* YOUR CODE HERE - Warp shuffle reduction */
        
        sharedData[0] = sum;  // Thread 0 has final result
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 2: Pure Warp Shuffle (No Shared Memory)
 * Use only warp shuffle for entire reduction within warp
 * TODO: Implement pure warp-level reduction
// ============================================================================
__global__ void pureWarpReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = tid % WARP_SIZE;  // Lane within warp
    int warpId = tid / WARP_SIZE;  // Warp ID within block
    
    // Each warp processes a portion of data
    float warpSum = 0.0f;
    
    // TODO: Grid-stride at warp level
    // Each warp collectively processes elements
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        warpSum += input[i];
    }
    
    // TODO: Reduce within warp using shuffle
    // float result = warpSum;
    // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    //     result += __shfl_down_sync(0xffffffff, result, offset);
    // }
    float warpResult = /* YOUR CODE HERE */;
    
    // First lane of each warp has warp's total
    if (laneId == 0) {
        // Store in shared memory for block-level reduction
        __shared__ float warpSums[BLOCK_SIZE / WARP_SIZE];
        warpSums[warpId] = warpResult;
    }
    __syncthreads();
    
    // TODO: Reduce warp sums (at most 8 values for 256 threads)
    // Can use another warp shuffle or simple shared memory reduction
    /* YOUR CODE HERE */
    
    if (tid == 0) {
        output[blockIdx.x] = warpSums[0];
    }
}

// ============================================================================
// KERNEL 3: Template-Based Warp Reduction (Optimized)
 * Use compile-time constants for better optimization
 * TODO: Complete the templated reduction
// ============================================================================
template <int BLOCK_SIZE>
__global__ void templateWarpReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sharedData[tid] = sum;
    __syncthreads();
    
    // Reduce shared memory to warp size
    // TODO: Unroll based on BLOCK_SIZE
    // For BLOCK_SIZE = 256: 256->128->64->32
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sharedData[tid] += sharedData[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sharedData[tid] += sharedData[tid + 64];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 64) {
        if (tid < 32) sharedData[tid] += sharedData[tid + 32];
    }
    
    // Warp-level reduction (now only 32 elements)
    if (tid < 32) {
        // TODO: Complete warp shuffle reduction
        // float val = sharedData[tid];
        // val += __shfl_down_sync(0xffffffff, val, 16);
        // val += __shfl_down_sync(0xffffffff, val, 8);
        // val += __shfl_down_sync(0xffffffff, val, 4);
        // val += __shfl_down_sync(0xffffffff, val, 2);
        // val += __shfl_down_sync(0xffffffff, val, 1);
        // sharedData[0] = val;
        /* YOUR CODE HERE */
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 4: Warp Shuffle with Broadcast
 * After reduction, broadcast result to all threads
 * TODO: Implement reduction + broadcast
// ============================================================================
__global__ void warpReductionBroadcast(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sharedData[tid] = sum;
    __syncthreads();
    
    // Reduce to single value in sharedData[0]
    // TODO: Implement reduction (shared memory + warp shuffle)
    /* YOUR CODE HERE */
    
    __syncthreads();
    
    // TODO: Broadcast result to all threads using __shfl_sync
    // float result = __shfl_sync(0xffffffff, sharedData[0], 0);
    // output[idx] = result;
    /* YOUR CODE HERE */
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

int main() {
    printf("=== Reduction Level 3: Warp-Level Primitives ===\n\n");
    
    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(100 * sizeof(float));
    
    initArray(h_input, N, 1.0f);
    float expected = (float)N;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 100 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = 64;
    
    printf("Input size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);
    
    // Test 1: Warp shuffle reduction
    printf("Test 1: Warp shuffle reduction (hybrid)\n");
    warpShuffleReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    float *h_temp = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the warp shuffle reduction\n");
    }
    
    // Test 2: Pure warp reduction
    printf("\nTest 2: Pure warp shuffle reduction\n");
    pureWarpReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Implement pure warp reduction\n");
    }
    
    // Test 3: Template-based reduction
    printf("\nTest 3: Template-based warp reduction\n");
    templateWarpReduction<256><<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the template reduction\n");
    }
    
    // Test 4: Reduction with broadcast
    printf("\nTest 4: Warp reduction with broadcast\n");
    warpReductionBroadcast<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_temp, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Implement reduction and broadcast\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_temp);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- __shfl_down_sync: move data between lanes in a warp\n");
    printf("- Warp shuffle is faster than shared memory for warp-sized data\n");
    printf("- Hybrid approach: shared memory + warp shuffle is optimal\n");
    printf("- __shfl_sync can broadcast to all lanes\n");
    printf("- No __syncthreads() needed within a warp!\n");
    printf("\nNext: Try level4_multi_block_reduction.cu for large datasets\n");
    
    return 0;
}
