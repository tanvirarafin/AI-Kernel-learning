/*
 * Warp Primitives Level 3: Warp Reduction
 *
 * EXERCISE: Implement efficient reduction using warp shuffle.
 *
 * CONCEPTS:
 * - Warp-synchronous reduction
 * - No synchronization needed
 * - Tree reduction within warp
 * - Hybrid shared memory + warp
 *
 * SKILLS PRACTICED:
 * - __shfl_down_sync for reduction
 * - Warp-level tree reduction
 * - Combining with shared memory
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Pure Warp Reduction
 * Reduce within a single warp using shuffle
 * TODO: Complete the warp reduction
// ============================================================================
__global__ void pureWarpReduction(float *input, float *output, int n) {
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread accumulates its portion
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // TODO: Reduce within warp using shuffle down
    // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    //     sum += __shfl_down_sync(0xffffffff, sum, offset);
    // }
    
    /* YOUR CODE HERE */
    
    // Lane 0 has the warp's total
    if (laneId == 0) {
        output[blockIdx.x * gridDim.x + warpId] = sum;
    }
}

// ============================================================================
// KERNEL 2: Hybrid Shared Memory + Warp Reduction
 * Reduce to warp size with shared memory, then warp shuffle
 * TODO: Complete the hybrid reduction
// ============================================================================
__global__ void hybridReduction(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = tid % WARP_SIZE;
    
    // Grid-stride accumulation
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // Reduce shared memory to warp size
    // TODO: Reduce from 256 to 32 using shared memory
    // for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride >>= 1) {
    //     if (tid < stride) {
    //         sharedData[tid] += sharedData[tid + stride];
    //     }
    //     __syncthreads();
    // }
    
    /* YOUR CODE HERE */
    
    __syncthreads();
    
    // Warp-level reduction for final 32 elements
    if (tid < WARP_SIZE) {
        sum = sharedData[tid];
        
        // TODO: Complete warp shuffle reduction
        // for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        //     sum += __shfl_down_sync(0xffffffff, sum, offset);
        // }
        
        /* YOUR CODE HERE */
        
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// ============================================================================
// KERNEL 3: Unrolled Warp Reduction
 * Manually unroll the warp reduction loop
 * TODO: Complete the unrolled reduction
// ============================================================================
__global__ void unrolledWarpReduction(float *input, float *output, int n) {
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
    
    // Reduce to warp size
    if (BLOCK_SIZE >= 64) {
        if (tid < 32) sharedData[tid] += sharedData[tid + 32];
    }
    __syncthreads();
    
    // Warp-level (now only 32 elements, no sync needed)
    if (tid < 32) {
        sum = sharedData[tid];
        
        // TODO: Unroll warp reduction
        // sum += __shfl_down_sync(0xffffffff, sum, 16);
        // sum += __shfl_down_sync(0xffffffff, sum, 8);
        // sum += __shfl_down_sync(0xffffffff, sum, 4);
        // sum += __shfl_down_sync(0xffffffff, sum, 2);
        // sum += __shfl_down_sync(0xffffffff, sum, 1);
        
        /* YOUR CODE HERE */
        
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// ============================================================================
// KERNEL 4: Warp Reduction with Broadcast
 * After reduction, broadcast result to all threads
 * TODO: Complete reduction + broadcast
// ============================================================================
__global__ void warpReductionBroadcast(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = tid % WARP_SIZE;
    
    // Accumulate
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // Reduce to warp size
    if (tid < 32) {
        sharedData[tid] += sharedData[tid + 32];
        sharedData[tid] += sharedData[tid + 64];
        sharedData[tid] += sharedData[tid + 128];
    }
    __syncthreads();
    
    // Warp reduction and broadcast
    if (tid < 32) {
        sum = sharedData[tid];
        
        // Reduce
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        
        // TODO: Broadcast result to all lanes in warp
        // sum = __shfl_sync(0xffffffff, sum, 0);
        
        /* YOUR CODE HERE */
    }
    
    output[idx] = sum;
}

// ============================================================================
// KERNEL 5: Multi-Block Warp Reduction
 * Two-pass reduction: block-level then warp
 * TODO: Complete the multi-block reduction
// ============================================================================
__global__ void multiBlockWarpReduction(float *input, float *partial, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First pass: block-level reduction
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    sharedData[tid] = sum;
    __syncthreads();
    
    // Reduce in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial[blockIdx.x] = sharedData[0];
    }
    
    // Second pass: one block reduces the partials
    __syncthreads();
    
    if (blockIdx.x == 0) {
        sum = (tid < gridDim.x) ? partial[tid] : 0.0f;
        sharedData[tid] = sum;
        __syncthreads();
        
        // TODO: Use warp reduction for final pass
        /* YOUR CODE HERE */
        
        if (tid == 0) {
            *output = sharedData[0];
        }
    }
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) {
        arr[i] = val;
    }
}

int main() {
    printf("=== Warp Primitives Level 3: Warp Reduction ===\n\n");
    
    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(100 * sizeof(float));
    
    initArray(h_input, N, 1.0f);  // All 1s, sum = N
    
    float *d_input, *d_output, *d_partial;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 100 * sizeof(float));
    cudaMalloc(&d_partial, 100 * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = 64;
    
    printf("Input size: %d (all 1.0f)\n", N);
    printf("Expected sum: %d\n\n", N);
    
    // Test 1: Pure warp reduction
    printf("Test 1: Pure warp reduction\n");
    pureWarpReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    // Sum partial results
    float result = 0.0f;
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < gridSize; i++) result += h_output[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - N) < N * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the warp shuffle reduction\n");
    }
    
    // Test 2: Hybrid reduction
    printf("\nTest 2: Hybrid shared memory + warp reduction\n");
    hybridReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_output[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - N) < N * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the hybrid reduction\n");
    }
    
    // Test 3: Unrolled warp reduction
    printf("\nTest 3: Unrolled warp reduction\n");
    unrolledWarpReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_output[i];
    
    printf("  Result: %.0f\n", result);
    if (fabsf(result - N) < N * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the unrolled reduction\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Warp shuffle is faster than shared memory for final 32 elements\n");
    printf("- Hybrid: shared memory to warp size, then shuffle\n");
    printf("- Unrolled: manually expand the reduction loop\n");
    printf("- __shfl_sync can broadcast to all lanes\n");
    printf("- No __syncthreads() needed within warp!\n");
    printf("\nNext: Try level4_warp_broadcast.cu for broadcast patterns\n");
    
    return 0;
}
