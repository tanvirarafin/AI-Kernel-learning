/*
 * Warp Primitives Level 2: Ballot Operations
 *
 * EXERCISE: Learn warp voting operations for collective decisions.
 *
 * CONCEPTS:
 * - Warp-level predicates
 * - Bitmask voting results
 * - Any/All operations
 * - Conditional execution
 *
 * SKILLS PRACTICED:
 * - __ballot_sync
 * - __any_sync
 * - __all_sync
 * - Bitmask manipulation
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Basic Ballot
 * Get bitmask of threads satisfying condition
 * TODO: Complete the ballot operation
// ============================================================================
__global__ void basicBallot(float *input, unsigned int *ballotResult, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        // TODO: Create predicate
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Get ballot result (bitmask)
        // unsigned int mask = __ballot_sync(0xffffffff, predicate);
        
        /* YOUR CODE HERE */
        
        if (laneId == 0) {
            ballotResult[blockIdx.x] = mask;
        }
    }
}

// ============================================================================
// KERNEL 2: Any Sync
 * Check if ANY thread in warp satisfies condition
 * TODO: Complete the any operation
// ============================================================================
__global__ void anySync(float *input, int *anyResult, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Check if any thread satisfies condition
        // int any = __any_sync(0xffffffff, predicate);
        
        /* YOUR CODE HERE */
        
        if (threadIdx.x == 0) {
            anyResult[blockIdx.x] = any;
        }
    }
}

// ============================================================================
// KERNEL 3: All Sync
 * Check if ALL threads in warp satisfy condition
 * TODO: Complete the all operation
// ============================================================================
__global__ void allSync(float *input, int *allResult, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Check if all threads satisfy condition
        // int all = __all_sync(0xffffffff, predicate);
        
        /* YOUR CODE HERE */
        
        if (threadIdx.x == 0) {
            allResult[blockIdx.x] = all;
        }
    }
}

// ============================================================================
// KERNEL 4: Count Active Threads
 * Count how many threads satisfy condition using ballot
 * TODO: Complete the count operation
// ============================================================================
__global__ void countWithBallot(float *input, int *countResult, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Get ballot and count set bits
        // unsigned int mask = __ballot_sync(0xffffffff, predicate);
        // int count = __popc(mask);  // Population count
        
        /* YOUR CODE HERE */
        
        if (laneId == 0) {
            countResult[blockIdx.x] = count;
        }
    }
}

// ============================================================================
// KERNEL 5: Find First Active Thread
 * Find the first lane that satisfies condition
 * TODO: Complete the find-first operation
// ============================================================================
__global__ void findFirstActive(float *input, int *firstIdx, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Get ballot and find first set bit
        // unsigned int mask = __ballot_sync(0xffffffff, predicate);
        // int firstLane = __ffs(mask);  // Find first set (1-indexed)
        
        /* YOUR CODE HERE */
        
        if (laneId == 0) {
            firstIdx[blockIdx.x] = firstLane - 1;  // Convert to 0-indexed
        }
    }
}

// ============================================================================
// KERNEL 6: Warp-Level Conditional Execution
 * Execute different code paths based on warp vote
 * TODO: Complete the conditional execution
// ============================================================================
__global__ void warpConditional(float *input, float *output, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        int predicate = (input[idx] > threshold) ? 1 : 0;
        
        // TODO: Check if majority satisfies condition
        // unsigned int mask = __ballot_sync(0xffffffff, predicate);
        // int count = __popc(mask);
        // bool majority = (count > WARP_SIZE / 2);
        
        /* YOUR CODE HERE */
        
        if (majority) {
            output[idx] = input[idx] * 2.0f;
        } else {
            output[idx] = input[idx] * 0.5f;
        }
    }
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) {
        arr[i] = val;
    }
}

void initArrayMixed(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(i % 100);
    }
}

int countSetBits(unsigned int mask) {
    int count = 0;
    while (mask) {
        count += mask & 1;
        mask >>= 1;
    }
    return count;
}

int main() {
    printf("=== Warp Primitives Level 2: Ballot Operations ===\n\n");
    
    const int N = 1024;
    float *h_input = (float*)malloc(N * sizeof(float));
    unsigned int *h_ballot = (unsigned int*)malloc(32 * sizeof(unsigned int));
    int *h_result = (int*)malloc(32 * sizeof(int));
    
    initArrayMixed(h_input, N);
    
    float *d_input;
    unsigned int *d_ballot;
    int *d_result;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_ballot, 32 * sizeof(unsigned int));
    cudaMalloc(&d_result, 32 * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = WARP_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    const float THRESHOLD = 50.0f;
    
    // Test 1: Basic ballot
    printf("Test 1: Basic ballot (threshold = %.1f)\n", THRESHOLD);
    basicBallot<<<gridSize, blockSize>>>(d_input, d_ballot, N, THRESHOLD);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_ballot, d_ballot, 32 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("  First 4 warps ballot results:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Warp %d: mask = 0x%08X (%d threads)\n", 
               i, h_ballot[i], countSetBits(h_ballot[i]));
    }
    
    // Test 2: Any sync
    printf("\nTest 2: Any sync\n");
    anySync<<<gridSize, blockSize>>>(d_input, d_result, N, THRESHOLD);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("  First 4 warps any_result:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Warp %d: %s\n", i, h_result[i] ? "YES" : "NO");
    }
    
    // Test 3: All sync
    printf("\nTest 3: All sync\n");
    allSync<<<gridSize, blockSize>>>(d_input, d_result, N, THRESHOLD);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("  First 4 warps all_result:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Warp %d: %s\n", i, h_result[i] ? "YES" : "NO");
    }
    
    // Test 4: Count with ballot
    printf("\nTest 4: Count threads above threshold\n");
    countWithBallot<<<gridSize, blockSize>>>(d_input, d_result, N, THRESHOLD);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("  First 4 warps count:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Warp %d: %d threads above threshold\n", i, h_result[i]);
    }
    
    // Cleanup
    free(h_input);
    free(h_ballot);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_ballot);
    cudaFree(d_result);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- __ballot_sync: Returns bitmask of predicate results\n");
    printf("- __any_sync: Returns non-zero if any predicate is true\n");
    printf("- __all_sync: Returns non-zero if all predicates are true\n");
    printf("- __popc: Count set bits in ballot mask\n");
    printf("- __ffs: Find first set bit (1-indexed)\n");
    printf("\nNext: Try level3_warp_reduction.cu for reduction patterns\n");
    
    return 0;
}
