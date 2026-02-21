/*
 * Warp Primitives Level 1: Shuffle Operations
 *
 * EXERCISE: Learn basic warp shuffle instructions for efficient
 * intra-warp communication without shared memory.
 *
 * CONCEPTS:
 * - Warp-level communication
 * - Lane ID extraction
 * - Shuffle operations (no sync needed!)
 * - Register-to-register transfer
 *
 * SKILLS PRACTICED:
 * - __shfl_sync (broadcast)
 * - __shfl_down_sync (shift down)
 * - __shfl_up_sync (shift up)
 * - __shfl_xor_sync (exchange)
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Basic Shuffle Down
 * Move data from higher lane to lower lane
 * TODO: Complete the shuffle down operation
// ============================================================================
__global__ void shuffleDown(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Shuffle down by 1 lane
        // val from lane N goes to lane N-1
        // float shuffled = __shfl_down_sync(0xffffffff, val, 1);
        
        /* YOUR CODE HERE */
        
        output[idx] = shuffled;
    }
}

// ============================================================================
// KERNEL 2: Shuffle Up
 * Move data from lower lane to higher lane
 * TODO: Complete the shuffle up operation
// ============================================================================
__global__ void shuffleUp(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Shuffle up by 1 lane
        // val from lane N goes to lane N+1
        // float shuffled = __shfl_up_sync(0xffffffff, val, 1);
        
        // Note: Lane 0 receives 0 when shuffling up
        
        /* YOUR CODE HERE */
        
        output[idx] = shuffled;
    }
}

// ============================================================================
// KERNEL 3: Shuffle XOR (Exchange)
 * Exchange data with lane (laneId ^ offset)
 * TODO: Complete the shuffle XOR operation
// ============================================================================
__global__ void shuffleXor(float *input, float *output, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Shuffle with XOR offset
        // Lane 0 exchanges with lane (0 ^ offset)
        // Lane 1 exchanges with lane (1 ^ offset)
        // float shuffled = __shfl_xor_sync(0xffffffff, val, offset);
        
        /* YOUR CODE HERE */
        
        output[idx] = shuffled;
    }
}

// ============================================================================
// KERNEL 4: Shuffle (Broadcast from Specific Lane)
 * Broadcast value from one lane to all lanes in warp
 * TODO: Complete the broadcast operation
// ============================================================================
__global__ void shuffleBroadcast(float *input, float *output, int n, int srcLane) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Broadcast from srcLane to all lanes
        // float broadcast = __shfl_sync(0xffffffff, val, srcLane);
        
        /* YOUR CODE HERE */
        
        output[idx] = broadcast;
    }
}

// ============================================================================
// KERNEL 5: Shuffle Index (Get from Specific Lane)
 * Get value from a specific lane index
 * TODO: Complete the indexed shuffle
// ============================================================================
__global__ void shuffleIndex(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Get value from lane (31 - laneId) - reverse within warp
        // float reversed = __shfl_sync(0xffffffff, val, 31 - laneId);
        
        /* YOUR CODE HERE */
        
        output[idx] = reversed;
    }
}

// ============================================================================
// KERNEL 6: Warp Reverse Using Shuffle
 * Reverse the order of elements within each warp
 * TODO: Complete the warp reversal
// ============================================================================
__global__ void warpReverse(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Reverse using shuffle
        // Each lane gets value from (WARP_SIZE - 1 - laneId)
        // float reversed = __shfl_sync(0xffffffff, val, WARP_SIZE - 1 - laneId);
        
        /* YOUR CODE HERE */
        
        output[idx] = reversed;
    }
}

// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)i;
    }
}

void printWarpResults(float *arr, int warpSize, const char *label) {
    printf("%s (first warp):\n", label);
    for (int i = 0; i < warpSize; i++) {
        printf("  Lane %2d: %8.2f\n", i, arr[i]);
    }
}

int main() {
    printf("=== Warp Primitives Level 1: Shuffle Operations ===\n\n");
    
    const int N = 1024;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    initArray(h_input, N);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 32;  // One warp per block for clarity
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Array size: %d\n", N);
    printf("Block size: %d (one warp)\n\n", blockSize);
    
    // Test 1: Shuffle down
    printf("Test 1: Shuffle down by 1\n");
    shuffleDown<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printWarpResults(h_output, WARP_SIZE, "  Results");
    printf("  Expected: Lane i gets value from lane i+1\n");
    
    // Test 2: Shuffle up
    printf("\nTest 2: Shuffle up by 1\n");
    shuffleUp<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printWarpResults(h_output, WARP_SIZE, "  Results");
    printf("  Expected: Lane i gets value from lane i-1 (lane 0 gets 0)\n");
    
    // Test 3: Shuffle XOR
    printf("\nTest 3: Shuffle XOR with offset 1\n");
    shuffleXor<<<gridSize, blockSize>>>(d_input, d_output, N, 1);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printWarpResults(h_output, WARP_SIZE, "  Results");
    printf("  Expected: Lane i swaps with lane i^1\n");
    
    // Test 4: Broadcast
    printf("\nTest 4: Broadcast from lane 0\n");
    shuffleBroadcast<<<gridSize, blockSize>>>(d_input, d_output, N, 0);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printWarpResults(h_output, WARP_SIZE, "  Results");
    printf("  Expected: All lanes get value from lane 0\n");
    
    // Test 5: Warp reverse
    printf("\nTest 5: Warp reverse\n");
    warpReverse<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printWarpResults(h_output, WARP_SIZE, "  Results");
    printf("  Expected: Lane i gets value from lane 31-i\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- __shfl_down_sync: Move data to lower lane IDs\n");
    printf("- __shfl_up_sync: Move data to higher lane IDs\n");
    printf("- __shfl_xor_sync: Exchange with (laneId ^ offset)\n");
    printf("- __shfl_sync: Get value from specific lane\n");
    printf("- No __syncthreads() needed within a warp!\n");
    printf("- Mask 0xffffffff = all threads in warp participate\n");
    printf("\nNext: Try level2_ballot_ops.cu for warp voting\n");
    
    return 0;
}
