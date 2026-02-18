/*
 * Warp Primitives Level 5: Advanced Warp Patterns
 *
 * EXERCISE: Combine warp primitives for complex algorithms.
 *
 * CONCEPTS:
 * - Combined shuffle + ballot
 * - Warp-level match operations
 * - Complex communication patterns
 * - Warp-synchronous algorithms
 *
 * SKILLS PRACTICED:
 * - Primitive composition
 * - Advanced warp algorithms
 * - Performance optimization
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Warp Match
 * Find lanes with matching values
 * TODO: Complete the match operation
// ============================================================================
__global__ void warpMatch(float *input, int *matchCount, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float myVal = input[idx];
        
        // TODO: Count how many lanes have the same value
        // Use ballot to get all predicates, then count matches
        
        /* YOUR CODE HERE */
        
        if (laneId == 0) {
            matchCount[blockIdx.x] = count;
        }
    }
}

// ============================================================================
// KERNEL 2: Warp-Level Segmented Reduction
 * Reduce segments within a warp
 * TODO: Complete the segmented reduction
// ============================================================================
__global__ void warpSegmentedReduction(float *input, float *output, 
                                        int *segmentBoundaries, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        int segmentStart = segmentBoundaries[laneId];
        int segmentEnd = segmentBoundaries[laneId + 1];
        
        // TODO: Reduce within segment using shuffle
        // Need to handle variable segment sizes
        
        /* YOUR CODE HERE */
        
        if (laneId == segmentStart) {
            output[blockIdx.x * WARP_SIZE + laneId] = segmentSum;
        }
    }
}

// ============================================================================
// KERNEL 3: Warp-Level Sort (Odd-Even Transposition)
 * Sort elements within a warp using odd-even transposition
 * TODO: Complete the warp sort
// ============================================================================
__global__ void warpSort(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // Odd-even transposition sort within warp
        for (int iter = 0; iter < WARP_SIZE; iter++) {
            if (iter % 2 == 0) {
                // Even iteration: compare with lane+1
                if (laneId % 2 == 0 && laneId < WARP_SIZE - 1) {
                    float neighbor = __shfl_down_sync(0xffffffff, val, 1);
                    if (val > neighbor) {
                        // Swap
                        val = neighbor;
                        // Need to send our val to neighbor
                    }
                }
            } else {
                // Odd iteration: compare with lane-1
                if (laneId % 2 == 1) {
                    float neighbor = __shfl_up_sync(0xffffffff, val, 1);
                    if (val < neighbor) {
                        // Swap
                        val = neighbor;
                    }
                }
            }
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// KERNEL 4: Warp-Level Prefix Sum (Scan)
 * Inclusive scan within a warp
 * TODO: Complete the warp scan
// ============================================================================
__global__ void warpScan(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Implement warp-level inclusive scan
        // Using shuffle up:
        // float sum = val;
        // sum += __shfl_up_sync(0xffffffff, sum, 1);
        // sum += __shfl_up_sync(0xffffffff, sum, 2);
        // sum += __shfl_up_sync(0xffffffff, sum, 4);
        // sum += __shfl_up_sync(0xffffffff, sum, 8);
        // sum += __shfl_up_sync(0xffffffff, sum, 16);
        
        /* YOUR CODE HERE */
        
        output[idx] = sum;
    }
}

// ============================================================================
// KERNEL 5: Warp-Level Histogram
 * Build histogram within a warp
 * TODO: Complete the warp histogram
// ============================================================================
__global__ void warpHistogram(float *input, unsigned int *hist, 
                               int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        int bin = (int)input[idx];
        
        // TODO: Each lane counts its bin, then reduce
        // Use ballot to find all lanes with same bin
        
        /* YOUR CODE HERE */
        
        if (laneId == 0) {
            hist[blockIdx.x * numBins + bin] = count;
        }
    }
}

// ============================================================================
// KERNEL 6: Warp-Level Matrix Transpose
 * Transpose a WARP_SIZE x WARP_SIZE matrix within a warp
 * TODO: Complete the warp transpose
// ============================================================================
__global__ void warpMatrixTranspose(float *input, float *output, int matrixSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < matrixSize * matrixSize) {
        int row = idx / matrixSize;
        int col = idx % matrixSize;
        
        float val = input[idx];
        
        // TODO: Transpose using shuffle
        // Each lane holds one element, need to exchange
        // Transposed position: (col, row)
        
        /* YOUR CODE HERE */
        
        int transposedIdx = col * matrixSize + row;
        output[transposedIdx] = transposedVal;
    }
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) {
        arr[i] = val;
    }
}

void initArraySequential(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(i % 10);
    }
}

int main() {
    printf("=== Warp Primitives Level 5: Advanced Patterns ===\n\n");
    
    const int N = 1024;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    unsigned int *h_hist = (unsigned int*)malloc(32 * sizeof(unsigned int));
    
    initArraySequential(h_input, N);
    
    float *d_input, *d_output;
    unsigned int *d_hist;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_hist, 32 * sizeof(unsigned int));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = WARP_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Array size: %d\n", N);
    printf("Block size: %d (one warp)\n\n", blockSize);
    
    // Test 1: Warp scan
    printf("Test 1: Warp-level prefix sum (scan)\n");
    warpScan<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First warp results:\n");
    float expected = 0;
    for (int i = 0; i < 8; i++) {
        expected += h_input[i];
        printf("    Lane %d: %.2f (expected: %.2f)\n", i, h_output[i], expected);
    }
    
    // Test 2: Warp match
    printf("\nTest 2: Warp match (count matching values)\n");
    int *d_matchCount;
    cudaMalloc(&d_matchCount, 32 * sizeof(int));
    
    warpMatch<<<gridSize, blockSize>>>(d_input, d_matchCount, N);
    cudaDeviceSynchronize();
    
    int *h_matchCount = (int*)malloc(32 * sizeof(int));
    cudaMemcpy(h_matchCount, d_matchCount, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Match counts for first 4 warps:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Warp %d: %d matches\n", i, h_matchCount[i]);
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_hist);
    free(h_matchCount);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_matchCount);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Combine shuffle + ballot for complex patterns\n");
    printf("- Warp scan: use __shfl_up with increasing offsets\n");
    printf("- Odd-even sort: alternate comparison directions\n");
    printf("- Segmented reduction: handle variable segment sizes\n");
    printf("- Warp histogram: use ballot for bin counting\n");
    printf("\n=== Warp Primitives Module Complete ===\n");
    printf("Next: Explore cuda_streams for concurrent execution\n");
    
    return 0;
}
