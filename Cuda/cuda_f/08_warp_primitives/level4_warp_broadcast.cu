/*
 * Warp Primitives Level 4: Warp Broadcast
 *
 * EXERCISE: Efficient data distribution within a warp.
 *
 * CONCEPTS:
 * - Single source to all lanes
 * - Efficient data distribution
 * - Warp-level communication patterns
 *
 * SKILLS PRACTICED:
 * - __shfl for broadcast
 * - Multi-value broadcast
 * - Conditional broadcast
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define WARP_SIZE 32

// ============================================================================
// KERNEL 1: Basic Broadcast from Lane 0
 * Broadcast single value from lane 0 to all lanes
 * TODO: Complete the broadcast
// ============================================================================
__global__ void broadcastFromLane0(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Broadcast from lane 0 to all lanes
        // float broadcast = __shfl_sync(0xffffffff, val, 0);
        
        /* YOUR CODE HERE */
        
        output[idx] = broadcast;
    }
}

// ============================================================================
// KERNEL 2: Broadcast from Any Lane
 * Broadcast from a specified lane index
 * TODO: Complete the flexible broadcast
// ============================================================================
__global__ void broadcastFromLane(float *input, float *output, int n, int srcLane) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Broadcast from srcLane
        // float broadcast = __shfl_sync(0xffffffff, val, srcLane);
        
        /* YOUR CODE HERE */
        
        output[idx] = broadcast;
    }
}

// ============================================================================
// KERNEL 3: Multi-Value Broadcast
 * Broadcast multiple values from different lanes
 * TODO: Complete the multi-value broadcast
// ============================================================================
__global__ void multiValueBroadcast(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        // Each lane has a different value to broadcast
        float myVal = input[idx];
        
        // TODO: Broadcast each lane's value to all other lanes
        // This creates a situation where all lanes know all values
        // For simplicity, broadcast lane 0's value to position 0,
        // lane 1's value to position 1, etc.
        
        // float fromLane0 = __shfl_sync(0xffffffff, myVal, 0);
        // float fromLane1 = __shfl_sync(0xffffffff, myVal, 1);
        // ...
        
        /* YOUR CODE HERE - Broadcast from multiple lanes */
        
        // For this exercise, just broadcast from lane (laneId % WARP_SIZE)
        output[idx] = myVal;  // Replace with actual broadcast
    }
}

// ============================================================================
// KERNEL 4: Conditional Broadcast
 * Broadcast only if condition is met
 * TODO: Complete the conditional broadcast
// ============================================================================
__global__ void conditionalBroadcast(float *input, float *output, 
                                      int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        int shouldBroadcast = (val > threshold) ? 1 : 0;
        
        // TODO: Check if any lane should broadcast
        // int anyBroadcast = __any_sync(0xffffffff, shouldBroadcast);
        
        /* YOUR CODE HERE */
        
        if (anyBroadcast) {
            // TODO: Find first lane that meets condition and broadcast its value
            // Use ballot and ffs to find first
            /* YOUR CODE HERE */
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// KERNEL 5: Warp-Level Gather
 * Gather values from all lanes to one lane
 * TODO: Complete the gather operation
// ============================================================================
__global__ void warpGather(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n && laneId == 0) {
        // Lane 0 gathers values from all lanes in its warp
        float gathered[WARP_SIZE];
        
        // TODO: Gather from each lane
        // for (int i = 0; i < WARP_SIZE; i++) {
        //     gathered[i] = __shfl_sync(0xffffffff, input[idx + i], i);
        // }
        
        /* YOUR CODE HERE */
        
        // Store gathered values (just first one for this exercise)
        output[blockIdx.x] = gathered[0];
    }
}

// ============================================================================
// KERNEL 6: Rotate Broadcast
 * Rotate values within warp and broadcast
 * TODO: Complete the rotation
// ============================================================================
__global__ void rotateBroadcast(float *input, float *output, int n, int rotateBy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Rotate value by rotateBy positions
        // int srcLane = (laneId - rotateBy + WARP_SIZE) % WARP_SIZE;
        // float rotated = __shfl_sync(0xffffffff, val, srcLane);
        
        /* YOUR CODE HERE */
        
        output[idx] = rotated;
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
        arr[i] = (float)i;
    }
}

int main() {
    printf("=== Warp Primitives Level 4: Warp Broadcast ===\n\n");
    
    const int N = 1024;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    initArraySequential(h_input, N);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = WARP_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Array size: %d\n", N);
    printf("Block size: %d (one warp)\n\n", blockSize);
    
    // Test 1: Broadcast from lane 0
    printf("Test 1: Broadcast from lane 0\n");
    broadcastFromLane0<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First warp results:\n");
    for (int i = 0; i < 8; i++) {
        printf("    Lane %d: %.2f (should be %.2f)\n", i, h_output[i], h_input[0]);
    }
    
    // Test 2: Broadcast from specific lane
    printf("\nTest 2: Broadcast from lane 5\n");
    broadcastFromLane<<<gridSize, blockSize>>>(d_input, d_output, N, 5);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First warp results:\n");
    for (int i = 0; i < 8; i++) {
        printf("    Lane %d: %.2f (should be %.2f)\n", i, h_output[i], h_input[5]);
    }
    
    // Test 3: Rotate broadcast
    printf("\nTest 3: Rotate by 4 positions\n");
    rotateBroadcast<<<gridSize, blockSize>>>(d_input, d_output, N, 4);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  First warp results:\n");
    for (int i = 0; i < 8; i++) {
        int expectedIdx = (i - 4 + WARP_SIZE) % WARP_SIZE;
        printf("    Lane %d: %.2f (should be %.2f)\n", i, h_output[i], (float)expectedIdx);
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- __shfl_sync(srcLane, val, srcLane) broadcasts to all\n");
    printf("- Multi-value broadcast: call __shfl multiple times\n");
    printf("- Combine with ballot for conditional broadcast\n");
    printf("- Rotation: calculate source lane index\n");
    printf("\nNext: Try level5_advanced_warp.cu for complex patterns\n");
    
    return 0;
}
