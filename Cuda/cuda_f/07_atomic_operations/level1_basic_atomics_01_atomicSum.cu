/*
 * Atomic Operations Level 1: Basic Atomics - Kernel 1
 *
 * This kernel demonstrates basic atomic operations for thread-safe
 * accumulation, avoiding race conditions.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// ============================================================================
// KERNEL 1: Race Condition Example (WRONG - for demonstration)
// This kernel has a race condition - multiple threads write same location
// ============================================================================
__global__ void raceCondition(float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // BUG: Multiple threads may update output[0] simultaneously
    // This causes a race condition - results are undefined!
    if (idx < N) {
        output[0] = output[0] + input[idx];  // RACE CONDITION!
    }
}

// ============================================================================
// KERNEL 2: Atomic Sum
// Use atomicAdd to safely accumulate values
// ============================================================================
__global__ void atomicSum(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Use atomicAdd to safely accumulate
        atomicAdd(output, input[idx]);
    }
}

// ============================================================================
// KERNEL 3: Atomic Min/Max
// Find minimum and maximum values using atomics
// ============================================================================
__global__ void atomicMinMax(float *input, float *minOut, float *maxOut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = input[idx];
        
        // Use atomicMin and atomicMax (requires compute capability 3.5+)
        atomicMin(minOut, (int)val);
        atomicMax(maxOut, (int)val);
    }
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

void initArrayRandom(float *arr, int n, float maxVal) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % (int)(maxVal * 100)) / 100.0f;
    }
}

int main() {
    printf("=== Atomic Operations Level 1: Basic Atomics ===\n\n");

    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));

    // Test 1: Demonstrate race condition (will give wrong answer)
    printf("Test 1: Race condition demonstration (WRONG RESULTS)\n");
    initArray(h_input, N, 1.0f);

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaMemset(d_output, 0, sizeof(float));
    raceCondition<<<gridSize, blockSize>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f (Expected: %d, will be WRONG due to race)\n",
           *h_output, N);
    printf("  This demonstrates why we need atomics!\n");

    // Test 2: Atomic sum
    printf("\nTest 2: Atomic sum\n");
    cudaMemset(d_output, 0, sizeof(float));
    atomicSum<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f (Expected: %d)\n", *h_output, N);
    if ((int)*h_output == N) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Use atomicAdd\n");
    }

    // Test 3: Atomic min/max
    printf("\nTest 3: Atomic min/max\n");
    initArrayRandom(h_input, N, 1000.0f);
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int h_min = 999999, h_max = 0;
    int *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);

    atomicMinMax<<<gridSize, blockSize>>>(d_input, d_min, d_max, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Min: %d, Max: %d\n", h_min, h_max);
    printf("  (Should be close to 0 and 1000)\n");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_min);
    cudaFree(d_max);

    printf("\n=== Key Takeaways ===\n");
    printf("- Race conditions cause undefined behavior\n");
    printf("- atomicAdd safely accumulates values\n");
    printf("- atomicMin/atomicMax find extrema\n");
    printf("- Atomics serialize access - use sparingly\n");

    return 0;
}
