/*
 * Atomic Operations Level 1: Basic Atomics
 *
 * EXERCISE: Learn fundamental atomic operations for thread-safe updates.
 *
 * CONCEPTS:
 * - Race conditions without atomics
 * - Atomic add, min, max
 * - Atomic operation semantics
 * - Performance implications
 *
 * SKILLS PRACTICED:
 * - atomicAdd usage
 * - atomicMin/atomicMax
 * - Avoiding race conditions
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// ============================================================================
// KERNEL 1: Race Condition (Buggy - DO NOT USE)
 * This kernel has a race condition - multiple threads write same location
 * WARNING: This demonstrates what NOT to do!
// ============================================================================
__global__ void raceCondition(float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BUG: Multiple threads may update output[0] simultaneously
    // This causes a race condition - results are undefined!
    if (idx < N) {
        // TODO: This is WRONG - fix it using atomicAdd
        output[0] = output[0] + input[idx];  // RACE CONDITION!
    }
}

// ============================================================================
// KERNEL 2: Atomic Add for Summation
 * Use atomicAdd to safely accumulate values
 * TODO: Complete the atomic summation
// ============================================================================
__global__ void atomicSum(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Use atomicAdd to safely accumulate
        // atomicAdd(output, input[idx]);
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 3: Atomic Min/Max for Reduction
 * Find minimum and maximum values using atomics
 * TODO: Complete the atomic min/max reduction
// ============================================================================
__global__ void atomicMinMax(float *input, float *minOut, float *maxOut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Use atomicMin and atomicMax to find global min/max
        // Note: atomicMin/atomicMax for floats requires compute capability 3.5+
        // For older GPUs, use atomicCAS (covered in next level)
        
        // atomicMin(minOut, val);  // Requires proper initialization
        // atomicMax(maxOut, val);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 4: Atomic Increment for Counting
 * Count elements satisfying a condition
 * TODO: Complete the atomic counter
// ============================================================================
__global__ void atomicCount(float *input, int *count, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Atomically increment count if condition is met
        if (input[idx] > threshold) {
            // atomicAdd(count, 1);
            /* YOUR CODE HERE */
        }
    }
}

// ============================================================================
// KERNEL 5: Atomic Sub and Exchange
 * Use atomicSub and atomicExch for specific patterns
 * TODO: Complete the atomic operations
// ============================================================================
__global__ void atomicSubExch(float *buffer, int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Atomically decrement counter and get unique index
        // int myIdx = atomicSub(counter, 1);
        // buffer[myIdx] = idx;
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 6: Atomic Float Add (Manual Implementation)
 * Implement atomicAdd for floats using CAS (for older GPUs)
 * TODO: Complete the CAS-based float atomic
// ============================================================================
__global__ void atomicFloatAddCAS(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Implement atomic add for float using atomicCAS
        // This is needed for compute capability < 3.5
        
        // Pseudocode:
        // unsigned int* address_as_uint = (unsigned int*)output;
        // unsigned int old = *address_as_uint;
        // unsigned int assumed;
        // do {
        //     assumed = old;
        //     float sum = __int_as_float(assumed) + val;
        //     old = atomicCAS(address_as_uint, assumed, __float_as_int(sum));
        // } while (assumed != old);
        
        /* YOUR CODE HERE */
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
    int *h_count = (int*)malloc(sizeof(int));
    
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
    
    float h_min = 999999.0f, h_max = 0.0f;
    float *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    
    atomicMinMax<<<gridSize, blockSize>>>(d_input, d_min, d_max, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Min: %.2f, Max: %.2f\n", h_min, h_max);
    printf("  (Verify manually - should be close to 0.00 and 1000.00)\n");
    
    // Test 4: Atomic count
    printf("\nTest 4: Atomic count (elements > 500)\n");
    const float THRESHOLD = 500.0f;
    int h_count_val = 0;
    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &h_count_val, sizeof(int), cudaMemcpyHostToDevice);
    
    atomicCount<<<gridSize, blockSize>>>(d_input, d_count, N, THRESHOLD);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_count_val, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Count: %d elements > %.1f\n", h_count_val, THRESHOLD);
    
    // Verify
    int expected_count = 0;
    for (int i = 0; i < N; i++) {
        if (h_input[i] > THRESHOLD) expected_count++;
    }
    if (h_count_val == expected_count) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Expected %d\n", expected_count);
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_count);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_count);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Race conditions cause undefined behavior\n");
    printf("- atomicAdd safely accumulates values\n");
    printf("- atomicMin/atomicMax find extrema\n");
    printf("- Atomics serialize access - use sparingly\n");
    printf("- For sums, consider reduction instead (faster)\n");
    printf("\nNext: Try level2_cas_atomics.cu for compare-and-swap\n");
    
    return 0;
}
