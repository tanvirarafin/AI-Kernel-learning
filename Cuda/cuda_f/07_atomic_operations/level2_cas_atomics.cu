/*
 * Atomic Operations Level 2: Compare-And-Swap (CAS)
 *
 * EXERCISE: Master atomicCAS for complex lock-free algorithms.
 *
 * CONCEPTS:
 * - Compare-and-swap semantics
 * - Optimistic concurrency
 * - Retry loops
 * - Lock-free data structures
 *
 * SKILLS PRACTICED:
 * - atomicCAS usage
 * - CAS retry patterns
 * - Lock-free algorithms
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000

// ============================================================================
// KERNEL 1: Basic CAS Pattern
 * Classic compare-and-swap retry loop
 * TODO: Complete the CAS implementation
// ============================================================================
__global__ void basicCAS(float *array, float *target, int n, float newValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float* address = &array[idx];
        float assumed, old;
        
        // TODO: Implement CAS loop
        // Pattern:
        // assumed = *address;
        // do {
        //     old = assumed;
        //     if (assumed > *target) {  // Only update if greater than target
        //         assumed = newValue;
        //     }
        //     assumed = atomicCAS(address, old, assumed);
        // } while (assumed != old);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 2: Atomic Exchange
 * Use atomicExch to swap values
 * TODO: Complete the exchange pattern
// ============================================================================
__global__ void atomicExchange(float *array, float newValue, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Atomically exchange array[idx] with newValue
        // float old = atomicExch(&array[idx], newValue);
        // The old value is returned
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 3: CAS-based Float Min/Max
 * Implement atomic min/max for floats using CAS
 * TODO: Complete the float min/max with CAS
// ============================================================================
__global__ void casFloatMinMax(float *input, float *minOut, float *maxOut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // TODO: Update minimum using CAS
        // unsigned int* minAddr = (unsigned int*)minOut;
        // unsigned int old = *minAddr;
        // unsigned int assumed;
        // do {
        //     assumed = old;
        //     float oldVal = __int_as_float(assumed);
        //     if (val < oldVal) {
        //         old = atomicCAS(minAddr, assumed, __float_as_int(val));
        //     } else {
        //         break;  // No need to update
        //     }
        // } while (assumed != old);
        
        /* YOUR CODE HERE - Implement CAS-based min */
        
        // TODO: Similar for maximum
        /* YOUR CODE HERE - Implement CAS-based max */
    }
}

// ============================================================================
// KERNEL 4: Simple Spinlock
 * Implement a basic spinlock using CAS
 * TODO: Complete the lock/unlock pattern
// ============================================================================

// Lock state: 0 = unlocked, 1 = locked
__global__ void spinlockExample(int *lock, int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Acquire lock using CAS
        // int expected = 0;
        // while (atomicCAS(lock, expected, 1) != 0) {
        //     expected = 0;  // Reset for next iteration
        //     // Spin until lock is acquired
        // }
        
        // Critical section
        (*counter)++;
        
        // TODO: Release lock
        // atomicExch(lock, 0);  // Set to 0 (unlocked)
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 5: Lock-Free Stack Push
 * Implement a simple lock-free stack using CAS
 * TODO: Complete the stack push operation
// ============================================================================

// Stack node structure (simplified as indices)
__global__ void lockFreeStackPush(int *stackHead, int *stackNext, 
                                   int *stackData, int newData, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        int myNode = tid;  // Each thread has its own node
        stackData[myNode] = newData;
        
        // TODO: Implement lock-free stack push
        // Pattern:
        // do {
        //     stackNext[myNode] = *stackHead;  // Point to current head
        // } while (atomicCAS(stackHead, stackNext[myNode], myNode) != stackNext[myNode]);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 6: Test-And-Set Lock
 * Alternative lock implementation using atomicExch
 * TODO: Complete the test-and-set pattern
// ============================================================================
__global__ void testAndSetLock(int *lock, int *workDone, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Test-and-set using atomicExch
        // int oldLockState;
        // do {
        //     oldLockState = atomicExch(lock, 1);
        // } while (oldLockState == 1);  // Spin if was already locked
        
        // Critical section
        (*workDone)++;
        
        // TODO: Release lock
        // atomicExch(lock, 0);
        
        /* YOUR CODE HERE */
    }
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

int main() {
    printf("=== Atomic Operations Level 2: Compare-And-Swap ===\n\n");
    
    const int N = 10000;
    float *h_array = (float*)malloc(N * sizeof(float));
    int *h_lock = (int*)malloc(sizeof(int));
    int *h_counter = (int*)malloc(sizeof(int));
    
    // Test 1: Basic CAS
    printf("Test 1: Basic CAS pattern\n");
    initArray(h_array, N, 100.0f);
    float h_target = 50.0f;
    
    float *d_array, *d_target;
    int *d_lock, *d_counter;
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMalloc(&d_target, sizeof(float));
    cudaMalloc(&d_lock, sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &h_target, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    basicCAS<<<gridSize, blockSize>>>(d_array, d_target, N, 0.0f);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Values > target should be updated to 0\n");
    printf("  (Verify: h_array[0] = %.2f)\n", h_array[0]);
    
    // Test 2: Atomic exchange
    printf("\nTest 2: Atomic exchange\n");
    initArray(h_array, N, 1.0f);
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    
    atomicExchange<<<gridSize, blockSize>>>(d_array, 99.0f, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  All values should be 99.00\n");
    printf("  h_array[0] = %.2f\n", h_array[0]);
    
    // Test 3: Spinlock
    printf("\nTest 3: Spinlock example\n");
    int lock = 0;
    int counter = 0;
    cudaMemcpy(d_lock, &lock, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice);
    
    spinlockExample<<<gridSize, blockSize>>>(d_lock, d_counter, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Counter: %d (Expected: %d)\n", counter, N);
    if (counter == N) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Check lock implementation\n");
    }
    
    // Cleanup
    free(h_array);
    free(h_lock);
    free(h_counter);
    cudaFree(d_array);
    cudaFree(d_target);
    cudaFree(d_lock);
    cudaFree(d_counter);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- CAS enables complex lock-free algorithms\n");
    printf("- CAS retry loop: read, compute, CAS, retry if failed\n");
    printf("- Spinlocks use CAS or Exch for mutual exclusion\n");
    printf("- Lock-free stacks use CAS to update head pointer\n");
    printf("- CAS works with integers; use type-punning for floats\n");
    printf("\nNext: Try level3_histogram_atomics.cu for histogram patterns\n");
    
    return 0;
}
