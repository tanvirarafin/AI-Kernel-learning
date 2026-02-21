/*
 * Atomic Operations Level 4: Locks and Mutexes
 *
 * EXERCISE: Implement synchronization primitives using atomics.
 *
 * CONCEPTS:
 * - Mutual exclusion
 * - Spinlocks
 * - Mutex implementation
 * - Critical sections
 * - Deadlock avoidance
 *
 * SKILLS PRACTICED:
 * - Lock acquisition/release
 * - CAS-based locks
 * - Critical section protection
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000

// ============================================================================
// KERNEL 1: Simple Mutex Lock
 * Basic mutex using atomicCAS
 * TODO: Complete the lock/unlock implementation
// ============================================================================

// Mutex: 0 = unlocked, 1 = locked
__global__ void mutexLock(int *mutex, int *sharedCounter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Acquire lock
        // int expected = 0;
        // while (atomicCAS(mutex, expected, 1) != 0) {
        //     expected = 0;  // Reset for retry
        // }
        
        // Critical section: increment shared counter
        (*sharedCounter)++;
        
        // TODO: Release lock
        // atomicExch(mutex, 0);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 2: Ticket Lock (Fair Locking)
 * Implement a fair ticket-based lock
 * TODO: Complete the ticket lock
// ============================================================================
__global__ void ticketLock(int *nextTicket, int *nowServing, 
                           int *myTicket, int *workDone, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Get my ticket number
        // myTicket[idx] = atomicAdd(nextTicket, 1);
        
        // TODO: Wait for my turn
        // while (atomicAdd(nowServing, 0) != myTicket[idx]) {
        //     // Spin
        // }
        
        // Critical section
        (*workDone)++;
        
        // TODO: Release lock - increment nowServing
        // atomicAdd(nowServing, 1);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 3: Reader-Writer Lock (Read-Preferred)
 * Allow multiple readers or single writer
 * TODO: Complete the reader-writer lock
// ============================================================================
__global__ void readerWriterLock(int *readCount, int *writeLock,
                                  int *readData, int *results,
                                  int n, int isWriter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (isWriter) {
            // TODO: Writer acquisition
            // Acquire write lock (exclusive)
            // while (atomicCAS(writeLock, 0, 1) != 0) {}
            
            // Write operation
            *readData = idx;
            
            // TODO: Release write lock
            // atomicExch(writeLock, 0);
        } else {
            // TODO: Reader acquisition
            // int myRead = atomicAdd(readCount, 1);
            // if (myRead == 0) {
            //     // First reader - acquire write lock to block writers
            //     while (atomicCAS(writeLock, 0, 1) != 0) {}
            // }
            
            // Read operation
            results[idx] = *readData;
            
            // TODO: Reader release
            // int lastRead = atomicSub(readCount, 1);
            // if (lastRead == 1) {
            //     // Last reader - release write lock
            //     atomicExch(writeLock, 0);
            // }
            
            /* YOUR CODE HERE */
        }
    }
}

// ============================================================================
// KERNEL 4: Semaphore
 * Counting semaphore using atomics
 * TODO: Complete the semaphore implementation
// ============================================================================
__global__ void semaphore(int *semaphore, int *resource, int *results,
                          int n, int maxResources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: P operation (wait/acquire)
        // int available;
        // do {
        //     available = atomicAdd(semaphore, 0);  // Read current value
        //     if (available > 0) {
        //         // Try to decrement
        //     }
        // } while (atomicCAS(semaphore, available, available - 1) != available);
        
        // Use resource
        results[idx] = 1;
        
        // TODO: V operation (signal/release)
        // atomicAdd(semaphore, 1);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 5: Barrier Using Atomics
 * Implement a thread barrier using atomics
 * TODO: Complete the barrier
// ============================================================================
__global__ void atomicBarrier(int *barrierCount, int *data, int n, int totalThreads) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int localBarrier;
    
    if (idx < n) {
        data[idx] = idx * 2;  // Some work
    }
    __syncthreads();
    
    // TODO: Arrive at barrier
    // if (tid == 0) {
    //     *barrierCount = 0;  // Reset for this block
    // }
    // __syncthreads();
    
    // int oldVal = atomicAdd(barrierCount, 1);
    
    // TODO: Wait for all threads
    // while (*barrierCount < totalThreads) {
    //     // Spin
    // }
    
    __syncthreads();
    
    // Continue after barrier
    if (idx < n) {
        data[idx] = data[idx] + 1;
    }
}

// ============================================================================
// KERNEL 6: Reentrant Lock (with recursion count)
 * Lock that can be acquired multiple times by same thread
 * TODO: Complete the reentrant lock
// ============================================================================
__global__ void reentrantLock(int *lockOwner, int *lockCount, 
                               int *recursiveWork, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // TODO: Check if we own the lock
        // int myBlockId = blockIdx.x;
        // if (atomicAdd(lockOwner, 0) == myBlockId) {
        //     // We already own it - increment count
        //     atomicAdd(lockCount, 1);
        // } else {
        //     // Acquire lock
        //     while (atomicCAS(lockOwner, -1, myBlockId) != -1) {}
        //     lockCount = 1;
        // }
        
        // Recursive work
        (*recursiveWork)++;
        
        // TODO: Release (decrement count, release if zero)
        /* YOUR CODE HERE */
    }
}

// Utility functions
int main() {
    printf("=== Atomic Operations Level 4: Locks and Mutexes ===\n\n");
    
    const int N = 10000;
    int *d_mutex, *d_counter, *d_ticket, *d_nowServing;
    int *d_workDone, *d_readCount, *d_writeLock, *d_readData;
    int h_counter = 0, h_workDone = 0;
    
    cudaMalloc(&d_mutex, sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_ticket, sizeof(int));
    cudaMalloc(&d_nowServing, sizeof(int));
    cudaMalloc(&d_workDone, sizeof(int));
    cudaMalloc(&d_readCount, sizeof(int));
    cudaMalloc(&d_writeLock, sizeof(int));
    cudaMalloc(&d_readData, sizeof(int));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test 1: Mutex
    printf("Test 1: Mutex lock\n");
    cudaMemset(d_mutex, 0, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMemcpy(d_mutex, &h_counter, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);
    
    mutexLock<<<gridSize, blockSize>>>(d_mutex, d_counter, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Counter: %d (Expected: %d)\n", h_counter, N);
    if (h_counter == N) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Check lock implementation\n");
    }
    
    // Test 2: Ticket lock
    printf("\nTest 2: Ticket lock (fair locking)\n");
    int h_zero = 0;
    cudaMemcpy(d_ticket, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nowServing, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_workDone, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_myTicket;
    cudaMalloc(&d_myTicket, N * sizeof(int));
    
    ticketLock<<<gridSize, blockSize>>>(d_ticket, d_nowServing, d_myTicket, d_workDone, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_workDone, d_workDone, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Work done: %d (Expected: %d)\n", h_workDone, N);
    
    // Cleanup
    cudaFree(d_mutex);
    cudaFree(d_counter);
    cudaFree(d_ticket);
    cudaFree(d_nowServing);
    cudaFree(d_workDone);
    cudaFree(d_readCount);
    cudaFree(d_writeLock);
    cudaFree(d_readData);
    cudaFree(d_myTicket);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Mutex: Simple exclusive lock using CAS\n");
    printf("- Ticket lock: Fair FIFO ordering\n");
    printf("- Reader-writer: Multiple readers OR single writer\n");
    printf("- Semaphore: Counting resource access\n");
    printf("- Barrier: Synchronize thread progress\n");
    printf("\nNext: Try level5_advanced_patterns.cu for complex structures\n");
    
    return 0;
}
