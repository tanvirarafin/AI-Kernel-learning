/*
 * Atomic Operations Level 5: Advanced Atomic Patterns
 *
 * EXERCISE: Build complex lock-free data structures and algorithms.
 *
 * CONCEPTS:
 * - Lock-free queues
 * - Work stealing
 * - Concurrent linked lists
 * - Atomic memory pools
 * - Hazard pointers (concept)
 *
 * SKILLS PRACTICED:
 * - Complex CAS patterns
 * - Lock-free algorithms
 * - Memory ordering
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 10000
#define QUEUE_SIZE 1024

// ============================================================================
// KERNEL 1: Lock-Free Queue (Producer-Consumer)
 * Single producer, single consumer queue using atomics
 * TODO: Complete the queue operations
// ============================================================================
__global__ void lockFreeQueue(int *head, int *tail, int *queue, 
                               int *produced, int *consumed, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Producer side
        int myValue = tid + 1;
        
        // TODO: Enqueue (producer)
        // int oldTail = atomicAdd(tail, 1);
        // queue[oldTail % QUEUE_SIZE] = myValue;
        // atomicAdd(produced, 1);
        
        // Consumer side (different thread would do this)
        // TODO: Dequeue (consumer)
        // if (oldHead < *tail) {
        //     int value = queue[oldHead % QUEUE_SIZE];
        //     atomicAdd(consumed, 1);
        // }
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 2: Work Stealing Queue
 * Each thread has a deque, idle threads steal from others
 * TODO: Complete the work stealing pattern
// ============================================================================
__global__ void workStealing(int *workQueues, int *queueSizes,
                              int *results, int numQueues, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int myQueue = tid % numQueues;
    
    // TODO: Try to get work from own queue first
    // int myWork = atomicSub(&queueSizes[myQueue], 1);
    // if (myWork > 0) {
    //     // Process own work
    //     results[tid] = workQueues[myQueue * N + myWork - 1];
    // } else {
    //     // TODO: Steal from another queue
    //     int victim = (myQueue + 1) % numQueues;
    //     int stolen = atomicSub(&queueSizes[victim], 1);
    //     if (stolen > 0) {
    //         results[tid] = workQueues[victim * N + stolen - 1];
    //     }
    // }
    
    /* YOUR CODE HERE */
}

// ============================================================================
// KERNEL 3: Atomic Memory Pool
 * Lock-free memory allocation from a pool
 * TODO: Complete the memory pool
// ============================================================================
__global__ void atomicMemoryPool(int *freeList, int *nextFree,
                                  int *allocated, int *results, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // TODO: Allocate from free list
        // int myBlock = atomicAdd(freeList, 1);
        // if (myBlock < n) {
        //     allocated[tid] = myBlock;
        //     results[tid] = myBlock * 100;
        // }
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 4: Concurrent Counter with Reduction
 * Use atomics with reduction for efficient counting
 * TODO: Complete the hybrid approach
// ============================================================================
__global__ void concurrentCounter(float *input, int *blockCounts,
                                   int *globalCount, int n) {
    __shared__ int localCount;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) localCount = 0;
    __syncthreads();
    
    // Each thread counts locally
    if (idx < n && input[idx] > 0.5f) {
        atomicAdd(&localCount, 1);
    }
    __syncthreads();
    
    // Block representative adds to block count
    if (tid == 0) {
        blockCounts[blockIdx.x] = localCount;
    }
    __syncthreads();
    
    // TODO: One thread adds all block counts to global
    // if (tid == 0 && blockIdx.x == 0) {
    //     for (int i = 0; i < gridDim.x; i++) {
    //         atomicAdd(globalCount, blockCounts[i]);
    //     }
    // }
    
    /* YOUR CODE HERE */
}

// ============================================================================
// KERNEL 5: Atomic Linked List Insert
 * Lock-free linked list insertion
 * TODO: Complete the list insertion
// ============================================================================
__global__ void atomicLinkedList(int *listHead, int *nextPtr,
                                  int *nodeData, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        int myNode = tid;
        nodeData[myNode] = tid * 10;
        
        // TODO: Insert at head (lock-free)
        // do {
        //     nextPtr[myNode] = *listHead;
        // } while (atomicCAS(listHead, nextPtr[myNode], myNode) != nextPtr[myNode]);
        
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 6: Atomic Flag Operations
 * Use atomics for coordination flags
 * TODO: Complete the flag-based coordination
// ============================================================================
__global__ void atomicFlags(int *startFlag, int *doneFlags,
                             int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        // Wait for start signal
        // while (atomicAdd(startFlag, 0) == 0) {}
        
        // Do work
        data[tid] = tid * 2;
        
        // Signal completion
        // atomicAdd(doneFlags, 1);
        
        /* YOUR CODE HERE */
    }
}

// Utility functions
int main() {
    printf("=== Atomic Operations Level 5: Advanced Patterns ===\n\n");
    
    const int N = 1000;
    int *d_head, *d_tail, *d_queue, *d_produced, *d_consumed;
    int *d_freeList, *d_nextFree, *d_allocated;
    int h_zero = 0;
    
    cudaMalloc(&d_head, sizeof(int));
    cudaMalloc(&d_tail, sizeof(int));
    cudaMalloc(&d_queue, QUEUE_SIZE * sizeof(int));
    cudaMalloc(&d_produced, sizeof(int));
    cudaMalloc(&d_consumed, sizeof(int));
    cudaMalloc(&d_freeList, sizeof(int));
    cudaMalloc(&d_nextFree, N * sizeof(int));
    cudaMalloc(&d_allocated, N * sizeof(int));
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test 1: Lock-free queue
    printf("Test 1: Lock-free queue\n");
    cudaMemcpy(d_head, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tail, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_queue, 0, QUEUE_SIZE * sizeof(int));
    cudaMemcpy(d_produced, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_consumed, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    
    lockFreeQueue<<<gridSize, blockSize>>>(d_head, d_tail, d_queue,
                                            d_produced, d_consumed, N);
    cudaDeviceSynchronize();
    
    int h_produced, h_consumed;
    cudaMemcpy(&h_produced, d_produced, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_consumed, d_consumed, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Produced: %d, Consumed: %d\n", h_produced, h_consumed);
    
    // Test 2: Memory pool
    printf("\nTest 2: Atomic memory pool\n");
    h_zero = 0;
    cudaMemcpy(d_freeList, &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    
    atomicMemoryPool<<<gridSize, blockSize>>>(d_freeList, d_nextFree,
                                               d_allocated, d_allocated, N);
    cudaDeviceSynchronize();
    
    int h_freeList;
    cudaMemcpy(&h_freeList, d_freeList, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  Allocated blocks: %d\n", h_freeList);
    
    // Test 3: Atomic linked list
    printf("\nTest 3: Atomic linked list\n");
    int h_head = -1;
    cudaMemcpy(d_head, &h_head, sizeof(int), cudaMemcpyHostToDevice);
    
    int *d_nextPtr, *d_nodeData;
    cudaMalloc(&d_nextPtr, N * sizeof(int));
    cudaMalloc(&d_nodeData, N * sizeof(int));
    
    atomicLinkedList<<<gridSize, blockSize>>>(d_head, d_nextPtr, d_nodeData, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_head, d_head, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  List head: %d\n", h_head);
    
    // Cleanup
    cudaFree(d_head);
    cudaFree(d_tail);
    cudaFree(d_queue);
    cudaFree(d_produced);
    cudaFree(d_consumed);
    cudaFree(d_freeList);
    cudaFree(d_nextFree);
    cudaFree(d_allocated);
    cudaFree(d_nextPtr);
    cudaFree(d_nodeData);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Lock-free queues use CAS for head/tail updates\n");
    printf("- Work stealing balances load across threads\n");
    printf("- Memory pools provide lock-free allocation\n");
    printf("- Hybrid atomic+reduction reduces contention\n");
    printf("- Linked lists can be lock-free with CAS\n");
    printf("\n=== Atomic Operations Module Complete ===\n");
    printf("Next: Explore warp_primitives for intra-warp communication\n");
    
    return 0;
}
