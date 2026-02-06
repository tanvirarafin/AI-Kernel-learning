# Bitonic Sort: Parallel Sorting Algorithm

## Overview

Bitonic sort is a parallel sorting algorithm that works by creating and merging bitonic sequences. A bitonic sequence is a sequence that first increases and then decreases (or vice versa), or can be circularly shifted to form such a sequence. Unlike other sorting algorithms, bitonic sort has a fixed comparison pattern, making it ideal for parallel execution on GPUs.

## Why Bitonic Sort on GPUs?

While bitonic sort has O(n log²n) time complexity (worse than O(n log n) of algorithms like quicksort), it has several advantages for parallel execution:
- Fixed comparison pattern regardless of input data
- No data-dependent branching
- Regular memory access patterns
- Highly parallelizable structure
- Predictable execution time

## Key Concepts

### Bitonic Sequence
A bitonic sequence is a sequence that:
1. Monotonically increases and then monotonically decreases, OR
2. Can be circularly shifted to form such a sequence

Examples:
- [1, 2, 3, 4, 3, 2, 1] - increases then decreases
- [4, 3, 2, 1, 2, 3, 4] - can be shifted to [2, 3, 4, 4, 3, 2, 1]

### Bitonic Split
The fundamental operation in bitonic sort is the bitonic split, which partitions a bitonic sequence into two subsequences:
- One containing all smaller elements
- One containing all larger elements
- Both subsequences remain bitonic
- All elements in the first subsequence are smaller than all elements in the second

### Bitonic Merge
The process of converting a bitonic sequence into a monotonic sequence (either all increasing or all decreasing).

## Algorithm Structure

### Phase 1: Create Bitonic Sequence
Convert the input array into a bitonic sequence by recursively sorting subarrays in opposite directions.

### Phase 2: Convert to Monotonic
Convert the bitonic sequence into a fully sorted sequence using bitonic merge.

## Step-by-Step Implementation Guide

### Step 1: Sequential Bitonic Sort (for understanding)
```cpp
void compareAndSwap(int arr[], int i, int j, bool ascending) {
    if ((arr[i] > arr[j]) == !ascending) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void bitonicMerge(int arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        
        // Compare and swap elements in first half with corresponding elements in second half
        for (int i = low; i < low + k; i++) {
            compareAndSwap(arr, i, i + k, ascending);
        }
        
        // Recursively sort both halves
        bitonicMerge(arr, low, k, ascending);
        bitonicMerge(arr, low + k, k, ascending);
    }
}

void bitonicSort(int arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        
        // Sort first half in ascending order
        bitonicSort(arr, low, k, true);
        
        // Sort second half in descending order
        bitonicSort(arr, low + k, k, false);
        
        // Merge the two halves into a bitonic sequence
        bitonicMerge(arr, low, cnt, ascending);
    }
}

void sequentialBitonicSort(int arr[], int n) {
    bitonicSort(arr, 0, n, true);
}
```

### Step 2: Basic Parallel Bitonic Sort
```cpp
__global__ void bitonicSortStep(int* arr, int n, int k, int j) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = idx ^ j;  // XOR operation
    
    if(ixj > idx && idx < n && ixj < n) {
        if((idx & k) == 0) {
            // Sort in ascending order
            if(arr[idx] > arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if(arr[idx] < arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void gpuBitonicSort(int* d_arr, int n) {
    // Ensure n is a power of 2
    int powerOf2 = 1;
    while(powerOf2 < n) powerOf2 <<= 1;
    
    dim3 block(256);
    dim3 grid((powerOf2 + block.x - 1) / block.x);
    
    // Perform bitonic sort
    for(int k = 2; k <= powerOf2; k <<= 1) {
        for(int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<grid, block>>>(d_arr, n, k, j);
            cudaDeviceSynchronize();
        }
    }
}
```

### Step 3: Optimized GPU Bitonic Sort with Shared Memory
```cpp
__global__ void optimizedBitonicSort(int* input, int* output, int n, int k, int j) {
    extern __shared__ int sharedData[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load data into shared memory
    if(idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = INT_MAX; // Fill with max value for out-of-bounds
    }
    __syncthreads();
    
    // Perform bitonic sort step
    int ixj = tid ^ j;  // XOR operation within block
    
    if(ixj > tid && tid < blockDim.x) {
        bool ascending = (tid & k) == 0;
        
        if(ascending) {
            // Sort in ascending order
            if(sharedData[tid] > sharedData[ixj]) {
                int temp = sharedData[tid];
                sharedData[tid] = sharedData[ixj];
                sharedData[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if(sharedData[tid] < sharedData[ixj]) {
                int temp = sharedData[tid];
                sharedData[tid] = sharedData[ixj];
                sharedData[ixj] = temp;
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory
    if(idx < n) {
        output[idx] = sharedData[tid];
    }
}
```

### Step 4: Complete GPU Bitonic Sort Implementation
```cpp
__global__ void bitonicCompare(int* data, int size, int k, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;
    
    if(ixj > idx && idx < size && ixj < size) {
        // Determine direction based on k
        bool ascending = (idx & k) == 0;
        
        // Compare and swap if needed
        if(ascending) {
            if(data[idx] > data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if(data[idx] < data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void completeBitonicSort(int* d_data, int n) {
    // Ensure n is a power of 2
    int size = 1;
    while(size < n) size <<= 1;
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // Main sorting loop
    for(int k = 2; k <= size; k <<= 1) {
        for(int j = k >> 1; j > 0; j >>= 1) {
            bitonicCompare<<<grid, block>>>(d_data, n, k, j);
            
            // Synchronize after each step to ensure correctness
            cudaDeviceSynchronize();
        }
    }
}
```

## Common Pitfalls and Solutions

### 1. Array Size Requirements
- **Problem**: Bitonic sort requires array size to be a power of 2
- **Solution**: Pad array with maximum values or use hybrid approaches

### 2. Memory Access Patterns
- **Problem**: Irregular memory access can hurt performance
- **Solution**: Optimize for coalesced access where possible

### 3. Synchronization
- **Problem**: Incorrect synchronization can lead to race conditions
- **Solution**: Use proper synchronization between phases

### 4. Branch Divergence
- **Problem**: Different threads taking different paths
- **Solution**: Minimize conditional statements within warps

## Performance Considerations

### Memory Bandwidth
- Bitonic sort has high memory bandwidth requirements
- Optimize memory access patterns for best performance

### Occupancy
- Balance shared memory usage with thread count
- Consider the trade-off between block size and memory usage

### Problem Size
- More efficient for moderate-sized arrays
- For very large arrays, other algorithms may be more suitable

### GPU Architecture
- Performance varies based on memory subsystem
- Consider the specific GPU's capabilities

## Real-World Applications

- **Graphics**: Sorting for rendering algorithms
- **Scientific Computing**: Sorting particles or elements
- **Database Systems**: Parallel sorting of records
- **Machine Learning**: Sorting indices for sparse operations
- **Computational Geometry**: Sorting points or geometric objects

## Advanced Techniques

### Hybrid Approaches
- Combine with other algorithms for better performance
- Use bitonic sort for smaller subproblems

### Optimized Memory Access
- Use texture memory for read-only data
- Implement cache-friendly access patterns

### Variable-Length Sorting
- Adapt algorithm for non-power-of-2 sizes
- Use padding or special handling for boundaries

## Summary

Bitonic sort is a unique parallel sorting algorithm with a fixed comparison pattern, making it well-suited for GPU implementation. While it has worse theoretical complexity than comparison-based algorithms, its regular structure and predictable execution make it valuable for certain applications. Understanding its strengths and limitations is crucial for effective GPU programming.