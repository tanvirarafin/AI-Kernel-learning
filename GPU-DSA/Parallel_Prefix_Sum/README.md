# Parallel Prefix Sum / Scan: Blelloch and Kogge-Stone Algorithms

## Overview

Parallel prefix sum (also known as scan) is a fundamental operation that computes all prefixes of an array in parallel. Given an array [a₀, a₁, a₂, ..., aₙ₋₁], the inclusive scan produces [a₀, a₀+a₁, a₀+a₁+a₂, ..., a₀+a₁+...+aₙ₋₁].

## Why Parallel Prefix Sum?

Sequential prefix sum takes O(n) time, but parallel algorithms can achieve better performance on GPUs. This operation is essential for:
- Compacting sparse arrays
- Histogram construction
- Stream compaction
- Allocation-free algorithms
- Many parallel algorithms that need cumulative operations

## Key Concepts

### Inclusive vs Exclusive Scan
- **Inclusive**: Each position contains the sum of all elements up to and including itself
  - Input: [1, 2, 3, 4]
  - Output: [1, 3, 6, 10]
  
- **Exclusive**: Each position contains the sum of all elements before it
  - Input: [1, 2, 3, 4]
  - Output: [0, 1, 3, 6]

### Associative Operators
Like reduction, scan works with any associative operator:
- Addition: [1, 2, 3, 4] → [1, 3, 6, 10]
- Maximum: [1, 3, 2, 4] → [1, 3, 3, 4]
- Multiplication: [1, 2, 3, 4] → [1, 2, 6, 24]

## Blelloch Algorithm (Work-Efficient)

### Algorithm Description
The Blelloch algorithm is work-efficient (O(n) total operations) and consists of two phases:
1. **Up-sweep (Reduction)**: Builds a binary tree of partial sums
2. **Down-sweep (Distribution)**: Distribates the sums back to create the scan

### Up-Sweep Phase
```
Original: [a0, a1, a2, a3, a4, a5, a6, a7]
Step 1:   [a0+a1,  *, a2+a3,  *, a4+a5,  *, a6+a7,  *]
Step 2:   [  *,  *, (a0+a1)+(a2+a3), *, (a4+a5)+(a6+a7), *, *, *]
Step 3:   [  *,  *,  *,  *, [(a0..a3)+(a4..a7)], *, *, *]
```

### Down-Sweep Phase
```
Step 3:   [  *,  *,  *,  *, 0, *, *, *]
Step 2:   [  *,  *, 0, *,  *, (a0..a3), *, *]
Step 1:   [0, (a0+a1),  *, (a2+a3), *, *, (a4..a6), *]
```

## Kogge-Stone Algorithm (Time-Optimal)

### Algorithm Description
The Kogge-Stone algorithm is time-optimal (O(log n) steps) but not work-efficient (O(n log n) total operations). It performs a parallel prefix operation in each step.

### Algorithm Steps
```
Original: [a0, a1, a2, a3, a4, a5, a6, a7]
Step 1:   [a0, a0+a1, a2, a2+a3, a4, a4+a5, a6, a6+a7]
Step 2:   [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, a4, a4+a5, a4+a5+a6, a4+a5+a6+a7]
Step 3:   [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, a0+...+a4, a0+...+a5, a0+...+a6, a0+...+a7]
```

## Step-by-Step Implementation Guide

### Step 1: Sequential Prefix Sum (for comparison)
```cpp
void sequentialScan(float* input, float* output, int n) {
    output[0] = 0; // exclusive scan
    for(int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i-1];
    }
}
```

### Step 2: Simple Kogge-Stone Implementation
```cpp
__global__ void koggeStoneScan(int* input, int* output, int n) {
    int tid = threadIdx.x;
    int p = 1;
    
    // Copy input to output
    output[tid] = input[tid];
    __syncthreads();
    
    // Perform Kogge-Stone steps
    for(int offset = 1; offset < n; offset *= 2) {
        int temp = (tid >= offset) ? output[tid] + output[tid - offset] : output[tid];
        __syncthreads();
        output[tid] = temp;
        __syncthreads();
    }
}
```

### Step 3: Optimized Kogge-Stone with Shared Memory
```cpp
__global__ void optimizedKoggeStone(int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2*thid] = input[2*thid];
    temp[2*thid+1] = input[2*thid+1];
    
    // Up-sweep
    for(int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if(thid == 0) temp[n-1] = 0;
    
    // Down-sweep
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results to device memory
    output[2*thid] = temp[2*thid];
    output[2*thid+1] = temp[2*thid+1];
}
```

### Step 4: Work-Efficient Blelloch Algorithm
```cpp
__global__ void blellochScan(int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2*thid] = input[2*thid];
    temp[2*thid+1] = input[2*thid+1];
    
    // Up-sweep (reduce) phase
    for(int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if(thid == 0) temp[n-1] = 0;
    
    // Down-sweep phase
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results to device memory
    output[2*thid] = temp[2*thid];
    output[2*thid+1] = temp[2*thid+1];
}
```

## Common Pitfalls and Solutions

### 1. Race Conditions
- **Problem**: Multiple threads accessing the same memory location simultaneously
- **Solution**: Proper synchronization with `__syncthreads()`

### 2. Memory Bank Conflicts
- **Problem**: Multiple threads accessing the same shared memory bank
- **Solution**: Careful indexing and memory layout

### 3. Array Size Limitations
- **Problem**: Algorithms designed for power-of-2 arrays
- **Solution**: Pad arrays or handle non-power-of-2 sizes separately

## Performance Considerations

### Work Efficiency
- Blelloch: O(n) total work, O(log n) steps
- Kogge-Stone: O(n log n) total work, O(log n) steps

### Memory Access Patterns
- Coalesced access is crucial for performance
- Minimize shared memory bank conflicts

### Scalability
- Consider the trade-off between time and work efficiency
- For large arrays, work-efficient algorithms are usually better

## Real-World Applications

- **Stream Compaction**: Removing elements that don't meet criteria
- **Histogram Construction**: Building histograms in parallel
- **Allocation-Free Algorithms**: Algorithms that don't require dynamic allocation
- **Graph Algorithms**: Connected components, shortest paths
- **Image Processing**: Integral images, morphological operations

## Summary

Parallel prefix sum is a fundamental building block for many GPU algorithms. The choice between Blelloch (work-efficient) and Kogge-Stone (time-optimal) depends on your specific use case. Understanding these algorithms is crucial for implementing efficient parallel algorithms that require cumulative operations.