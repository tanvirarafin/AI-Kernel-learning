# Radix Sort: LSB and MSB Approaches

## Overview

Radix sort is a non-comparative sorting algorithm that sorts elements by processing individual digits or bits. Unlike comparison-based algorithms (like quicksort), radix sort can achieve O(k*n) time complexity, where k is the number of digits and n is the number of elements. This makes it highly efficient for sorting integers and fixed-length keys on GPUs.

## Why Radix Sort on GPUs?

Traditional comparison-based sorting algorithms are difficult to parallelize effectively. Radix sort, however, is naturally suited for parallel execution because:
- Each digit processing step can be parallelized
- No complex comparisons needed
- Regular memory access patterns
- Predictable execution paths (no branching)

## Key Concepts

### Least Significant Digit (LSD) vs Most Significant Digit (MSD)

**LSD Radix Sort:**
- Processes digits from right to left (least significant to most significant)
- Stable sorting algorithm
- Better for shorter keys
- Processes all elements at each digit position

**MSD Radix Sort:**
- Processes digits from left to right (most significant to least significant)
- Can be faster for longer keys
- Naturally divides data into buckets
- More complex to implement in parallel

### Stable Sorting
A sorting algorithm is stable if equal elements maintain their relative order from the input. LSD radix sort is inherently stable, which is important for many applications.

### Bit vs Decimal Radix
- **Binary Radix**: Processes one bit at a time (radix = 2)
- **Decimal Radix**: Processes one decimal digit at a time (radix = 10)
- **Power-of-2 Radix**: Processes multiple bits (e.g., radix = 256 for 8 bits)

## Key Operations in GPU Radix Sort

### Histogram (Counting)
Count how many elements belong to each bucket for the current digit being processed.

### Exclusive Scan (Prefix Sum)
Compute starting positions for each bucket in the output array.

### Scatter
Move elements to their correct positions based on the computed positions.

## Step-by-Step Implementation Guide

### Step 1: Sequential LSD Radix Sort (for comparison)
```cpp
void sequentialRadixSort(int* input, int* output, int n, int numBits) {
    int* temp = new int[n];
    int* src = input;
    int* dst = temp;
    
    // Process each bit position
    for(int shift = 0; shift < numBits; shift += 1) {
        int mask = 1 << shift;
        
        // Count 0s and 1s
        int count[2] = {0, 0};
        for(int i = 0; i < n; i++) {
            count[(src[i] & mask) ? 1 : 0]++;
        }
        
        // Compute starting positions
        int startPos[2] = {0, count[0]};
        
        // Scatter elements to destination
        for(int i = 0; i < n; i++) {
            int digit = (src[i] & mask) ? 1 : 0;
            dst[startPos[digit]++] = src[i];
        }
        
        // Swap source and destination
        int* tmp = src;
        src = dst;
        dst = tmp;
    }
    
    // If result is in temp, copy back to input
    if(src != input) {
        for(int i = 0; i < n; i++) {
            input[i] = temp[i];
        }
    }
    
    delete[] temp;
}
```

### Step 2: Parallel Histogram (for digit counting)
```cpp
__global__ void countDigits(int* input, int* counts, int n, int shift, int mask) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Shared memory for block-level counts
    extern __shared__ int sCounts[];
    
    // Initialize shared memory
    for(int i = tid; i < 2; i += blockDim.x) {
        sCounts[i] = 0;
    }
    __syncthreads();
    
    // Count digits in parallel
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        atomicAdd(&sCounts[digit], 1);
    }
    __syncthreads();
    
    // Write block counts to global memory
    for(int i = tid; i < 2; i += blockDim.x) {
        counts[bid * 2 + i] = sCounts[i];
    }
}
```

### Step 3: Parallel Prefix Sum (for position calculation)
```cpp
__global__ void prefixSum(int* counts, int* prefixSums, int numBlocks) {
    // Assuming 2 bins per block (for binary radix)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if(bid == 0 && tid < 2) {
        // Calculate prefix sums across blocks
        int total = 0;
        for(int i = 0; i < numBlocks; i++) {
            int temp = counts[i * 2 + tid];
            prefixSums[i * 2 + tid] = total;
            total += temp;
        }
    }
}
```

### Step 4: Parallel Scatter (for rearranging elements)
```cpp
__global__ void scatter(int* input, int* output, int* prefixSums, 
                       int n, int shift, int mask) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if(idx < n) {
        int value = input[idx];
        int digit = (value >> shift) & mask;
        
        // Calculate position using prefix sum
        int pos = prefixSums[bid * 2 + digit];
        
        // Add offset within block
        extern __shared__ int sCounts[];
        sCounts[digit] = 1;
        __syncthreads();
        
        // Use exclusive scan within block to get exact position
        for(int i = 0; i < tid; i++) {
            if(((input[bid * blockDim.x + i] >> shift) & mask) == digit) {
                pos++;
            }
        }
        
        output[pos] = value;
    }
}
```

### Step 5: Complete GPU Radix Sort Kernel
```cpp
__global__ void gpuRadixSortPass(int* input, int* output, int* tempStorage, 
                                int n, int shift, int mask) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Shared memory for counts and positions
    extern __shared__ int sharedMem[];
    int* sCounts = sharedMem;
    int* sPositions = &sharedMem[2]; // Assuming 2 bins
    
    // Initialize counts
    for(int i = tid; i < 2; i += blockDim.x) {
        sCounts[i] = 0;
    }
    __syncthreads();
    
    // Count digits in this block
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        atomicAdd(&sCounts[digit], 1);
    }
    __syncthreads();
    
    // Store block-level counts in global memory
    if(tid < 2) {
        tempStorage[bid * 2 + tid] = sCounts[tid];
    }
    __syncthreads();
    
    // After global prefix sum is computed externally:
    __syncthreads();
    
    // Calculate local positions within block
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        
        // Reset position counters
        if(tid < 2) {
            sPositions[digit] = 0;
        }
        __syncthreads();
        
        // Calculate local offset
        if(digit == 0) {
            // Count how many 0s come before this element in the block
            for(int i = 0; i < tid && (bid * blockDim.x + i) < n; i++) {
                if(((input[bid * blockDim.x + i] >> shift) & mask) == 0) {
                    sPositions[0]++;
                }
            }
        } else {
            // Count how many 1s come before this element in the block
            for(int i = 0; i < tid && (bid * blockDim.x + i) < n; i++) {
                if(((input[bid * blockDim.x + i] >> shift) & mask) == 1) {
                    sPositions[1]++;
                }
            }
        }
        __syncthreads();
        
        // Calculate final position
        int globalPos = tempStorage[bid * 2 + digit] + sPositions[digit];
        output[globalPos] = input[idx];
    }
}
```

## Common Pitfalls and Solutions

### 1. Memory Bank Conflicts
- **Problem**: Multiple threads accessing the same shared memory bank
- **Solution**: Use proper memory layout and padding

### 2. Load Imbalance
- **Problem**: Uneven distribution of elements across buckets
- **Solution**: Use balanced partitioning techniques

### 3. Bitonic Sort Alternative
- **Problem**: Radix sort may not be optimal for all data distributions
- **Solution**: Consider hybrid approaches with bitonic sort

### 4. Signed Numbers
- **Problem**: Negative numbers have leading 1s in two's complement
- **Solution**: Handle sign bit specially or convert to unsigned

## Performance Considerations

### Radix Size
- Larger radix reduces number of passes but increases memory usage
- Power-of-2 radices (256, 512) work well with GPU architecture

### Memory Access Patterns
- Coalesced access is crucial for performance
- Consider using texture memory for lookup tables

### Occupancy
- Balance between shared memory usage and thread count
- Optimize block size for your specific GPU

### Data Characteristics
- Performance varies significantly with data distribution
- Nearly sorted data may not benefit as much from radix sort

## Real-World Applications

- **Database Systems**: Sorting large datasets efficiently
- **Graphics**: Depth sorting for rendering
- **Scientific Computing**: Sorting particles in simulations
- **Machine Learning**: Sorting indices for sparse operations
- **Cryptography**: Sorting large integer arrays

## Advanced Techniques

### Hybrid Sorting
Combine radix sort with other algorithms:
- Use quicksort for small arrays
- Switch to insertion sort for nearly sorted data

### Optimized Radix Sizes
- Use different radix sizes based on data characteristics
- Adaptive radix selection during execution

## Summary

GPU radix sort is a powerful technique for efficiently sorting large arrays of integers. By leveraging parallel histogramming, prefix sum, and scatter operations, we can achieve excellent performance on GPU hardware. The choice of radix size and implementation details significantly impacts performance, so careful tuning is necessary for optimal results.