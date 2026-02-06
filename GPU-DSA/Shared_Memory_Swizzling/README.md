# Shared Memory Swizzling: XOR Layouts to Avoid Bank Conflicts

## Overview

Shared memory swizzling is a technique used in GPU programming to rearrange data in shared memory to avoid bank conflicts. GPUs organize shared memory into banks, and when multiple threads access different addresses in the same bank simultaneously, it causes serialization and performance degradation. Swizzling techniques, particularly XOR-based layouts, help distribute memory accesses across different banks.

## Why Shared Memory Swizzling?

GPUs typically organize shared memory into 32 banks (matching warp size). When threads in a warp access shared memory:
- If multiple threads access different addresses in the same bank → bank conflict
- If threads access different banks → simultaneous access possible
- If all threads access the same address → broadcast (efficient)

Bank conflicts can severely degrade performance, making swizzling essential for optimizing shared memory access patterns.

## Key Concepts

### Shared Memory Banks
- Modern GPUs typically have 32 shared memory banks
- Each bank can service one access per cycle
- Bank number determined by address: `(address / 4) % 32` (for 32-bit words)

### Bank Conflicts
- **No Conflict**: All threads access different banks
- **Broadcast**: All threads access the same address
- **Conflict**: Multiple threads access the same bank
  - 2-way to 32-way conflicts possible
  - Causes serialization of accesses

### Swizzling
Rearranging data layout to spread accesses across different banks:
- **Address Swizzling**: Modify addresses before access
- **Data Swizzling**: Rearrange data during storage
- **XOR Swizzling**: Use XOR operation to remap addresses

## Types of Access Patterns

### Coalesced Access
- Consecutive threads access consecutive memory locations
- Usually efficient but can cause bank conflicts in shared memory

### Strided Access
- Threads access memory with a fixed stride
- Can cause severe bank conflicts depending on stride

### Transposed Access
- Access pattern changes from row-major to column-major
- Often causes bank conflicts without swizzling

## Step-by-Step Implementation Guide

### Step 1: Understanding Bank Conflicts
```cpp
// Example of problematic access pattern
__global__ void problematicAccess(float* input, float* output) {
    __shared__ float sdata[128];  // 128 floats = 512 bytes
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = input[tid];
    __syncthreads();
    
    // Problematic access: diagonal access causes bank conflicts
    // In a 32-thread warp, threads 0, 32, 64, 96 access the same bank
    float value = sdata[tid * 2];  // Stride-2 access
    output[tid] = value;
}
```

### Step 2: Basic Swizzling Without XOR
```cpp
#define BANK_NUM 32
#define BANK_WIDTH 4  // 4-byte words

// Function to calculate swizzled address
__device__ int getSwizzledAddr(int addr) {
    // Simple swizzling: add bank index to address
    int bank = (addr / BANK_WIDTH) % BANK_NUM;
    return addr + bank;
}

__global__ void basicSwizzledAccess(float* input, float* output) {
    // Adjust shared memory size to account for swizzling
    __shared__ float sdata[128 + BANK_NUM];  // Add padding
    
    int tid = threadIdx.x;
    
    // Load with swizzled address
    sdata[getSwizzledAddr(tid)] = input[tid];
    __syncthreads();
    
    // Access with swizzled address
    float value = sdata[getSwizzledAddr(tid * 2)];
    output[tid] = value;
}
```

### Step 3: XOR-Based Swizzling
```cpp
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5  // log2(NUM_BANKS)

// XOR swizzling function
__device__ int xorSwizzle(int addr) {
    // XOR the address with the bank index to spread accesses
    int bank = (addr / 4) % NUM_BANKS;  // 4-byte words
    return addr ^ (bank << 2);  // Shift bank to align with word boundary
}

__global__ void xorSwizzledAccess(float* input, float* output) {
    // Need extra space for swizzling
    __shared__ float sdata[128 + NUM_BANKS * 4];
    
    int tid = threadIdx.x;
    
    // Store with swizzled address
    sdata[xorSwizzle(tid)] = input[tid];
    __syncthreads();
    
    // Load with swizzled address
    float value = sdata[xorSwizzle(tid * 2)];
    output[tid] = value;
}
```

### Step 4: Optimized Swizzling for Matrix Operations
```cpp
#define TILE_SIZE 32
#define S_TILE_SIZE (TILE_SIZE * (TILE_SIZE + 1))  // +1 to avoid bank conflicts

__global__ void swizzledMatrixTranspose(float* input, float* output) {
    // Use padded tile size to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load in row-major order
    if(x < gridDim.x * TILE_SIZE && y < gridDim.y * TILE_SIZE) {
        tile[threadIdx.y][threadIdx.x] = input[y * gridDim.x * TILE_SIZE + x];
    }
    __syncthreads();
    
    // Transpose: read in column-major order
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if(x < gridDim.y * TILE_SIZE && y < gridDim.x * TILE_SIZE) {
        output[y * gridDim.y * TILE_SIZE + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Step 5: Advanced XOR Swizzling for Different Access Patterns
```cpp
#define WARP_SIZE 32
#define NUM_BANKS 32

// Advanced swizzling for multiple access patterns
__device__ int advancedXorSwizzle(int addr, int stride) {
    // Different swizzling based on access pattern
    int bank = (addr / 4) % NUM_BANKS;
    
    // For strided access, use stride to determine swizzling
    int offset = (stride * bank) % NUM_BANKS;
    return addr ^ (offset << 2);
}

__global__ void advancedSwizzledKernel(float* input, float* output, int stride) {
    __shared__ float sdata[1024 + NUM_BANKS * 4];  // Extra space for swizzling
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Swizzled storage
    sdata[advancedXorSwizzle(threadIdx.x, stride)] = input[tid];
    __syncthreads();
    
    // Swizzled access with different pattern
    int access_idx = (threadIdx.x * stride) % 32;
    float value = sdata[advancedXorSwizzle(access_idx, stride)];
    
    output[tid] = value;
}
```

### Step 6: Complete Example with Multiple Swizzling Techniques
```cpp
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define NUM_BANKS 32

__global__ void completeSwizzledGemm(float* A, float* B, float* C, 
                                   int M, int N, int K) {
    // Use padding to avoid bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + 1];
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    
    float sum = 0.0f;
    
    for(int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        // Load tiles with padding to avoid bank conflicts
        int a_row = row;
        int a_col = tile * TILE_DIM + tx;
        As[ty][tx] = ((a_row < M) && (a_col < K)) ? 
                      A[a_row * K + a_col] : 0.0f;
        
        int b_row = tile * TILE_DIM + ty;
        int b_col = col;
        Bs[ty][tx] = ((b_row < K) && (b_col < N)) ? 
                      B[b_row * N + b_col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial result
        for(int k = 0; k < TILE_DIM; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Common Pitfalls and Solutions

### 1. Incorrect Address Calculation
- **Problem**: Wrong bank calculation formula
- **Solution**: Use correct formula: `(address / word_size) % num_banks`

### 2. Insufficient Padding
- **Problem**: Not enough extra space for swizzling
- **Solution**: Add appropriate padding to shared memory arrays

### 3. Over-Swizzling
- **Problem**: Excessive swizzling causing memory waste
- **Solution**: Use minimal swizzling needed for the access pattern

### 4. Ignoring Access Patterns
- **Problem**: Same swizzling for different access patterns
- **Solution**: Adapt swizzling to specific access patterns

## Performance Considerations

### Memory Usage
- Swizzling requires additional memory space
- Balance between performance gain and memory cost

### Access Pattern Analysis
- Analyze actual access patterns in your kernel
- Apply swizzling only where needed

### Bank Width
- Consider the width of memory banks (usually 32 or 64 bits)
- Adjust swizzling accordingly

### Occupancy
- Ensure swizzling doesn't reduce occupancy
- Balance shared memory usage with thread count

## Real-World Applications

- **Matrix Operations**: Transpose, multiplication, convolution
- **FFT Algorithms**: Cooley-Tukey FFT implementations
- **Sorting Algorithms**: Bitonic sort, merge sort
- **Graph Algorithms**: Breadth-first search, shortest path
- **Image Processing**: Filtering, transformation operations

## Advanced Techniques

### Dynamic Swizzling
- Adapt swizzling pattern based on runtime parameters
- More complex but potentially more efficient

### Multi-dimensional Swizzling
- Extend XOR swizzling to 2D and 3D data layouts
- Handle complex access patterns in multiple dimensions

### Bank Conflict Detection
- Tools and techniques to detect bank conflicts
- Profiling to identify performance bottlenecks

## Summary

Shared memory swizzling is a critical optimization technique for avoiding bank conflicts in GPU programming. By understanding how memory banks work and applying appropriate swizzling techniques (particularly XOR-based), we can significantly improve the performance of GPU kernels that heavily use shared memory. The key is to analyze access patterns and apply the right amount of swizzling to achieve optimal performance.