# Module 5: Mainloop Pipelining - Temporal Overlap & Throughput

## Overview

This module focuses on mainloop pipelining, which is the heart of high-performance GEMM kernels. The concept involves overlapping memory loads with computation to hide memory latency through a "double-buffered" approach.

## Key Concepts

### 1. Double Buffering
- Two buffers (A and B) alternate between loading and computing
- While current tile is being computed, the next tile is loaded into the other buffer
- Creates a pipeline where memory operations and compute operations happen concurrently

### 2. Pipeline Stages
- **Stage 0**: Load tiles for iteration 0 into buffer 0
- **Stage 1**: Load tiles for iteration 1 into buffer 1, compute tiles from buffer 0
- **Stage 2+**: Alternate loading into one buffer while computing from the other

### 3. Temporal Overlap
- Current computation overlaps with next tile loading
- Maximizes utilization of compute and memory bandwidth
- Hides memory latency by keeping the compute units busy

### 4. Throughput Optimization
- Maximizes sustained performance
- Balances memory bandwidth utilization with compute capacity
- Critical for achieving peak hardware performance

## Implementation Details

The implementation demonstrates:

1. **Shared Memory Allocation**: Two buffers for A and B matrices
2. **Thread Mapping**: Efficient partitioning of work among threads
3. **Pipeline Control**: Buffer toggling mechanism
4. **Memory Access Patterns**: Coalesced access for optimal bandwidth

## Compilation

To compile this module:

```bash
nvcc -std=c++17 -arch=sm_89 -I. -I../third_party/cutlass/include main.cu -o module5
```

Or using CMake:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Expected Output

The program will:
1. Execute a double-buffered GEMM operation
2. Report execution time and performance (GFLOPs)
3. Demonstrate the benefits of temporal overlap

## Learning Objectives

After completing this module, you should understand:
- How to implement double buffering for memory operations
- The importance of temporal overlap in high-performance kernels
- How to structure a pipelined mainloop
- Techniques for maximizing throughput in GEMM operations

## Next Steps

This module represents the core of high-performance kernels. Understanding these concepts is crucial for developing production-level GEMM implementations that achieve near-peak hardware performance.