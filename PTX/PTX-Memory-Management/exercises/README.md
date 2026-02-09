# PTX Memory Management Exercises

This directory contains hands-on exercises to reinforce your understanding of PTX memory management and optimization.

## Exercise 1: Shared Memory Tiling

### Objective
Implement a matrix multiplication kernel using shared memory tiling to improve performance.

### Files
- `matmul_tiled.ptx` - PTX implementation with shared memory tiling
- `test_matmul_tiled.cu` - CUDA test harness
- `benchmark_tiled.cu` - Performance comparison

## Exercise 2: Memory Coalescing

### Objective
Compare coalesced vs non-coalesced memory access patterns in PTX.

### Files
- `coalesced_access.ptx` - PTX with coalesced access
- `non_coalesced_access.ptx` - PTX with non-coalesced access
- `test_coalescing.cu` - CUDA test harness with timing

## Exercise 3: Memory Bank Conflicts

### Objective
Demonstrate shared memory bank conflicts and how to avoid them.

### Files
- `bank_conflict.ptx` - PTX with bank conflicts
- `avoid_bank_conflict.ptx` - PTX without bank conflicts
- `test_banks.cu` - CUDA test harness

## Exercise 4: Global Memory Optimization

### Objective
Implement various global memory optimization techniques in PTX.

### Files
- `global_optimized.ptx` - PTX with optimized global memory access
- `global_naive.ptx` - PTX with naive global memory access
- `test_global.cu` - CUDA test harness

## How to Use These Exercises

1. Study the provided PTX examples
2. Modify them to complete the exercises
3. Compile with `nvcc -ptx filename.ptx -o filename.ptx` (to validate syntax)
4. Test with the provided CUDA harness
5. Profile with `nvprof` or `Nsight Compute` to measure performance differences
6. Examine the generated assembly with `cuobjdump`