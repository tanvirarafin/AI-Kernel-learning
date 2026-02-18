# Reduction Patterns in CUDA

Master parallel reduction algorithms - a fundamental pattern in GPU computing.

## Concepts Covered
- Tree-based reduction
- Shared memory optimization
- Warp-level primitives
- Atomic operations
- Multi-block reduction

## Optimization Levels

### Level 1: Naive Reduction (`level1_naive_reduction.cu`)
- **Goal**: Understand basic reduction concept
- **Missing**: Tree reduction logic, synchronization
- **Concepts**: Sequential addressing, global memory reduction
- **Performance**: Slow - O(n) with global memory accesses

### Level 2: Shared Memory Reduction (`level2_shared_memory_reduction.cu`)
- **Goal**: Optimize using shared memory
- **Missing**: Shared memory loading, block-level reduction
- **Concepts**: Cooperative reduction, __syncthreads()
- **Performance**: Better - reduces global memory traffic

### Level 3: Warp-Level Optimization (`level3_warp_level_reduction.cu`)
- **Goal**: Use warp shuffle instructions
- **Missing**: Shuffle operations, warp-level primitives
- **Concepts**: __shfl_down_sync, warp-level reduction
- **Performance**: Excellent - no shared memory needed for final warp

### Level 4: Multi-Block Reduction (`level4_multi_block_reduction.cu`)
- **Goal**: Handle large datasets across multiple blocks
- **Missing**: Grid-stride loops, atomic operations
- **Concepts**: Two-pass reduction, atomicAdd
- **Performance**: Scales to any input size

### Level 5: Advanced Patterns (`level5_advanced_patterns.cu`)
- **Goal**: Complex reduction variants
- **Missing**: Custom operators, segmented reduction
- **Concepts**: Min/Max, ArgMax, histogram as reduction
- **Performance**: Application-specific optimizations

## Compilation
```bash
nvcc level1_naive_reduction.cu -o level1
nvcc level2_shared_memory_reduction.cu -o level2
nvcc level3_warp_level_reduction.cu -o level3
nvcc level4_multi_block_reduction.cu -o level4
nvcc level5_advanced_patterns.cu -o level5
```

## Key Principles
1. **Tree Reduction**: Halve the working set each iteration
2. **Shared Memory**: Use for block-level intermediate results
3. **Warp Shuffle**: Fastest for final warp reduction
4. **Atomic Operations**: Combine results from multiple blocks
5. **Memory Coalescing**: Ensure coalesced reads even in reduction
