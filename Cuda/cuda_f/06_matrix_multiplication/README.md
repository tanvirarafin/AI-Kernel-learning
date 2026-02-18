# Matrix Multiplication in CUDA

Master matrix multiplication optimizations - the "Hello World" of GPU computing.

## Concepts Covered
- Naive global memory implementation
- Shared memory tiling
- Bank conflict avoidance
- Register tiling
- Vectorized memory access
- Tensor Core operations (Volta+)

## Optimization Levels

### Level 1: Naive Matrix Multiplication (`level1_naive_matmul.cu`)
- **Goal**: Understand basic matrix multiplication on GPU
- **Missing**: Thread indexing, bounds checking
- **Concepts**: Global memory access, row-major layout
- **Performance**: Baseline - memory bound

### Level 2: Shared Memory Tiling (`level2_shared_mem_tiled.cu`)
- **Goal**: Optimize using shared memory tiles
- **Missing**: Tile loading, synchronization, tile computation
- **Concepts**: Tiling, data reuse, computational intensity
- **Performance**: 5-10x faster than naive

### Level 3: Optimized Tiling (`level3_optimized_tiled.cu`)
- **Goal**: Avoid bank conflicts and optimize further
- **Missing**: Padding, vectorized loads, register caching
- **Concepts**: Bank conflicts, coalesced access, register tiling
- **Performance**: 2-3x faster than basic tiling

### Level 4: Register Tiling (`level4_register_tiling.cu`)
- **Goal**: Each thread computes multiple output elements
- **Missing**: 2D register tiling, increased arithmetic intensity
- **Concepts**: Register-level tiling, thread-level parallelism
- **Performance**: Better occupancy and throughput

### Level 5: Tensor Core MatMul (`level5_tensor_core_matmul.cu`)
- **Goal**: Use Tensor Cores for maximum performance (Volta+)
- **Missing**: WMMA API usage, mixed precision
- **Concepts**: Tensor Cores, mixed precision, matrix fragments
- **Performance**: 10-20x faster than naive (hardware dependent)

## Compilation
```bash
nvcc level1_naive_matmul.cu -o level1
nvcc level2_shared_mem_tiled.cu -o level2
nvcc level3_optimized_tiled.cu -o level3
nvcc level4_register_tiling.cu -o level4
nvcc -arch=sm_70 level5_tensor_core_matmul.cu -o level5  # Requires Volta+
```

## Key Principles
1. **Tiling**: Load submatrices to shared memory for reuse
2. **Synchronization**: Always sync after loading tiles
3. **Bank Conflicts**: Pad shared memory to avoid conflicts
4. **Register Tiling**: Each thread computes multiple outputs
5. **Tensor Cores**: Use mixed precision for maximum throughput

## Performance Tips
- Tile size: 32x32 or 16x16 typically optimal
- Padding: Add 1 element to avoid bank conflicts
- Vectorized loads: Use float4 for 4x bandwidth
- Double buffering: Overlap load and compute
