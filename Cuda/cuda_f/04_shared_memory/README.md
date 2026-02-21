# Shared Memory Fundamentals

Master shared memory programming for high-performance CUDA kernels.

## Concepts Covered
- Shared memory declaration and usage
- Thread synchronization with `__syncthreads()`
- Bank conflict identification and avoidance
- Tiling patterns for data reuse

## Exercise Levels

### Level 1: Basic Shared Memory (`level1_basic_shared.cu`)
- **Goal**: Understand shared memory basics and synchronization
- **Missing**: Shared memory declaration, data loading, synchronization
- **Concepts**: `__shared__`, `__syncthreads()`, block-level communication

### Level 2: Tiled Matrix Multiplication (`level2_tiled_matmul.cu`)
- **Goal**: Implement tiled matrix multiplication with shared memory
- **Missing**: Tile loading, synchronization points, tile computation
- **Concepts**: Memory reuse, computational intensity, tiling

### Level 3: Bank Conflict Resolution (`level3_bank_conflicts.cu`)
- **Goal**: Identify and fix bank conflicts
- **Missing**: Padding strategies, access pattern modification
- **Concepts**: Memory banks, conflict-free access, padding

### Level 4: Advanced Shared Memory Patterns (`level4_advanced_patterns.cu`)
- **Goal**: Complex shared memory algorithms
- **Missing**: Multi-stage algorithms, dynamic shared memory
- **Concepts**: Histograms, sorting, dynamic allocation

## Compilation
```bash
nvcc level1_basic_shared.cu -o level1
nvcc level2_tiled_matmul.cu -o level2
nvcc level3_bank_conflicts.cu -o level3
nvcc level4_advanced_patterns.cu -o level4
```

## Key Principles
1. **Declaration**: Use `__shared__` keyword for shared memory
2. **Synchronization**: Always sync before reading data written by other threads
3. **Bank Conflicts**: 32 banks on modern GPUs, avoid multiple threads accessing same bank
4. **Tiling**: Load data in tiles for maximum reuse before eviction
