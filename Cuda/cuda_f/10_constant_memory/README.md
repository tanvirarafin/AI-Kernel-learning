# Constant Memory in CUDA

Master constant memory for read-only uniform data access.

## Concepts Covered
- Constant memory declaration
- Constant memory caching
- Broadcast optimization
- 64KB limit

## Levels

### Level 1: Constant Memory Basics (`level1_constant_basics.cu`)
- **Goal**: Learn constant memory declaration and usage
- **Missing**: __constant__ declaration, cudaMemcpyToSymbol
- **Concepts**: Constant memory space, broadcast reads

### Level 2: Lookup Tables (`level2_lookup_tables.cu`)
- **Goal**: Use constant memory for lookup tables
- **Missing**: Table initialization, indexed access
- **Concepts**: Precomputed values, table-driven algorithms

### Level 3: Kernel Parameters (`level3_kernel_params.cu`)
- **Goal**: Store kernel parameters in constant memory
- **Missing**: Struct in constant memory, bulk copy
- **Concepts**: Parameter batches, uniform parameters

### Level 4: Optimization Patterns (`level4_optimization.cu`)
- **Goal**: Optimize for constant memory broadcast
- **Missing**: Access pattern optimization
- **Concepts**: Broadcast vs cached reads

## Key Principles
1. **64KB Limit**: Total constant memory space
2. **Broadcast**: Fast when all threads read same address
3. **Cached**: Constant cache optimized for read-only
4. ** cudaMemcpyToSymbol**: Copy data to constant memory
