# Warp Primitives in CUDA

Master warp-level primitives for efficient intra-warp communication.

## Concepts Covered
- Warp shuffle operations
- Warp vote (ballot)
- Warp match
- Lane ID operations
- Warp-synchronous algorithms

## Levels

### Level 1: Warp Shuffle (`level1_shuffle_ops.cu`)
- **Goal**: Learn basic shuffle operations
- **Missing**: __shfl, __shfl_down, __shfl_up, __shfl_xor
- **Concepts**: Lane communication, register forwarding
- **Operations**: Data movement within warp

### Level 2: Warp Ballot (`level2_ballot_ops.cu`)
- **Goal**: Use ballot for warp-level decisions
- **Missing**: __ballot_sync, __any_sync, __all_sync
- **Concepts**: Warp-level predicates, voting
- **Operations**: Bitmask operations

### Level 3: Warp Reduction (`level3_warp_reduction.cu`)
- **Goal**: Efficient reduction using shuffle
- **Missing**: Tree reduction with shuffle
- **Concepts**: Warp-synchronous reduction
- **Operations**: __shfl_down for reduction

### Level 4: Warp Broadcast (`level4_warp_broadcast.cu`)
- **Goal**: Broadcast data within warp
- **Missing**: __shfl for broadcast
- **Concepts**: Single source to all lanes
- **Operations**: Efficient data distribution

### Level 5: Advanced Warp Patterns (`level5_advanced_warp.cu`)
- **Goal**: Complex warp-level algorithms
- **Missing**: Warp-level primitives composition
- **Concepts**: Combined operations
- **Operations**: Match, rotate, swap

## Compilation
```bash
nvcc level1_shuffle_ops.cu -o level1
nvcc level2_ballot_ops.cu -o level2
nvcc level3_warp_reduction.cu -o level3
nvcc level4_warp_broadcast.cu -o level4
nvcc level5_advanced_warp.cu -o level5
```

## Key Principles
1. **Warp Size**: Typically 32 threads (warpSize)
2. **No Sync Needed**: Warp executes in lockstep
3. **Shuffle Mask**: 0xffffffff for full warp
4. **Lane ID**: threadIdx.x % warpSize
5. **Efficiency**: Faster than shared memory for warp-sized data

## Important Notes
- Shuffle instructions require compute capability 3.0+
- Sync-free within a warp (threads execute in lockstep)
- Use __shfl_sync variants for explicit mask control
- Cross-warp communication requires shared memory
