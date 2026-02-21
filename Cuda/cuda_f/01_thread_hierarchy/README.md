# Thread Hierarchy Fundamentals

Master the CUDA thread execution model through progressive kernel implementations.

## Concepts Covered
- Grid, block, and thread indexing
- Multi-dimensional thread configurations (1D, 2D, 3D)
- Grid-stride loops for flexible problem sizes
- Warp-level execution understanding

## Exercise Levels

### Level 1: Basic Indexing (`level1_basic_indexing.cu`)
- **Goal**: Implement correct thread indexing for 1D, 2D, and 3D configurations
- **Missing**: Global index calculations, bounds checking
- **Concepts**: `blockIdx`, `threadIdx`, `blockDim`, `gridDim`

### Level 2: Grid-Stride Loop (`level2_grid_stride.cu`)
- **Goal**: Implement grid-stride loop for arbitrary problem sizes
- **Missing**: Loop stride calculation, iteration logic
- **Concepts**: Scalable kernel design, large dataset handling

### Level 3: Multi-Dimensional Data (`level3_multidim_data.cu`)
- **Goal**: Process 2D/3D data structures (images, volumes)
- **Missing**: Dimension mapping, boundary conditions
- **Concepts**: Image processing, volume data patterns

### Level 4: Warp-Aware Programming (`level4_warp_aware.cu`)
- **Goal**: Optimize considering warp execution
- **Missing**: Warp synchronization, divergence avoidance
- **Concepts**: Warp primitives, execution efficiency

## Compilation
```bash
nvcc level1_basic_indexing.cu -o level1
nvcc level2_grid_stride.cu -o level2
nvcc level3_multidim_data.cu -o level3
nvcc level4_warp_aware.cu -o level4
```
