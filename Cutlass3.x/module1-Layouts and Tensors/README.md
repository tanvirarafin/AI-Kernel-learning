# Module 1: Layouts and Tensors

## First Principles Explanation

CuTe (CUDA Templates for Element-wise operations) introduces a revolutionary approach to tensor programming through "layout algebra". Rather than manually managing indices and strides, CuTe allows us to define mathematical relationships between logical and physical memory representations.

### Key Concepts:

1. **Layout**: A mapping from logical coordinates to linear memory addresses
2. **Shape**: The dimensions of the tensor (e.g., [M, N] for a 2D matrix)
3. **Stride**: The step size between elements in each dimension
4. **Composition**: Combining layouts using mathematical operations

### Layout Algebra Foundation:

A `cute::Layout` is fundamentally defined by:
- **Shape**: Defines the extent of each dimension
- **Stride**: Defines how many elements to skip to move one position in each dimension

For a 2D matrix A[M][N]:
- Row-major: Layout({M, N}, {N, 1}) - stride N in first dim, stride 1 in second
- Column-major: Layout({M, N}, {1, M}) - stride 1 in first dim, stride M in second

The address calculation becomes: `addr = i * stride_0 + j * stride_1`

## Composable Data Access

Instead of nested loops with manual indexing, CuTe enables us to think in terms of "tensor partitioning":
- Partition a large tensor into tiles that map to threads
- Each thread operates on its assigned tile
- Mathematical composition handles the complex indexing

This approach eliminates bounds checking, reduces indexing errors, and enables more readable, maintainable code that closely mirrors mathematical notation.