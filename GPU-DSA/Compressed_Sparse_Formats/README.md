# Compressed Sparse Formats: CSR and Blocked-Ellpack

## Overview

Compressed sparse formats are specialized data structures designed to efficiently store and process sparse matrices (matrices with many zero elements). These formats are crucial for applications in scientific computing, machine learning, and graph processing where sparse matrices are common. The two main formats covered here are Compressed Sparse Row (CSR) and Blocked-Ellpack (BSR).

## Why Compressed Sparse Formats?

Dense matrix representations waste memory and computation when matrices are mostly zeros:
- **Memory**: Store only non-zero elements
- **Computation**: Skip operations involving zeros
- **Bandwidth**: Reduce memory traffic
- **Performance**: Focus computation on meaningful data

Sparse formats address these issues by:
- Compressing storage to eliminate zeros
- Providing efficient access patterns for computation
- Enabling specialized algorithms for sparse operations

## Key Concepts

### Sparsity
Sparsity refers to the proportion of zero elements in a matrix. A matrix with 90% zeros is 90% sparse.

### Non-Zero Elements (NNZ)
The count of non-zero elements determines the storage and computation requirements.

### Compression
Storing only non-zero values along with structural information to reconstruct positions.

## Compressed Sparse Row (CSR) Format

### Structure
CSR uses three arrays:
1. **Values**: Contains non-zero values
2. **Column Indices**: Column index for each non-zero value
3. **Row Pointers**: Starting index in values/columns for each row

### Example
```
Dense Matrix:     CSR Representation:
[3 0 4 0]        Values: [3 4 2 1 5]
[0 0 2 1]        ColInd: [0 2 2 3 4]
[0 0 0 5]        RowPtr: [0 2 4 5]
```

### Advantages
- Efficient row access
- Good cache performance for row-wise operations
- Low memory overhead

### Disadvantages
- Inefficient column access
- Irregular memory access patterns for some operations

## Blocked-Ellpack (BSR) Format

### Structure
BSR extends CSR by grouping elements into dense blocks:
1. **Block Values**: Dense blocks of non-zero values
2. **Column Indices**: Column index for each block
3. **Row Pointers**: Starting index for each row of blocks

### Advantages
- Better cache performance for block operations
- Vectorization opportunities
- Reduced metadata overhead

### Disadvantages
- Less flexible than CSR
- Performance depends on block size selection
- Zero padding within blocks

## Step-by-Step Implementation Guide

### Step 1: CSR Format Definition
```cpp
#include <cuda_runtime.h>
#include <vector>

struct CSRMatrix {
    float* values;           // Non-zero values
    int* column_indices;     // Column index for each value
    int* row_pointers;       // Start index for each row
    int rows, cols;          // Matrix dimensions
    int nnz;                 // Number of non-zero elements
    
    CSRMatrix(int r, int c, int n) : rows(r), cols(c), nnz(n) {
        cudaMalloc(&values, nnz * sizeof(float));
        cudaMalloc(&column_indices, nnz * sizeof(int));
        cudaMalloc(&row_pointers, (rows + 1) * sizeof(int));
    }
    
    ~CSRMatrix() {
        cudaFree(values);
        cudaFree(column_indices);
        cudaFree(row_pointers);
    }
};
```

### Step 2: Basic CSR Matrix-Vector Multiplication
```cpp
__global__ void csrMV(const float* values, 
                     const int* column_indices,
                     const int* row_pointers,
                     const float* vec,
                     float* result,
                     int rows) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < rows) {
        float sum = 0.0f;
        int start = row_pointers[row];
        int end = row_pointers[row + 1];
        
        for(int j = start; j < end; j++) {
            int col = column_indices[j];
            sum += values[j] * vec[col];
        }
        
        result[row] = sum;
    }
}
```

### Step 3: Optimized CSR Matrix-Vector Multiplication
```cpp
__global__ void optimizedCsrMV(const float* values,
                              const int* column_indices,
                              const int* row_pointers,
                              const float* vec,
                              float* result,
                              int rows) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < rows) {
        float sum = 0.0f;
        int start = row_pointers[row];
        int end = row_pointers[row + 1];
        
        // Unroll loop for better performance
        int j = start;
        for(; j + 3 < end; j += 4) {
            sum += values[j] * vec[column_indices[j]];
            sum += values[j+1] * vec[column_indices[j+1]];
            sum += values[j+2] * vec[column_indices[j+2]];
            sum += values[j+3] * vec[column_indices[j+3]];
        }
        
        // Handle remaining elements
        for(; j < end; j++) {
            sum += values[j] * vec[column_indices[j]];
        }
        
        result[row] = sum;
    }
}
```

### Step 4: Blocked CSR (BSR) Format Definition
```cpp
struct BSRMatrix {
    float* block_values;     // Dense blocks of values
    int* column_indices;     // Column index for each block
    int* row_pointers;       // Start index for each row of blocks
    int rows, cols;          // Matrix dimensions (in blocks)
    int block_size;          // Size of each block (typically square)
    int nnz_blocks;          // Number of non-zero blocks
    
    BSRMatrix(int r, int c, int bs, int nb) 
        : rows(r), cols(c), block_size(bs), nnz_blocks(nb) {
        
        int total_elements = nnz_blocks * block_size * block_size;
        cudaMalloc(&block_values, total_elements * sizeof(float));
        cudaMalloc(&column_indices, nnz_blocks * sizeof(int));
        cudaMalloc(&row_pointers, (rows + 1) * sizeof(int));
    }
    
    ~BSRMatrix() {
        cudaFree(block_values);
        cudaFree(column_indices);
        cudaFree(row_pointers);
    }
};
```

### Step 5: BSR Matrix-Vector Multiplication
```cpp
__global__ void bsrMV(const float* block_values,
                     const int* column_indices,
                     const int* row_pointers,
                     const float* vec,
                     float* result,
                     int rows,
                     int block_size) {
    
    int row_block = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row_block < rows) {
        int start = row_pointers[row_block];
        int end = row_pointers[row_block + 1];
        
        // Each thread handles one block row
        for(int b = start; b < end; b++) {
            int col_block = column_indices[b];
            
            // Process the block
            for(int i = 0; i < block_size; i++) {
                float sum = 0.0f;
                for(int j = 0; j < block_size; j++) {
                    int block_idx = b * block_size * block_size + i * block_size + j;
                    int vec_idx = col_block * block_size + j;
                    sum += block_values[block_idx] * vec[vec_idx];
                }
                
                int result_idx = row_block * block_size + i;
                result[result_idx] += sum;
            }
        }
    }
}
```

### Step 6: Complete Sparse Matrix Operations
```cpp
class SparseMatrixOperations {
public:
    // Convert dense matrix to CSR format
    static CSRMatrix* denseToCSR(const float* dense_matrix, int rows, int cols) {
        // Count non-zeros
        int nnz = 0;
        for(int i = 0; i < rows * cols; i++) {
            if(fabsf(dense_matrix[i]) > 1e-9f) nnz++;
        }
        
        CSRMatrix* csr = new CSRMatrix(rows, cols, nnz);
        
        // Allocate host memory for conversion
        std::vector<float> h_values(nnz);
        std::vector<int> h_col_indices(nnz);
        std::vector<int> h_row_ptrs(rows + 1);
        
        // Fill the arrays
        int val_idx = 0;
        h_row_ptrs[0] = 0;
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(fabsf(dense_matrix[i * cols + j]) > 1e-9f) {
                    h_values[val_idx] = dense_matrix[i * cols + j];
                    h_col_indices[val_idx] = j;
                    val_idx++;
                }
            }
            h_row_ptrs[i + 1] = val_idx;
        }
        
        // Copy to device
        cudaMemcpy(csr->values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(csr->column_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csr->row_pointers, h_row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        return csr;
    }
    
    // Perform CSR matrix-vector multiplication
    static void csrMatrixVectorMultiply(const CSRMatrix* A, const float* x, float* y) {
        int block_size = 256;
        int grid_size = (A->rows + block_size - 1) / block_size;
        
        optimizedCsrMV<<<grid_size, block_size>>>(
            A->values, A->column_indices, A->row_pointers, x, y, A->rows
        );
        
        cudaDeviceSynchronize();
    }
    
    // Convert dense to BSR format
    static BSRMatrix* denseToBSR(const float* dense_matrix, 
                                int rows, int cols, int block_size) {
        // This is a simplified version - a full implementation would be more complex
        int row_blocks = (rows + block_size - 1) / block_size;
        int col_blocks = (cols + block_size - 1) / block_size;
        
        // Count non-zero blocks
        int nnz_blocks = 0;
        for(int rb = 0; rb < row_blocks; rb++) {
            for(int cb = 0; cb < col_blocks; cb++) {
                bool has_nonzero = false;
                for(int bi = 0; bi < block_size && (rb * block_size + bi) < rows; bi++) {
                    for(int bj = 0; bj < block_size && (cb * block_size + bj) < cols; bj++) {
                        int r = rb * block_size + bi;
                        int c = cb * block_size + bj;
                        if(fabsf(dense_matrix[r * cols + c]) > 1e-9f) {
                            has_nonzero = true;
                            break;
                        }
                    }
                    if(has_nonzero) break;
                }
                if(has_nonzero) nnz_blocks++;
            }
        }
        
        BSRMatrix* bsr = new BSRMatrix(row_blocks, col_blocks, block_size, nnz_blocks);
        
        // Allocate host memory for conversion
        std::vector<float> h_block_values(nnz_blocks * block_size * block_size);
        std::vector<int> h_col_indices(nnz_blocks);
        std::vector<int> h_row_ptrs(row_blocks + 1);
        
        // Fill the arrays (simplified implementation)
        int block_idx = 0;
        h_row_ptrs[0] = 0;
        
        for(int rb = 0; rb < row_blocks; rb++) {
            for(int cb = 0; cb < col_blocks; cb++) {
                bool has_nonzero = false;
                for(int bi = 0; bi < block_size && (rb * block_size + bi) < rows; bi++) {
                    for(int bj = 0; bj < block_size && (cb * block_size + bj) < cols; bj++) {
                        int r = rb * block_size + bi;
                        int c = cb * block_size + bj;
                        if(fabsf(dense_matrix[r * cols + c]) > 1e-9f) {
                            has_nonzero = true;
                            break;
                        }
                    }
                    if(has_nonzero) break;
                }
                
                if(has_nonzero) {
                    h_col_indices[block_idx] = cb;
                    
                    // Copy block values
                    for(int bi = 0; bi < block_size; bi++) {
                        for(int bj = 0; bj < block_size; bj++) {
                            int r = rb * block_size + bi;
                            int c = cb * block_size + bj;
                            int block_val_idx = block_idx * block_size * block_size + bi * block_size + bj;
                            
                            if(r < rows && c < cols) {
                                h_block_values[block_val_idx] = dense_matrix[r * cols + c];
                            } else {
                                h_block_values[block_val_idx] = 0.0f;
                            }
                        }
                    }
                    block_idx++;
                }
            }
            h_row_ptrs[rb + 1] = block_idx;
        }
        
        // Copy to device
        int total_elements = nnz_blocks * block_size * block_size;
        cudaMemcpy(bsr->block_values, h_block_values.data(), 
                   total_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bsr->column_indices, h_col_indices.data(), 
                   nnz_blocks * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(bsr->row_pointers, h_row_ptrs.data(), 
                   (row_blocks + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        return bsr;
    }
};
```

## Common Pitfalls and Solutions

### 1. Memory Access Patterns
- **Problem**: Irregular memory access in sparse operations
- **Solution**: Optimize for cache efficiency where possible

### 2. Load Balancing
- **Problem**: Different rows having different numbers of non-zeros
- **Solution**: Use dynamic scheduling or work-stealing approaches

### 3. Block Size Selection (for BSR)
- **Problem**: Poor performance with inappropriate block sizes
- **Solution**: Profile and tune for specific matrices

### 4. Conversion Overhead
- **Problem**: Expensive conversion from dense to sparse
- **Solution**: Convert once and reuse, or generate directly in sparse format

## Performance Considerations

### Memory Bandwidth
- Sparse operations often have lower arithmetic intensity
- Optimize for memory bandwidth rather than compute

### Cache Performance
- CSR: Good for row-wise operations
- BSR: Better cache performance for block operations

### Parallelization
- Row-level parallelism in CSR
- Block-level parallelism in BSR

### Sparsity Pattern
- Performance depends heavily on sparsity pattern
- Regular patterns perform better than irregular ones

## Real-World Applications

- **Graph Processing**: Adjacency matrices for graph algorithms
- **Scientific Computing**: Finite element matrices
- **Machine Learning**: Feature matrices with missing values
- **Network Analysis**: Connectivity matrices
- **Recommendation Systems**: User-item interaction matrices

## Advanced Techniques

### Adaptive Format Selection
Choose the best sparse format based on matrix characteristics.

### Hybrid Formats
Combine multiple sparse formats in a single matrix.

### Compression Techniques
Apply additional compression to sparse formats for even greater efficiency.

## Summary

Compressed sparse formats are essential for efficient processing of sparse matrices in GPU computing. CSR provides flexibility and good performance for general sparse matrices, while BSR offers better performance for matrices with block structure. Understanding these formats and their implementations is crucial for high-performance sparse linear algebra operations.