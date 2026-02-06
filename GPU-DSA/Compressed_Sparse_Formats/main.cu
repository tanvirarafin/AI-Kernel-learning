#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

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

// CSR Matrix-Vector Multiplication
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

// Optimized CSR Matrix-Vector Multiplication
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

// BSR Matrix-Vector Multiplication
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

// Convert dense matrix to CSR format
CSRMatrix* denseToCSR(const float* dense_matrix, int rows, int cols) {
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

// Convert dense to BSR format
BSRMatrix* denseToBSR(const float* dense_matrix, 
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

int main() {
    const int ROWS = 128;
    const int COLS = 128;
    const int TOTAL_SIZE = ROWS * COLS;
    const int VEC_SIZE = COLS;
    
    // Host memory allocation
    std::vector<float> h_dense(TOTAL_SIZE, 0.0f);
    std::vector<float> h_vector(VEC_SIZE);
    std::vector<float> h_result(ROWS);
    
    // Create a sparse matrix (only fill ~10% of the matrix)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<int> row_dis(0, ROWS-1);
    std::uniform_int_distribution<int> col_dis(0, COLS-1);
    
    // Fill about 10% of the matrix with random values
    int nnz_target = TOTAL_SIZE / 10;
    for(int i = 0; i < nnz_target; i++) {
        int r = row_dis(gen);
        int c = col_dis(gen);
        h_dense[r * COLS + c] = dis(gen);
    }
    
    // Initialize vector with random values
    for(int i = 0; i < VEC_SIZE; i++) {
        h_vector[i] = dis(gen);
    }
    
    // Convert dense matrix to CSR format
    CSRMatrix* csr_matrix = denseToCSR(h_dense.data(), ROWS, COLS);
    
    // Device memory allocation
    float *d_vector, *d_result;
    cudaMalloc(&d_vector, VEC_SIZE * sizeof(float));
    cudaMalloc(&d_result, ROWS * sizeof(float));
    
    // Copy vector to device
    cudaMemcpy(d_vector, h_vector.data(), VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch CSR matrix-vector multiplication
    int blockSize = 256;
    int gridSize = (ROWS + blockSize - 1) / blockSize;
    
    optimizedCsrMV<<<gridSize, blockSize>>>(
        csr_matrix->values, 
        csr_matrix->column_indices, 
        csr_matrix->row_pointers, 
        d_vector, 
        d_result, 
        ROWS
    );
    
    // Copy result back to host
    cudaMemcpy(h_result.data(), d_result, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "CSR Matrix-Vector Multiplication completed." << std::endl;
    std::cout << "Result (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with BSR format
    const int BLOCK_SIZE = 8;
    BSRMatrix* bsr_matrix = denseToBSR(h_dense.data(), ROWS, COLS, BLOCK_SIZE);
    
    // Launch BSR matrix-vector multiplication
    int bsr_gridSize = ((ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE + blockSize - 1) / blockSize;
    
    bsrMV<<<bsr_gridSize, blockSize>>>(
        bsr_matrix->block_values,
        bsr_matrix->column_indices,
        bsr_matrix->row_pointers,
        d_vector,
        d_result,
        (ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Number of block rows
        BLOCK_SIZE
    );
    
    // Copy result back to host
    cudaMemcpy(h_result.data(), d_result, ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nBSR Matrix-Vector Multiplication completed." << std::endl;
    std::cout << "Result (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify sparsity
    int actual_nnz = 0;
    for(int i = 0; i < TOTAL_SIZE; i++) {
        if(fabsf(h_dense[i]) > 1e-9f) actual_nnz++;
    }
    
    std::cout << "\nMatrix sparsity information:" << std::endl;
    std::cout << "Matrix size: " << ROWS << " x " << COLS << std::endl;
    std::cout << "Total elements: " << TOTAL_SIZE << std::endl;
    std::cout << "Non-zero elements: " << actual_nnz << std::endl;
    std::cout << "Sparsity: " << (1.0f - (float)actual_nnz / TOTAL_SIZE) * 100.0f << "%" << std::endl;
    std::cout << "CSR format stores " << csr_matrix->nnz << " values" << std::endl;
    
    // Cleanup
    delete csr_matrix;
    delete bsr_matrix;
    cudaFree(d_vector);
    cudaFree(d_result);
    
    return 0;
}