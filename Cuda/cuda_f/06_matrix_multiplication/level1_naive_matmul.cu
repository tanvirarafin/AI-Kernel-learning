/*
 * Matrix Multiplication Level 1: Naive Implementation
 *
 * EXERCISE: Learn the basics of matrix multiplication on GPU.
 *
 * CONCEPTS:
 * - Thread indexing for 2D data
 * - Row-major matrix layout
 * - Dot product computation
 * - Bounds checking
 *
 * SKILLS PRACTICED:
 * - 2D thread indexing
 * - Global memory access patterns
 * - Basic matrix operations
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 512

// ============================================================================
// KERNEL 1: Naive Matrix Multiplication
 * C = A * B where all matrices are in row-major format
 * TODO: Complete the thread indexing and dot product
// ============================================================================
__global__ void naiveMatMul(float *A, float *B, float *C, int width) {
    // TODO: Calculate row and column for this thread
    // Row: which row of C this thread computes
    // Col: which column of C this thread computes
    int row = /* YOUR CODE HERE */;
    int col = /* YOUR CODE HERE */;
    
    // TODO: Add bounds checking
    if (/* YOUR CODE HERE - Check row and col < width */) {
        // TODO: Compute dot product of row from A and column from B
        // C[row][col] = sum(A[row][k] * B[k][col]) for k = 0 to width-1
        float sum = 0.0f;
        
        /* YOUR CODE HERE - Dot product loop */
        
        // TODO: Store result in C (remember row-major layout!)
        // C[row * width + col] = /* YOUR CODE HERE */;
    }
}

// ============================================================================
// KERNEL 2: Naive with 1D Grid (Alternative)
 * Use 1D grid indexing for potentially simpler launch configuration
 * TODO: Complete the 1D to 2D mapping
// ============================================================================
__global__ void naiveMatMul1D(float *A, float *B, float *C, int width) {
    // TODO: Map 1D thread index to 2D matrix coordinates
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Calculate row and column from 1D thread index
    // Total elements = width * width
    // row = tid / width, col = tid % width
    int row = /* YOUR CODE HERE */;
    int col = /* YOUR CODE HERE */;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // TODO: Compute dot product
        /* YOUR CODE HERE */
        
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 3: Naive with Transposed B (Optimization Preview)
 * Transpose B first for coalesced memory access
 * TODO: Complete the transposed access pattern
// ============================================================================
__global__ void naiveMatMulTransposed(float *A, float *B, float *C, 
                                       float *B_transposed, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        // TODO: First, load and transpose B element
        // B_transposed[col * width + row] = B[row * width + col];
        // (This would be done in a separate kernel typically)
        
        // Compute dot product with transposed B
        // Now both A and B_transposed have coalesced access!
        float sum = 0.0f;
        
        for (int k = 0; k < width; k++) {
            // TODO: Access B_transposed for coalesced reads
            // sum += A[row * width + k] * B_transposed[/* YOUR CODE HERE */];
        }
        
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 4: Batched Matrix Multiplication
 * Multiply multiple matrices in one kernel launch
 * TODO: Complete the batched implementation
// ============================================================================
__global__ void batchedMatMul(float *A, float *B, float *C, 
                               int width, int batchSize) {
    // TODO: Calculate which matrix in the batch this thread works on
    int batchId = blockIdx.z;  // Use z-dimension for batch
    
    // TODO: Calculate row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batchId < batchSize && row < width && col < width) {
        // TODO: Calculate offsets for this batch
        int offset = batchId * width * width;
        
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            // TODO: Access elements with batch offset
            /* YOUR CODE HERE */
        }
        
        C[offset + row * width + col] = sum;
    }
}

// Utility functions
void initMatrix(float *mat, int width, float val) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = val;
    }
}

void initMatrixSequential(float *mat, int width) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)i;
    }
}

bool verifyMatMul(float *C, float *A, float *B, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0.0f;
            for (int k = 0; k < width; k++) {
                expected += A[row * width + k] * B[k * width + col];
            }
            float diff = fabsf(C[row * width + col] - expected);
            if (diff > 1e-2f * width) {  // Scale tolerance with matrix size
                printf("Mismatch at (%d, %d): expected %f, got %f\n", 
                       row, col, expected, C[row * width + col]);
                return false;
            }
        }
    }
    return true;
}

void printMatrixCorner(float *mat, int width, const char *label) {
    printf("%s (top-left 5x5):\n", label);
    for (int i = 0; i < 5; i++) {
        printf("  ");
        for (int j = 0; j < 5; j++) {
            printf("%10.2f ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Matrix Multiplication Level 1: Naive Implementation ===\n\n");
    
    const int WIDTH = MATRIX_SIZE;
    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);
    
    // Initialize matrices: A = all 1s, B = sequential
    initMatrix(h_A, WIDTH, 1.0f);
    initMatrixSequential(h_B, WIDTH);
    
    // Compute reference result on CPU
    for (int row = 0; row < WIDTH; row++) {
        for (int col = 0; col < WIDTH; col++) {
            float sum = 0.0f;
            for (int k = 0; k < WIDTH; k++) {
                sum += h_A[row * WIDTH + k] * h_B[k * WIDTH + col];
            }
            h_C_ref[row * WIDTH + col] = sum;
        }
    }
    
    printMatrixCorner(h_A, WIDTH, "Matrix A");
    printMatrixCorner(h_B, WIDTH, "Matrix B");
    printMatrixCorner(h_C_ref, WIDTH, "Reference C = A*B");
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch configuration for 2D grid
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + 31) / 32, (WIDTH + 31) / 32);
    
    printf("\nMatrix size: %d x %d\n", WIDTH, WIDTH);
    printf("Grid: %d x %d blocks, Block: %d x %d threads\n\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Test 1: Naive matrix multiplication
    printf("Test 1: Naive matrix multiplication\n");
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the thread indexing and dot product\n");
    }
    
    // Test 2: Naive with 1D grid
    printf("\nTest 2: Naive matrix multiplication (1D grid)\n");
    int totalThreads = WIDTH * WIDTH;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    naiveMatMul1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the 1D to 2D mapping\n");
    }
    
    // Test 3: Batched matrix multiplication (single batch for testing)
    printf("\nTest 3: Batched matrix multiplication\n");
    float *d_A_batch, *d_B_batch, *d_C_batch;
    int batchSize = 4;
    size_t batchSizeBytes = size * batchSize;
    
    cudaMalloc(&d_A_batch, batchSizeBytes);
    cudaMalloc(&d_B_batch, batchSizeBytes);
    cudaMalloc(&d_C_batch, batchSizeBytes);
    
    // Copy same matrices to all batches
    for (int i = 0; i < batchSize; i++) {
        cudaMemcpy(d_A_batch + i * N, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_batch + i * N, h_B, size, cudaMemcpyHostToDevice);
    }
    
    dim3 gridDimBatch(gridDim.x, gridDim.y, batchSize);
    batchedMatMul<<<gridDimBatch, blockDim>>>(d_A_batch, d_B_batch, d_C_batch, 
                                               WIDTH, batchSize);
    cudaDeviceSynchronize();
    
    // Verify first batch
    cudaMemcpy(h_C, d_C_batch, size, cudaMemcpyDeviceToHost);
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED (first batch verified)\n");
    } else {
        printf("  ✗ FAILED - Complete the batched implementation\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_batch);
    cudaFree(d_B_batch);
    cudaFree(d_C_batch);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Matrix multiplication is O(n³) - great for parallelization\n");
    printf("- Row-major layout: element (row, col) at index row*width + col\n");
    printf("- 2D thread mapping matches 2D data naturally\n");
    printf("- Naive version is memory-bound (repeated global memory accesses)\n");
    printf("\nNext: Try level2_shared_mem_tiled.cu for shared memory optimization\n");
    
    return 0;
}
