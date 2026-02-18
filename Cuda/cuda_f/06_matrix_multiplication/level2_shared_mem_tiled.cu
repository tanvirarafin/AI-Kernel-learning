/*
 * Matrix Multiplication Level 2: Shared Memory Tiling
 *
 * EXERCISE: Optimize matrix multiplication using shared memory tiles.
 *
 * CONCEPTS:
 * - Tiling for data reuse
 * - Cooperative tile loading
 * - Synchronization points
 * - Reduced global memory bandwidth
 *
 * SKILLS PRACTICED:
 * - Shared memory declaration
 * - Tile-based algorithms
 * - Multiple synchronization points
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 512
#define TILE_WIDTH 32

// ============================================================================
// KERNEL 1: Basic Tiled Matrix Multiplication
 * Load tiles of A and B to shared memory, compute tile contribution
 * TODO: Complete the tile loading and computation
// ============================================================================
__global__ void tiledMatMul(float *A, float *B, float *C, int width) {
    // TODO: Declare shared memory for tiles
    // Need one tile for A and one tile for B
    // __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    /* YOUR DECLARATIONS HERE */
    
    // Calculate thread and block indices
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // Accumulator for this thread's output element
    float accumulator = 0.0f;
    
    // Number of tiles needed to cover the matrix
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tile of A into shared memory
        // Each thread loads one element cooperatively
        // A's tile: row is fixed, column varies with t
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (row < width && aCol < width) {
            // tileA[threadIdx.y][threadIdx.x] = /* YOUR CODE HERE */;
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Load tile of B into shared memory
        // B's tile: row varies with t, column is fixed
        int bRow = t * TILE_WIDTH + threadIdx.y;
        if (/* YOUR CONDITION HERE */) {
            /* YOUR CODE HERE - Load B element */
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Synchronize before using shared memory
        // All threads must finish loading before computation
        /* YOUR CODE HERE */
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            // TODO: Accumulate product from shared memory
            // accumulator += tileA[threadIdx.y][k] * tileB[/* YOUR CODE HERE */];
        }
        
        // TODO: Synchronize before loading next tile
        /* YOUR CODE HERE */
    }
    
    // Write result to global memory
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 2: Tiled MatMul with 1D Block
 * Use 1D block configuration for potentially better occupancy
 * TODO: Complete the 1D thread mapping
// ============================================================================
__global__ void tiledMatMul1D(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    // TODO: Map 1D thread index to 2D tile coordinates
    int tid = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + (tid / TILE_WIDTH);
    int col = blockIdx.x * TILE_WIDTH + (tid % TILE_WIDTH);
    
    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tiles with 1D thread indexing
        // Each thread loads one element
        int aCol = t * TILE_WIDTH + (tid % TILE_WIDTH);
        int bRow = t * TILE_WIDTH + (tid / TILE_WIDTH);
        
        // Load A and B tiles
        if (row < width && aCol < width) {
            tileA[tid / TILE_WIDTH][tid % TILE_WIDTH] = A[row * width + aCol];
        } else {
            tileA[tid / TILE_WIDTH][tid % TILE_WIDTH] = 0.0f;
        }
        
        // TODO: Load B tile
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[tid / TILE_WIDTH][k] * tileB[k][tid % TILE_WIDTH];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 3: Tiled MatMul with Boundary Handling
 * Properly handle non-multiple-of-tile-size matrices
 * TODO: Complete the boundary-safe implementation
// ============================================================================
__global__ void tiledMatMulBoundary(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accumulator = 0.0f;
    
    // Calculate actual number of tiles needed
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Carefully handle boundary conditions
        // Check if the element we're loading is within bounds
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        
        // Load A with bounds checking
        if (row < width && aCol < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Load B with bounds checking
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute - only accumulate valid elements
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // TODO: Store with bounds checking
    if (row < width && col < width) {
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 4: Multi-Stage Tiled MatMul
 * Separate loading and computation into different kernel launches
 * TODO: Complete the multi-stage approach
// ============================================================================

// First kernel: Load tiles and compute partial results
__global__ void tiledMatMulStage1(float *A, float *B, float *partial, 
                                   int width, int numTiles) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tileId = blockIdx.z;  // Which tile pair we're processing
    
    float accumulator = 0.0f;
    
    // TODO: Load specific tile pair based on tileId
    int t = tileId;
    int aCol = t * TILE_WIDTH + threadIdx.x;
    int bRow = t * TILE_WIDTH + threadIdx.y;
    
    // Load tiles
    if (row < width && aCol < width) {
        tileA[threadIdx.y][threadIdx.x] = A[row * width + aCol];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // TODO: Load B tile
    /* YOUR CODE HERE */
    
    __syncthreads();
    
    // Compute
    for (int k = 0; k < TILE_WIDTH; k++) {
        accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    
    // Store partial result for this tile
    if (row < width && col < width) {
        partial[tileId * width * width + row * width + col] = accumulator;
    }
}

// Second kernel: Sum partial results
__global__ void tiledMatMulStage2(float *partial, float *C, 
                                   int width, int numTiles) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // TODO: Sum all partial results for this output element
        for (int t = 0; t < numTiles; t++) {
            sum += partial[t * width * width + row * width + col];
        }
        
        C[row * width + col] = sum;
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
        mat[i] = (float)(i % 100);
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
            if (diff > 1e-2f * width) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("=== Matrix Multiplication Level 2: Shared Memory Tiling ===\n\n");
    
    const int WIDTH = MATRIX_SIZE;
    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    initMatrix(h_A, WIDTH, 1.0f);
    initMatrix(h_B, WIDTH, 2.0f);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
                 (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);
    
    printf("Matrix size: %d x %d\n", WIDTH, WIDTH);
    printf("Tile size: %d x %d\n", TILE_WIDTH, TILE_WIDTH);
    printf("Grid: %d x %d blocks\n\n", gridDim.x, gridDim.y);
    
    // Test 1: Basic tiled matrix multiplication
    printf("Test 1: Basic tiled matrix multiplication\n");
    tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the tile loading and computation\n");
    }
    
    // Test 2: Tiled with 1D block
    printf("\nTest 2: Tiled matrix multiplication (1D block)\n");
    dim3 blockDim1D(TILE_WIDTH * TILE_WIDTH);
    tiledMatMul1D<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the 1D thread mapping\n");
    }
    
    // Test 3: Tiled with boundary handling
    printf("\nTest 3: Tiled with boundary handling (non-multiple size)\n");
    int oddWidth = 500;  // Not a multiple of TILE_WIDTH
    size_t oddSize = oddWidth * oddWidth * sizeof(float);
    float *h_A_odd = (float*)malloc(oddSize);
    float *h_B_odd = (float*)malloc(oddSize);
    float *h_C_odd = (float*)malloc(oddSize);
    float *d_A_odd, *d_B_odd, *d_C_odd;
    
    initMatrix(h_A_odd, oddWidth, 1.0f);
    initMatrix(h_B_odd, oddWidth, 2.0f);
    
    cudaMalloc(&d_A_odd, oddSize);
    cudaMalloc(&d_B_odd, oddSize);
    cudaMalloc(&d_C_odd, oddSize);
    cudaMemcpy(d_A_odd, h_A_odd, oddSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_odd, h_B_odd, oddSize, cudaMemcpyHostToDevice);
    
    dim3 gridDimOdd((oddWidth + TILE_WIDTH - 1) / TILE_WIDTH,
                    (oddWidth + TILE_WIDTH - 1) / TILE_WIDTH);
    
    tiledMatMulBoundary<<<gridDimOdd, blockDim>>>(d_A_odd, d_B_odd, d_C_odd, oddWidth);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C_odd, d_C_odd, oddSize, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C_odd, h_A_odd, h_B_odd, oddWidth)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Fix boundary handling\n");
    }
    
    // Test 4: Multi-stage tiled matmul
    printf("\nTest 4: Multi-stage tiled matrix multiplication\n");
    int numTiles = (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH;
    float *d_partial;
    cudaMalloc(&d_partial, numTiles * size);
    
    // Launch first stage for each tile
    for (int t = 0; t < numTiles; t++) {
        dim3 gridDimStage(gridDim.x, gridDim.y, 1);
        tiledMatMulStage1<<<gridDimStage, blockDim>>>(d_A, d_B, d_partial, WIDTH, numTiles);
    }
    cudaDeviceSynchronize();
    
    // Launch second stage to sum partials
    tiledMatMulStage2<<<gridDim, blockDim>>>(d_partial, d_C, WIDTH, numTiles);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete both stages\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_odd);
    free(h_B_odd);
    free(h_C_odd);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_odd);
    cudaFree(d_B_odd);
    cudaFree(d_C_odd);
    cudaFree(d_partial);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Tiling reduces global memory accesses by reusing data\n");
    printf("- Each tile is loaded once to shared memory, used TILE_WIDTH times\n");
    printf("- Synchronization is critical between tile loading and computation\n");
    printf("- Boundary handling requires careful padding with zeros\n");
    printf("\nNext: Try level3_optimized_tiled.cu for bank conflict avoidance\n");
    
    return 0;
}
