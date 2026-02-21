/*
 * Matrix Multiplication Level 4: Register Tiling
 *
 * EXERCISE: Each thread computes multiple output elements using
 * register tiling for higher arithmetic intensity.
 *
 * CONCEPTS:
 * - Register-level tiling
 * - Thread-level parallelism
 * - Increased computational density
 * - Better resource utilization
 *
 * SKILLS PRACTICED:
 * - Multi-output thread computation
 * - Register allocation
 * - Advanced tiling strategies
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 512
#define TILE_WIDTH 16
#define THREAD_TILE 2  // Each thread computes THREAD_TILE x THREAD_TILE outputs

// ============================================================================
// KERNEL 1: 2x2 Register Tiling
 * Each thread computes a 2x2 block of output elements
 * TODO: Complete the register-tiled implementation
// ============================================================================
__global__ void registerTiledMatMul2x2(float *A, float *B, float *C, int width) {
    // Shared memory: each thread needs to load for 2x2 output block
    __shared__ float tileA[TILE_WIDTH * THREAD_TILE][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * THREAD_TILE];
    
    // Calculate starting row and column for this thread's 2x2 block
    int row = blockIdx.y * (TILE_WIDTH * THREAD_TILE) + threadIdx.y * THREAD_TILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accum[THREAD_TILE][THREAD_TILE] = {0.0f};  // 2x2 accumulator in registers
    
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tiles cooperatively
        // Each thread loads elements needed for its 2x2 output block
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y * THREAD_TILE;
        
        // Load A tile (each thread loads 2 rows for its 2x2 block)
        for (int i = 0; i < THREAD_TILE; i++) {
            int loadRow = row + i;
            if (loadRow < width && aCol < width) {
                tileA[threadIdx.y * THREAD_TILE + i][threadIdx.x] = 
                    A[loadRow * width + aCol];
            } else {
                tileA[threadIdx.y * THREAD_TILE + i][threadIdx.x] = 0.0f;
            }
        }
        
        // TODO: Load B tile (each thread loads 2 columns)
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute 2x2 output block
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    accum[i][j] += tileA[threadIdx.y * THREAD_TILE + i][k] *
                                   tileB[k][threadIdx.x * THREAD_TILE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // TODO: Store 2x2 output block
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width) {
                /* YOUR CODE HERE */
            }
        }
    }
}

// ============================================================================
// KERNEL 2: 4x4 Register Tiling (Higher Intensity)
 * Each thread computes a 4x4 block of output elements
 * TODO: Complete the 4x4 register tiling
// ============================================================================
__global__ void registerTiledMatMul4x4(float *A, float *B, float *C, int width) {
    const int RTILE = 4;  // 4x4 register tile
    __shared__ float tileA[TILE_WIDTH * RTILE][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * RTILE];
    
    int row = blockIdx.y * (TILE_WIDTH * RTILE) + threadIdx.y * RTILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // TODO: Declare 4x4 accumulator array in registers
    float accum[RTILE][RTILE];
    /* YOUR CODE HERE - Initialize accumulators to 0 */
    
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tiles for 4x4 output block
        // Each thread loads 4 rows of A and 4 columns of B
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // TODO: Compute 4x4 output block (64 multiply-adds per tile!)
        /* YOUR CODE HERE */
        
        __syncthreads();
    }
    
    // TODO: Store 4x4 output block
    /* YOUR CODE HERE */
}

// ============================================================================
// KERNEL 3: 1D Thread Block with Register Tiling
 * Use 1D thread block for potentially better occupancy
 * TODO: Complete the 1D register-tiled implementation
// ============================================================================
__global__ void registerTiledMatMul1D(float *A, float *B, float *C, int width) {
    const int RTILE = 2;
    const int BLOCK_THREADS = TILE_WIDTH * TILE_WIDTH;
    
    __shared__ float tileA[TILE_WIDTH * RTILE][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * RTILE];
    
    int tid = threadIdx.x;
    int row = blockIdx.y * (TILE_WIDTH * RTILE) + (tid / TILE_WIDTH) * RTILE;
    int col = blockIdx.x * TILE_WIDTH + (tid % TILE_WIDTH);
    
    float accum[RTILE][RTILE] = {0.0f};
    
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load tiles with 1D thread indexing
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < RTILE; i++) {
                for (int j = 0; j < RTILE; j++) {
                    accum[i][j] += tileA[(tid / TILE_WIDTH) * RTILE + i][k] *
                                   tileB[k][(tid % TILE_WIDTH) * RTILE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store
    for (int i = 0; i < RTILE; i++) {
        for (int j = 0; j < RTILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width) {
                C[storeRow * width + storeCol] = accum[i][j];
            }
        }
    }
}

// ============================================================================
// KERNEL 4: Asynchronous Load with Register Tiling
 * Use double-buffering to overlap load and compute
 * TODO: Complete the double-buffered implementation
// ============================================================================
__global__ void registerTiledMatMulAsync(float *A, float *B, float *C, int width) {
    const int RTILE = 2;
    
    // Double-buffered shared memory (2 sets of tiles)
    __shared__ float tileA[2][TILE_WIDTH * RTILE][TILE_WIDTH];
    __shared__ float tileB[2][TILE_WIDTH][TILE_WIDTH * RTILE];
    
    int row = blockIdx.y * (TILE_WIDTH * RTILE) + threadIdx.y * RTILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accum[RTILE][RTILE] = {0.0f};
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // TODO: Implement double-buffering
    // While computing with tile buffer [t % 2], prefetch [(t+1) % 2]
    
    for (int t = 0; t < numTiles; t++) {
        int curBuf = t % 2;
        int nextBuf = (t + 1) % 2;
        
        // TODO: Load current tile into curBuf
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute with current buffer
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < RTILE; i++) {
                for (int j = 0; j < RTILE; j++) {
                    accum[i][j] += tileA[curBuf][threadIdx.y * RTILE + i][k] *
                                   tileB[curBuf][k][threadIdx.x * RTILE + j];
                }
            }
        }
        
        __syncthreads();
        
        // TODO: Could prefetch next tile here (async load)
        // But need CUDA streams or async copy for true overlap
    }
    
    // Store
    for (int i = 0; i < RTILE; i++) {
        for (int j = 0; j < RTILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width) {
                C[storeRow * width + storeCol] = accum[i][j];
            }
        }
    }
}

// Utility functions
void initMatrix(float *mat, int width, float val) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = val;
    }
}

bool verifyMatMul(float *C, float *A, float *B, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0.0f;
            for (int k = 0; k < width; k++) {
                expected += A[row * width + k] * B[k * width + col];
            }
            if (fabsf(C[row * width + col] - expected) > 1e-2f * width) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("=== Matrix Multiplication Level 4: Register Tiling ===\n\n");
    
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
    
    // For 2x2 register tile: each block covers 32x16 output
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
                 (WIDTH + TILE_WIDTH * THREAD_TILE - 1) / (TILE_WIDTH * THREAD_TILE));
    
    printf("Matrix size: %d x %d\n", WIDTH, WIDTH);
    printf("Register tile: %d x %d (each thread computes %d outputs)\n\n", 
           THREAD_TILE, THREAD_TILE, THREAD_TILE * THREAD_TILE);
    
    // Test 1: 2x2 Register tiling
    printf("Test 1: 2x2 Register tiling\n");
    registerTiledMatMul2x2<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the 2x2 register tiling\n");
    }
    
    // Test 2: 4x4 Register tiling
    printf("\nTest 2: 4x4 Register tiling\n");
    const int RTILE4 = 4;
    dim3 gridDim4((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
                  (WIDTH + TILE_WIDTH * RTILE4 - 1) / (TILE_WIDTH * RTILE4));
    
    registerTiledMatMul4x4<<<gridDim4, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the 4x4 register tiling\n");
    }
    
    // Test 3: 1D thread block
    printf("\nTest 3: 1D thread block with register tiling\n");
    dim3 blockDim1D(TILE_WIDTH * TILE_WIDTH);
    registerTiledMatMul1D<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the 1D implementation\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Register tiling increases arithmetic intensity\n");
    printf("- 2x2: Each thread computes 4 outputs (4x more compute per load)\n");
    printf("- 4x4: Each thread computes 16 outputs (16x more compute per load)\n");
    printf("- Higher register usage may reduce occupancy\n");
    printf("\nNext: Try level5_tensor_core_matmul.cu for Tensor Core acceleration\n");
    
    return 0;
}
