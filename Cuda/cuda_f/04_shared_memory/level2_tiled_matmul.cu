/*
 * Shared Memory Level 2: Tiled Matrix Multiplication
 * 
 * EXERCISE: Implement tiled matrix multiplication using shared memory
 * to reduce global memory accesses and improve performance.
 * 
 * SKILLS PRACTICED:
 * - Tiled algorithm implementation
 * - Shared memory for data reuse
 * - Multiple synchronization points
 * - Computational intensity optimization
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32

// ============================================================================
// KERNEL 1: Naive Matrix Multiplication (Baseline)
 * No shared memory - loads from global memory repeatedly
// ============================================================================
__global__ void naiveMatMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 2: Tiled Matrix Multiplication (Incomplete)
 * Use shared memory tiles to reduce global memory bandwidth
// TODO: Complete the implementation
// ============================================================================
__global__ void tiledMatMul(float *A, float *B, float *C, int width) {
    // TODO: Declare shared memory for A and B tiles
    // __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    // __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    /* YOUR DECLARATIONS HERE */;
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // TODO: Load tile of A into shared memory
        // Each thread loads one element of the A tile
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (row < width && aCol < width) {
            // As[threadIdx.y][threadIdx.x] = /* YOUR CODE HERE */;
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Load tile of B into shared memory
        int bRow = t * TILE_WIDTH + threadIdx.y;
        if (/* YOUR CONDITION HERE */) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * width + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Synchronize before using shared memory
        // __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            // TODO: Accumulate product from shared memory
            sum += /* YOUR CODE HERE */;
        }
        
        // TODO: Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 3: Tiled MatMul with 1D Block (Challenge)
 * Use 1D block configuration for potentially better occupancy
// ============================================================================
__global__ void tiledMatMul1D(float *A, float *B, float *C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + (threadIdx.x / TILE_WIDTH);
    int col = blockIdx.x * TILE_WIDTH + (threadIdx.x % TILE_WIDTH);
    
    float sum = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // TODO: Complete the tiled multiplication with 1D thread mapping
    // Similar to tiledMatMul but with 1D thread indexing
    for (int t = 0; t < numTiles; t++) {
        // Load tiles, compute, synchronize
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 4: Register Tiling Optimization (Advanced)
 * Each thread computes multiple output elements using larger tiles
// ============================================================================
__global__ void registerTiledMatMul(float *A, float *B, float *C, int width) {
    // Each thread computes a 2x2 block of the output
    const int THREAD_TILE = 2;
    __shared__ float As[TILE_WIDTH * THREAD_TILE][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * THREAD_TILE];
    
    int row = blockIdx.y * (TILE_WIDTH * THREAD_TILE) + threadIdx.y * THREAD_TILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // TODO: Implement register tiling where each thread computes multiple outputs
}

// Utility functions
void initMatrix(float *mat, int width, float val) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = val;
    }
}

bool verifyMatMul(float *result, float *A, float *B, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0.0f;
            for (int k = 0; k < width; k++) {
                expected += A[row * width + k] * B[k * width + col];
            }
            if (fabsf(result[row * width + col] - expected) > 1e-3f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Shared Memory Level 2: Tiled Matrix Multiplication ===\n\n");
    
    const int WIDTH = 512;
    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_expected = (float*)malloc(size);
    
    initMatrix(h_A, WIDTH, 1.0f);
    initMatrix(h_B, WIDTH, 2.0f);
    
    // Compute expected result on CPU
    for (int row = 0; row < WIDTH; row++) {
        for (int col = 0; col < WIDTH; col++) {
            float sum = 0.0f;
            for (int k = 0; k < WIDTH; k++) {
                sum += h_A[row * WIDTH + k] * h_B[k * WIDTH + col];
            }
            h_expected[row * WIDTH + col] = sum;
        }
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, 
              (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Test naive (baseline)
    printf("Testing naive matrix multiplication...\n");
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("✓ Naive matmul correctness PASSED\n");
    } else {
        printf("✗ Naive matmul correctness FAILED\n");
    }
    
    // Test tiled matmul
    printf("\nTesting tiled matrix multiplication...\n");
    cudaMemset(d_C, 0, size);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("✓ Tiled matmul correctness PASSED\n");
    } else {
        printf("✗ Tiled matmul correctness FAILED - Complete the implementation\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_expected);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n=== Level 2 Complete ===\n");
    printf("Next: Try level3_bank_conflicts.cu for bank conflict resolution\n");
    
    return 0;
}
