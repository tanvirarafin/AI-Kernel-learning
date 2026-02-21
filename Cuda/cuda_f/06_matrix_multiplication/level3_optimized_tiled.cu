/*
 * Matrix Multiplication Level 3: Optimized Tiling
 *
 * EXERCISE: Optimize tiled matrix multiplication by avoiding bank conflicts
 * and using vectorized memory access.
 *
 * CONCEPTS:
 * - Bank conflict avoidance with padding
 * - Vectorized global memory loads (float4)
 * - Register caching
 * - Improved instruction-level parallelism
 *
 * SKILLS PRACTICED:
 * - Padding for bank conflict avoidance
 * - Vectorized memory operations
 * - Micro-optimizations
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 512
#define TILE_WIDTH 32

// ============================================================================
// KERNEL 1: Tiled MatMul with Padding (Bank Conflict Free)
 * Add padding to shared memory to eliminate bank conflicts
 * TODO: Complete the padded implementation
// ============================================================================
__global__ void tiledMatMulPadded(float *A, float *B, float *C, int width) {
    // TODO: Add padding to avoid bank conflicts
    // With 32 banks, add 1 column padding: [TILE_WIDTH][TILE_WIDTH + 1]
    // This ensures consecutive threads access different banks
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH + 1];  // Padded!
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH + 1];  // Padded!
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Load with padded indexing
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        
        // Load A tile with padding
        if (row < width && aCol < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Load B tile with padding
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        // Compute with padded indexing
        for (int k = 0; k < TILE_WIDTH; k++) {
            // Note: k index into original tile size, but access with padding
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 2: Vectorized Load Tiled MatMul
 * Use float4 loads for 4x memory bandwidth
 * TODO: Complete the vectorized implementation
// ============================================================================
__global__ void tiledMatMulVectorized(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // TODO: Use vectorized loads when possible
        // Check if we can load 4 floats at once (aligned access)
        int aCol = t * TILE_WIDTH + threadIdx.x;
        
        // For vectorized load, need:
        // 1. Aligned address (divisible by 4)
        // 2. In bounds
        // 3. threadIdx.x % 4 == 0
        
        if (row < width && aCol < width) {
            // TODO: Try vectorized load
            // float4 a_vec = reinterpret_cast<float4*>(&A[row * width + aCol])[0];
            // But need to handle alignment and bounds carefully
            
            // For now, scalar load (optimize this!)
            tileA[threadIdx.y][threadIdx.x] = A[row * width + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // TODO: Load B tile (try vectorized if possible)
        /* YOUR CODE HERE */
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 3: Register Cached MatMul
 * Cache frequently accessed values in registers
 * TODO: Complete the register-optimized version
// ============================================================================
__global__ void tiledMatMulRegisterCached(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH + 1];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    
    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        
        tileA[threadIdx.y][threadIdx.x] = (row < width && aCol < width) 
                                           ? A[row * width + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < width && col < width)
                                           ? B[bRow * width + col] : 0.0f;
        
        __syncthreads();
        
        // TODO: Cache B row in register to reduce shared memory reads
        // Each thread can cache the B value it will reuse
        float b_cache[TILE_WIDTH];  // Register array
        for (int k = 0; k < TILE_WIDTH; k++) {
            b_cache[k] = tileB[k][threadIdx.x];
        }
        
        // Compute using cached B values
        for (int k = 0; k < TILE_WIDTH; k++) {
            // TODO: Use cached b_cache[k] instead of tileB[k][threadIdx.x]
            accumulator += tileA[threadIdx.y][k] * /* YOUR CODE HERE */;
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 4: Loop Unrolling Optimization
 * Manually unroll the inner loop for better ILP
 * TODO: Complete the unrolled implementation
// ============================================================================
__global__ void tiledMatMulUnrolled(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH + 1];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        
        tileA[threadIdx.y][threadIdx.x] = (row < width && aCol < width)
                                           ? A[row * width + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < width && col < width)
                                           ? B[bRow * width + col] : 0.0f;
        
        __syncthreads();
        
        // TODO: Unroll the inner loop
        // Instead of: for (int k = 0; k < TILE_WIDTH; k++)
        // Write out each iteration explicitly for TILE_WIDTH = 32
        // Or use #pragma unroll
        
        // Example for TILE_WIDTH = 4 (expand to 32):
        // accumulator += tileA[threadIdx.y][0] * tileB[0][threadIdx.x];
        // accumulator += tileA[threadIdx.y][1] * tileB[1][threadIdx.x];
        // accumulator += tileA[threadIdx.y][2] * tileB[2][threadIdx.x];
        // accumulator += tileA[threadIdx.y][3] * tileB[3][threadIdx.x];
        
        /* YOUR CODE HERE - Unrolled loop */
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// ============================================================================
// KERNEL 5: Combined Optimizations
 * Apply all optimizations together
 * TODO: Integrate padding, vectorization, and unrolling
// ============================================================================
__global__ void tiledMatMulOptimized(float *A, float *B, float *C, int width) {
    // TODO: Combine all optimizations:
    // 1. Padded shared memory
    // 2. Vectorized loads where possible
    // 3. Register caching
    // 4. Loop unrolling
    
    /* YOUR CODE HERE - Implement fully optimized version */
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
    printf("=== Matrix Multiplication Level 3: Optimized Tiling ===\n\n");
    
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
    
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + 31) / 32, (WIDTH + 31) / 32);
    
    printf("Matrix size: %d x %d\n", WIDTH, WIDTH);
    printf("Tile size: %d x %d (with padding)\n\n", TILE_WIDTH, TILE_WIDTH);
    
    // Test 1: Padded shared memory
    printf("Test 1: Tiled MatMul with padding (bank conflict free)\n");
    tiledMatMulPadded<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Fix the padded implementation\n");
    }
    
    // Test 2: Vectorized loads
    printf("\nTest 2: Tiled MatMul with vectorized loads\n");
    tiledMatMulVectorized<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete vectorized loads\n");
    }
    
    // Test 3: Register cached
    printf("\nTest 3: Tiled MatMul with register caching\n");
    tiledMatMulRegisterCached<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete register caching\n");
    }
    
    // Test 4: Loop unrolling
    printf("\nTest 4: Tiled MatMul with loop unrolling\n");
    tiledMatMulUnrolled<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete loop unrolling\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Padding eliminates bank conflicts (32 banks on most GPUs)\n");
    printf("- Vectorized loads (float4) give 4x memory bandwidth\n");
    printf("- Register caching reduces shared memory reads\n");
    printf("- Loop unrolling improves instruction-level parallelism\n");
    printf("\nNext: Try level4_register_tiling.cu for thread-level optimization\n");
    
    return 0;
}
