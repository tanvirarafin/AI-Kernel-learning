/*
 * Memory Coalescing Level 2: Matrix Transpose
 * 
 * EXERCISE: Implement efficient matrix transpose using shared memory
 * to achieve coalesced reads AND writes.
 * 
 * SKILLS PRACTICED:
 * - Shared memory tiling
 * - Bank conflict avoidance
 * - Coalesced read/write patterns
 * - Memory throughput optimization
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

// ============================================================================
// KERNEL 1: Naive Transpose (Baseline - UNOPTIMIZED)
 * This has uncoalesced writes - use as performance baseline
// ============================================================================
__global__ void naiveTranspose(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Read is coalesced (row-major), but WRITE is uncoalesced!
        output[x * height + y] = input[y * width + x];
    }
}

// ============================================================================
// KERNEL 2: Shared Memory Transpose (Incomplete)
 * Use shared memory to achieve coalesced reads AND writes
// TODO: Complete the implementation
// ============================================================================
__global__ void sharedMemTranspose(float *input, float *output, int width, int height) {
    // TODO: Declare shared memory tile
    // Hint: __shared__ float tile[TILE_SIZE][TILE_SIZE];
    /* YOUR DECLARATION HERE */;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: Load data into shared memory (coalesced read)
    // Each thread loads one element
    if (x < width && y < height) {
        // TODO: Store in tile with correct indexing
        // tile[threadIdx.y][threadIdx.x] = /* YOUR CODE HERE */;
    }
    
    __syncthreads();
    
    // TODO: Calculate transposed coordinates for writing
    // int transposedX = /* YOUR CODE HERE */;
    // int transposedY = /* YOUR CODE HERE */;
    
    // TODO: Read from shared memory and write to output (coalesced write)
    // if (transposedX < height && transposedY < width) {
    //     output[transposedY * height + transposedX] = /* YOUR CODE HERE */;
    // }
}

// ============================================================================
// KERNEL 3: Bank Conflict-Free Transpose (Challenge)
 * Add padding to avoid shared memory bank conflicts
// ============================================================================
__global__ void bankConflictFreeTranspose(float *input, float *output, int width, int height) {
    // TODO: Declare shared memory with padding to avoid bank conflicts
    // Hint: Add one extra column: tile[TILE_SIZE][TILE_SIZE + 1]
    /* YOUR DECLARATION HERE */;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // TODO: Load with padding consideration
        // tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // TODO: Calculate transposed coordinates
    int transposedX = threadIdx.y;
    int transposedY = threadIdx.x;
    
    // TODO: Write with padding consideration for the read
    // output[transposedY * height + transposedX] = tile[transposedX][transposedY];
}

// ============================================================================
// KERNEL 4: 1D Block Transpose (Optimization)
 * Use 1D block configuration for better occupancy
// ============================================================================
__global__ void transpose1DBlock(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + (threadIdx.x / TILE_SIZE);  // 1D to 2D mapping
    
    // TODO: Complete the implementation
    // Load, sync, transpose, sync, store
}

// Utility functions
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n; i++) {
        mat[i] = i * 0.5f;
    }
}

bool verifyTranspose(float *result, float *expected, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

void transposeCPU(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

int main() {
    printf("=== Memory Coalescing Level 2: Matrix Transpose ===\n\n");
    
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int N = WIDTH * HEIGHT;
    
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));
    
    initMatrix(h_input, N);
    transposeCPU(h_input, h_expected, WIDTH, HEIGHT);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE, 
              (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);
    
    // Test naive transpose (baseline)
    printf("Testing naive transpose (baseline)...\n");
    cudaMemset(d_output, 0, N * sizeof(float));
    naiveTranspose<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyTranspose(h_output, h_expected, N)) {
        printf("✓ Naive transpose correctness PASSED\n");
    } else {
        printf("✗ Naive transpose correctness FAILED\n");
    }
    
    // Test shared memory transpose
    printf("\nTesting shared memory transpose...\n");
    cudaMemset(d_output, 0, N * sizeof(float));
    sharedMemTranspose<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyTranspose(h_output, h_expected, N)) {
        printf("✓ Shared memory transpose correctness PASSED\n");
    } else {
        printf("✗ Shared memory transpose correctness FAILED - Complete the implementation\n");
    }
    
    // Test bank-conflict-free transpose
    printf("\nTesting bank-conflict-free transpose...\n");
    cudaMemset(d_output, 0, N * sizeof(float));
    bankConflictFreeTranspose<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyTranspose(h_output, h_expected, N)) {
        printf("✓ Bank-conflict-free transpose correctness PASSED\n");
    } else {
        printf("✗ Bank-conflict-free transpose FAILED - Add padding correctly\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n=== Level 2 Complete ===\n");
    printf("Next: Try level3_soa_aos.cu for data layout optimization\n");
    
    return 0;
}
