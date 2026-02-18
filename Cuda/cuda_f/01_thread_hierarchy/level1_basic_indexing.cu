/*
 * Level 1: Basic Thread Indexing
 * 
 * EXERCISE: Complete the thread indexing calculations for different
 * dimensional configurations. Each kernel processes data using a
 * different thread organization pattern.
 * 
 * SKILLS PRACTICED:
 * - 1D, 2D, 3D thread indexing
 * - Bounds checking
 * - Linear indexing from multi-dimensional threads
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: 1D Thread Configuration
// Complete the global index calculation
// ============================================================================
__global__ void process1D(float *output, int n) {
    // TODO: Calculate global thread index for 1D configuration
    // Hint: global_index = blockIdx.x * blockDim.x + threadIdx.x
    int idx = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check before writing
    // if (/* YOUR CODE HERE */) {
    if (idx < n) {
        output[idx] = idx * 2.0f;
    }
}

// ============================================================================
// KERNEL 2: 2D Thread Configuration for 1D Data
// Complete the 2D to 1D index mapping
// ============================================================================
__global__ void process2D(float *output, int width, int height) {
    // TODO: Calculate column and row from 2D thread configuration
    int col = /* YOUR CODE HERE */ 0;
    int row = /* YOUR CODE HERE */ 0;
    
    // TODO: Calculate linear index from 2D coordinates
    // Hint: idx = row * width + col
    int idx = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check
    if (/* YOUR CODE HERE */) {
        output[idx] = (row * width + col) * 2.0f;
    }
}

// ============================================================================
// KERNEL 3: 3D Thread Configuration for Volume Data
// Complete the 3D to 1D index mapping
// ============================================================================
__global__ void process3D(float *output, int width, int height, int depth) {
    // TODO: Calculate x, y, z coordinates from 3D thread configuration
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    int z = /* YOUR CODE HERE */ 0;
    
    // TODO: Calculate linear index from 3D coordinates
    // Hint: idx = z * (width * height) + y * width + x
    int idx = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check for all three dimensions
    if (/* YOUR CODE HERE */) {
        output[idx] = (z * width * height + y * width + x) * 2.0f;
    }
}

// ============================================================================
// KERNEL 4: Block Index Calculation (Challenge)
// Calculate which block a thread belongs to in a 2D grid
// ============================================================================
__global__ void blockInfo2D(int *blockIds, int gridWidth, int gridHeight) {
    // TODO: Get this thread's block coordinates
    int blockX = /* YOUR CODE HERE */ 0;
    int blockY = /* YOUR CODE HERE */ 0;
    
    // TODO: Get this thread's position within the block
    int threadX = /* YOUR CODE HERE */ 0;
    int threadY = /* YOUR CODE HERE */ 0;
    
    // TODO: Calculate unique block ID (row-major order)
    int blockId = /* YOUR CODE HERE */ 0;
    
    int idx = blockY * gridWidth + blockX;
    blockIds[idx] = blockId;
}

// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = 0.0f;
}

bool verify1D(float *result, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != i * 2.0f) return false;
    }
    return true;
}

bool verify2D(float *result, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (result[idx] != idx * 2.0f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Thread Hierarchy Level 1: Basic Indexing ===\n\n");
    
    // Test 1D configuration
    printf("Testing 1D thread configuration...\n");
    const int N = 1024;
    float *d_out1D;
    cudaMalloc(&d_out1D, N * sizeof(float));
    cudaMemset(d_out1D, 0, N * sizeof(float));
    
    process1D<<<(N + 255) / 256, 256>>>(d_out1D, N);
    cudaDeviceSynchronize();
    
    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out1D, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify1D(h_result, N)) {
        printf("✓ 1D indexing PASSED\n");
    } else {
        printf("✗ 1D indexing FAILED - Check your index calculation\n");
    }
    
    // Test 2D configuration
    printf("\nTesting 2D thread configuration...\n");
    const int WIDTH = 32, HEIGHT = 32;
    const int N2D = WIDTH * HEIGHT;
    float *d_out2D;
    cudaMalloc(&d_out2D, N2D * sizeof(float));
    cudaMemset(d_out2D, 0, N2D * sizeof(float));
    
    dim3 block2D(16, 16);
    dim3 grid2D((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    process2D<<<grid2D, block2D>>>(d_out2D, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_out2D, N2D * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verify2D(h_result, WIDTH, HEIGHT)) {
        printf("✓ 2D indexing PASSED\n");
    } else {
        printf("✗ 2D indexing FAILED - Check your 2D to 1D mapping\n");
    }
    
    // Test 3D configuration
    printf("\nTesting 3D thread configuration...\n");
    const int W = 8, H = 8, D = 8;
    const int N3D = W * H * D;
    float *d_out3D;
    cudaMalloc(&d_out3D, N3D * sizeof(float));
    cudaMemset(d_out3D, 0, N3D * sizeof(float));
    
    dim3 block3D(4, 4, 4);
    dim3 grid3D((W + 3) / 4, (H + 3) / 4, (D + 3) / 4);
    process3D<<<grid3D, block3D>>>(d_out3D, W, H, D);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_out3D, N3D * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass3D = true;
    for (int z = 0; z < D && pass3D; z++) {
        for (int y = 0; y < H && pass3D; y++) {
            for (int x = 0; x < W && pass3D; x++) {
                int idx = z * W * H + y * W + x;
                if (h_result[idx] != idx * 2.0f) pass3D = false;
            }
        }
    }
    
    if (pass3D) {
        printf("✓ 3D indexing PASSED\n");
    } else {
        printf("✗ 3D indexing FAILED - Check your 3D to 1D mapping\n");
    }
    
    // Cleanup
    free(h_result);
    cudaFree(d_out1D);
    cudaFree(d_out2D);
    cudaFree(d_out3D);
    
    printf("\n=== Level 1 Complete ===\n");
    printf("Next: Try level2_grid_stride.cu for scalable kernel patterns\n");
    
    return 0;
}
