/*
 * Level 1: Basic Thread Indexing - Kernel 2: 2D Thread Configuration
 *
 * This kernel demonstrates 2D thread indexing for processing 2D data
 * mapped to a 1D array (row-major order).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 2: 2D Thread Configuration for 1D Data
// Calculate 2D coordinates and map to linear index
// ============================================================================
__global__ void process2D(float *output, int width, int height) {
    // Calculate column and row from 2D thread configuration
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate linear index from 2D coordinates (row-major order)
    int idx = row * width + col;

    // Bounds check
    if (idx < width * height) {
        output[idx] = (row * width + col) * 2.0f;
    }
}

// Utility functions
bool verify2D(float *result, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (result[idx] != idx * 2.0f) return false;
        }
    }
    return true;
}

void print2DCorner(float *arr, int width, int height, const char *label) {
    printf("%s (top-left 5x5 corner):\n", label);
    for (int row = 0; row < 5 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 5 && col < width; col++) {
            printf("%8.2f ", arr[row * width + col]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 1: 2D Thread Configuration ===\n\n");

    // Test 2D configuration
    const int WIDTH = 32, HEIGHT = 32;
    const int N2D = WIDTH * HEIGHT;
    float *d_out2D;
    cudaMalloc(&d_out2D, N2D * sizeof(float));
    cudaMemset(d_out2D, 0, N2D * sizeof(float));

    dim3 block2D(16, 16);
    dim3 grid2D((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    printf("Launching process2D kernel...\n");
    printf("  Matrix size: %d x %d = %d elements\n", WIDTH, HEIGHT, N2D);
    printf("  Block size: %d x %d threads\n", block2D.x, block2D.y);
    printf("  Grid size: %d x %d blocks\n\n", grid2D.x, grid2D.y);

    process2D<<<grid2D, block2D>>>(d_out2D, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out2D);
        return 1;
    }

    float *h_result = (float*)malloc(N2D * sizeof(float));
    cudaMemcpy(h_result, d_out2D, N2D * sizeof(float), cudaMemcpyDeviceToHost);

    print2DCorner(h_result, WIDTH, HEIGHT, "Results");

    if (verify2D(h_result, WIDTH, HEIGHT)) {
        printf("\n✓ 2D indexing PASSED\n");
    } else {
        printf("\n✗ 2D indexing FAILED - Check your 2D to 1D mapping\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out2D);

    printf("\n=== Level 1.2 Complete ===\n");
    printf("Next: Try level1_basic_indexing_03_process3D.cu for 3D thread configuration\n");

    return 0;
}
