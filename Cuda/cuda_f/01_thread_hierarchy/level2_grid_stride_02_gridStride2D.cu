/*
 * Level 2: Grid-Stride Loop Pattern - Kernel 2: 2D Grid-Stride Loop
 *
 * This kernel demonstrates the 2D grid-stride loop pattern for processing
 * 2D data (like images) with a fixed grid configuration.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 2: 2D Grid-Stride Loop for Image Processing
// Each thread processes multiple elements in both dimensions
// ============================================================================
__global__ void gridStride2D(float *output, int width, int height, float multiplier) {
    // Calculate starting column and row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate strides in each dimension
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    // Implement 2D grid-stride loop
    for (int y = row; y < height; y += strideY) {
        for (int x = col; x < width; x += strideX) {
            int idx = y * width + x;
            output[idx] = (y * width + x) * multiplier;
        }
    }
}

// Utility functions
bool verify2D(float *result, int width, int height, float mult) {
    for (int i = 0; i < width * height; i++) {
        if (result[i] != i * mult) return false;
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
    printf("=== Thread Hierarchy Level 2: 2D Grid-Stride Loop ===\n\n");

    // Test 2D grid-stride
    const int W = 256, H = 256;
    const int N2D = W * H;
    float *d_out;
    cudaMalloc(&d_out, N2D * sizeof(float));

    dim3 block2D(16, 16);
    dim3 grid2D(8, 8);  // Small grid forces multiple iterations

    printf("Launching gridStride2D kernel...\n");
    printf("  Image size: %d x %d = %d elements\n", W, H, N2D);
    printf("  Block size: %d x %d threads\n", block2D.x, block2D.y);
    printf("  Grid size: %d x %d blocks\n", grid2D.x, grid2D.y);
    printf("  Stride X: %d, Stride Y: %d\n\n", 
           block2D.x * grid2D.x, block2D.y * grid2D.y);

    cudaMemset(d_out, 0, N2D * sizeof(float));
    gridStride2D<<<grid2D, block2D>>>(d_out, W, H, 2.5f);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    float *h_result = (float*)malloc(N2D * sizeof(float));
    cudaMemcpy(h_result, d_out, N2D * sizeof(float), cudaMemcpyDeviceToHost);

    print2DCorner(h_result, W, H, "Results");

    if (verify2D(h_result, W, H, 2.5f)) {
        printf("\n✓ 2D grid-stride PASSED\n");
    } else {
        printf("\n✗ 2D grid-stride FAILED - Check your 2D stride implementation\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out);

    printf("\n=== Level 2.2 Complete ===\n");
    printf("Next: Try level2_grid_stride_03_gridStrideAccumulate.cu for accumulation pattern\n");

    return 0;
}
