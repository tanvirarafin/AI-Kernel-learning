/*
 * Level 2: Grid-Stride Loop Pattern - Kernel 1: Basic Grid-Stride Loop
 *
 * This kernel demonstrates the grid-stride loop pattern which allows
 * a fixed grid configuration to process datasets of any size.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Basic Grid-Stride Loop
// Each thread processes multiple elements spaced by stride
// ============================================================================
__global__ void gridStrideBasic(float *output, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate grid stride (total threads across all blocks)
    int stride = blockDim.x * gridDim.x;

    // Implement grid-stride loop
    // Each thread processes multiple elements spaced by 'stride'
    for (int i = idx; i < n; i += stride) {
        output[i] = i * 3.0f;
    }
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

bool verifyBasic(float *result, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != i * 3.0f) return false;
    }
    return true;
}

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Thread Hierarchy Level 2: Basic Grid-Stride Loop ===\n\n");

    // Test basic grid-stride with small grid (forces multiple iterations)
    const int N = 10000;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Use small grid to force multiple iterations per thread
    int blockSize = 256;
    int gridSize = 10;  // Only 10 blocks, so each thread handles multiple elements

    printf("Launching gridStrideBasic kernel...\n");
    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Grid size: %d blocks\n", gridSize);
    printf("  Stride: %d (each thread processes ~%d elements)\n\n", 
           blockSize * gridSize, (N + blockSize * gridSize - 1) / (blockSize * gridSize));

    cudaMemset(d_out, 0, N * sizeof(float));
    gridStrideBasic<<<gridSize, blockSize>>>(d_out, N);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printFirstElements(h_result, N, "First 10 results");

    if (verifyBasic(h_result, N)) {
        printf("\n✓ Basic grid-stride PASSED\n");
    } else {
        printf("\n✗ Basic grid-stride FAILED - Check your stride calculation\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out);

    printf("\n=== Level 2.1 Complete ===\n");
    printf("Next: Try level2_grid_stride_02_gridStride2D.cu for 2D grid-stride pattern\n");

    return 0;
}
