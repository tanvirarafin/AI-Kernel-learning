/*
 * Level 2: Grid-Stride Loop Pattern - Kernel 4: Flexible Grid-Stride
 *
 * This kernel demonstrates a flexible grid-stride implementation that
 * works with any grid/block configuration automatically.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 4: Flexible Grid-Stride (Challenge)
// Works with any grid/block configuration automatically
// ============================================================================
__global__ void flexibleGridStride(float *output, int n, float value) {
    // Calculate index using all available dimension information
    // Support both 1D and 2D grid configurations
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int idx = blockId * blockDim.x * blockDim.y + threadId;

    // Calculate total stride for any grid configuration
    int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    // Implement the loop
    for (int i = idx; i < n; i += stride) {
        output[i] = value;
    }
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

bool verifyFlexible(float *result, int n, float expected) {
    for (int i = 0; i < n; i++) {
        if (result[i] != expected) return false;
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
    printf("=== Thread Hierarchy Level 2: Flexible Grid-Stride ===\n\n");

    // Test flexible grid-stride
    const int N = 10000;
    const float VALUE = 42.0f;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    // Test with 1D configuration
    printf("Test 1: 1D Grid Configuration\n");
    int blockSize1D = 256;
    int gridSize1D = (N + blockSize1D - 1) / blockSize1D;

    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d threads (1D)\n", blockSize1D);
    printf("  Grid size: %d blocks (1D)\n\n", gridSize1D);

    cudaMemset(d_out, 0, N * sizeof(float));
    flexibleGridStride<<<dim3(gridSize1D), dim3(blockSize1D)>>>(d_out, N, VALUE);
    cudaDeviceSynchronize();

    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printFirstElements(h_result, N, "First 10 results");

    if (verifyFlexible(h_result, N, VALUE)) {
        printf("  ✓ 1D flexible grid-stride PASSED\n");
    } else {
        printf("  ✗ 1D flexible grid-stride FAILED\n");
    }

    // Test with 2D configuration
    printf("\nTest 2: 2D Grid Configuration\n");
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D(8, 8);

    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d x %d threads (2D)\n", blockSize2D.x, blockSize2D.y);
    printf("  Grid size: %d x %d blocks (2D)\n\n", gridSize2D.x, gridSize2D.y);

    cudaMemset(d_out, 0, N * sizeof(float));
    flexibleGridStride<<<gridSize2D, blockSize2D>>>(d_out, N, VALUE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printFirstElements(h_result, N, "First 10 results");

    if (verifyFlexible(h_result, N, VALUE)) {
        printf("  ✓ 2D flexible grid-stride PASSED\n");
    } else {
        printf("  ✗ 2D flexible grid-stride FAILED\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out);

    printf("\n=== Level 2.4 Complete ===\n");
    printf("Next: Try level3_multidim_data.cu for image/volume processing\n");

    return 0;
}
