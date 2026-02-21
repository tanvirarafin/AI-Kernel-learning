/*
 * Level 1: Basic Thread Indexing - Kernel 1: 1D Thread Configuration
 *
 * This kernel demonstrates basic 1D thread indexing for linear data processing.
 * Each thread computes a global index and processes one element.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: 1D Thread Configuration
// Calculate global index for 1D configuration and process data
// ============================================================================
__global__ void process1D(float *output, int n) {
    // Calculate global thread index for 1D configuration
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check before writing
    if (idx < n) {
        output[idx] = idx * 2.0f;
    }
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

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Thread Hierarchy Level 1: 1D Thread Configuration ===\n\n");

    // Test 1D configuration
    const int N = 1024;
    float *d_out1D;
    cudaMalloc(&d_out1D, N * sizeof(float));
    cudaMemset(d_out1D, 0, N * sizeof(float));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    printf("Launching process1D kernel...\n");
    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Grid size: %d blocks\n\n", gridSize);

    process1D<<<gridSize, blockSize>>>(d_out1D, N);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out1D);
        return 1;
    }

    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_out1D, N * sizeof(float), cudaMemcpyDeviceToHost);

    printFirstElements(h_result, N, "First 10 results");

    if (verify1D(h_result, N)) {
        printf("\n✓ 1D indexing PASSED\n");
    } else {
        printf("\n✗ 1D indexing FAILED - Check your index calculation\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out1D);

    printf("\n=== Level 1.1 Complete ===\n");
    printf("Next: Try level1_basic_indexing_02_process2D.cu for 2D thread configuration\n");

    return 0;
}
