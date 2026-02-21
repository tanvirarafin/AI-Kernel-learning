/*
 * Memory Hierarchy Level 1: Global Memory Basics - Kernel 1
 *
 * This kernel demonstrates basic global memory access patterns
 * with proper thread indexing and bounds checking.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define SCALE_FACTOR 2.5f

// ============================================================================
// KERNEL 1: Global Memory Transform
// Copy input to output with scaling factor
// ============================================================================
__global__ void globalMemoryTransform(float *input, float *output, int n, float scale_factor) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check and perform the transformation
    if (idx < n) {
        output[idx] = input[idx] * scale_factor;
    }
}

// ============================================================================
// KERNEL 2: Global Memory Transform Strided
// Handle cases where N > total threads using grid-stride loop
// ============================================================================
__global__ void globalMemoryTransformStrided(float *input, float *output, int n, float scale_factor) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate total grid stride
    int stride = blockDim.x * gridDim.x;

    // Implement strided loop to handle all elements
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * scale_factor;
    }
}

void verifyResults(float *output, int n, float expected) {
    bool success = true;
    for (int i = 0; i < n && i < 10; i++) {
        if (fabsf(output[i] - expected) > 1e-5f) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, output[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("✓ Verification passed!\n");
    }
}

int main() {
    printf("=== Memory Hierarchy Level 1: Global Memory Basics ===\n\n");

    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Test 1: Basic global memory transform\n");
    printf("  Array size: %d elements\n", N);
    printf("  Block size: %d threads\n", threadsPerBlock);
    printf("  Grid size: %d blocks\n", blocksPerGrid);
    printf("  Scale factor: %.2f\n\n", SCALE_FACTOR);

    // Launch kernel
    globalMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, SCALE_FACTOR);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy results back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    printf("Checking first 10 results (expected: %.2f):\n", SCALE_FACTOR);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n\n");
    verifyResults(h_output, N, SCALE_FACTOR);

    // Test strided version
    printf("\nTest 2: Strided global memory transform\n");
    int stridedBlocks = 10;  // Force striding
    printf("  Using only %d blocks to force grid-stride loop\n\n", stridedBlocks);

    cudaMemset(d_output, 0, N * sizeof(float));
    globalMemoryTransformStrided<<<stridedBlocks, threadsPerBlock>>>(d_input, d_output, N, SCALE_FACTOR);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_output, N, SCALE_FACTOR);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    printf("\n=== Level 1 Complete ===\n");
    printf("Global memory exercise completed!\n");

    return 0;
}
