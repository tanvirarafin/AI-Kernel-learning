/*
 * Level 1: Basic Thread Indexing - Kernel 3: 3D Thread Configuration
 *
 * This kernel demonstrates 3D thread indexing for processing volume data
 * mapped to a 1D array.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 3: 3D Thread Configuration for Volume Data
// Calculate 3D coordinates and map to linear index
// ============================================================================
__global__ void process3D(float *output, int width, int height, int depth) {
    // Calculate x, y, z coordinates from 3D thread configuration
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate linear index from 3D coordinates (z-major order)
    int idx = z * (width * height) + y * width + x;

    // Bounds check for all three dimensions
    if (x < width && y < height && z < depth) {
        output[idx] = (z * width * height + y * width + x) * 2.0f;
    }
}

// Utility functions
bool verify3D(float *result, int width, int height, int depth) {
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = z * width * height + y * width + x;
                if (result[idx] != idx * 2.0f) return false;
            }
        }
    }
    return true;
}

void printSlice(float *arr, int width, int height, int z, const char *label) {
    printf("%s (z=%d slice, top-left 5x5 corner):\n", label, z);
    for (int row = 0; row < 5 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 5 && col < width; col++) {
            int idx = z * width * height + row * width + col;
            printf("%8.2f ", arr[idx]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 1: 3D Thread Configuration ===\n\n");

    // Test 3D configuration
    const int W = 8, H = 8, D = 8;
    const int N3D = W * H * D;
    float *d_out3D;
    cudaMalloc(&d_out3D, N3D * sizeof(float));
    cudaMemset(d_out3D, 0, N3D * sizeof(float));

    dim3 block3D(4, 4, 4);
    dim3 grid3D((W + 3) / 4, (H + 3) / 4, (D + 3) / 4);

    printf("Launching process3D kernel...\n");
    printf("  Volume size: %d x %d x %d = %d elements\n", W, H, D, N3D);
    printf("  Block size: %d x %d x %d threads\n", block3D.x, block3D.y, block3D.z);
    printf("  Grid size: %d x %d x %d blocks\n\n", grid3D.x, grid3D.y, grid3D.z);

    process3D<<<grid3D, block3D>>>(d_out3D, W, H, D);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out3D);
        return 1;
    }

    float *h_result = (float*)malloc(N3D * sizeof(float));
    cudaMemcpy(h_result, d_out3D, N3D * sizeof(float), cudaMemcpyDeviceToHost);

    printSlice(h_result, W, H, 0, "Results");

    if (verify3D(h_result, W, H, D)) {
        printf("\n✓ 3D indexing PASSED\n");
    } else {
        printf("\n✗ 3D indexing FAILED - Check your 3D to 1D mapping\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_out3D);

    printf("\n=== Level 1.3 Complete ===\n");
    printf("Next: Try level1_basic_indexing_04_blockInfo2D.cu for block index calculation\n");

    return 0;
}
