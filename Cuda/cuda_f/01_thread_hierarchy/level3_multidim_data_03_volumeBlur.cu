/*
 * Level 3: Multi-Dimensional Data Processing - Kernel 3: 3D Volume Blur
 *
 * This kernel applies a simple 3x3x3 Gaussian-like blur to 3D volume data
 * using shared neighborhood access.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 3: 3D Volume Gaussian Blur (Simplified)
// Apply a simple 3x3x3 blur kernel to volume data
// ============================================================================
__global__ void volumeBlur(float *input, float *output, int width, int height, int depth) {
    // Calculate voxel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Add bounds check (skip boundary voxels for simplicity)
    if (x > 0 && x < width-1 && y > 0 && y < height-1 && z > 0 && z < depth-1) {
        float sum = 0.0f;
        int count = 0;

        // Loop over 3x3x3 neighborhood and accumulate
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    int nidx = nz * (width * height) + ny * width + nx;
                    sum += input[nidx];
                    count++;
                }
            }
        }

        output[z * width * height + y * width + x] = sum / count;
    }
}

// Utility functions
void initVolume(float *vol, int width, int height, int depth) {
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = z * width * height + y * width + x;
                vol[idx] = (x + y + z) * 0.1f;
            }
        }
    }
}

bool verifyBlur(float *result, float *input, int width, int height, int depth) {
    for (int z = 1; z < depth-1; z++) {
        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                int idx = z * width * height + y * width + x;
                float expected = 0.0f;
                int count = 0;
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nidx = (z+dz) * width * height + (y+dy) * width + (x+dx);
                            expected += input[nidx];
                            count++;
                        }
                    }
                }
                expected /= count;
                if (fabsf(result[idx] - expected) > 1e-5f) return false;
            }
        }
    }
    return true;
}

void printSlice(float *vol, int width, int height, int z, const char *label) {
    printf("%s (z=%d slice, top-left 5x5):\n", label, z);
    for (int row = 0; row < 5 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 5 && col < width; col++) {
            int idx = z * width * height + row * width + col;
            printf("%6.3f ", vol[idx]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 3: 3D Volume Blur ===\n\n");

    // Test volume blur
    const int W = 16, H = 16, D = 8;
    const int N3D = W * H * D;
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    h_input = (float*)malloc(N3D * sizeof(float));
    h_output = (float*)malloc(N3D * sizeof(float));
    initVolume(h_input, W, H, D);

    cudaMalloc(&d_input, N3D * sizeof(float));
    cudaMalloc(&d_output, N3D * sizeof(float));
    cudaMemcpy(d_input, h_input, N3D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(4, 4, 4);
    dim3 grid((W + 3) / 4, (H + 3) / 4, (D + 3) / 4);

    printf("Launching volumeBlur kernel...\n");
    printf("  Volume size: %d x %d x %d = %d voxels\n", W, H, D, N3D);
    printf("  Block size: %d x %d x %d threads\n", block.x, block.y, block.z);
    printf("  Grid size: %d x %d x %d blocks\n\n", grid.x, grid.y, grid.z);

    printSlice(h_input, W, H, 2, "Original volume");

    volumeBlur<<<grid, block>>>(d_input, d_output, W, H, D);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return 1;
    }

    cudaMemcpy(h_output, d_output, N3D * sizeof(float), cudaMemcpyDeviceToHost);

    printSlice(h_output, W, H, 2, "Blurred volume");

    if (verifyBlur(h_output, h_input, W, H, D)) {
        printf("\n✓ Volume blur PASSED\n");
    } else {
        printf("\n✗ Volume blur FAILED - Check 3D neighborhood indexing\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Level 3.3 Complete ===\n");
    printf("Next: Try level3_multidim_data_04_imageConvolution.cu for 2D convolution\n");

    return 0;
}
