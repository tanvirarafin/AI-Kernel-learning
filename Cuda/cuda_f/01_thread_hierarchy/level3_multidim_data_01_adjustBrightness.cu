/*
 * Level 3: Multi-Dimensional Data Processing - Kernel 1: Grayscale Image Brightness
 *
 * This kernel applies brightness adjustment to a 2D grayscale image
 * with proper boundary handling and clamping.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Grayscale Image Brightness Adjustment
// Apply brightness adjustment to a 2D grayscale image
// ============================================================================
__global__ void adjustBrightness(unsigned char *image, int width, int height, float delta) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Add bounds check
    if (x < width && y < height) {
        int idx = y * width + x;
        float adjusted = image[idx] + delta;
        // Clamp to valid range [0, 255]
        image[idx] = (unsigned char)(adjusted < 0 ? 0 : (adjusted > 255 ? 255 : adjusted));
    }
}

// Utility functions
void initImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        img[i] = (i % 256);
    }
}

bool verifyBrightness(unsigned char *result, int width, int height, float delta) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float expected = (idx % 256) + delta;
            unsigned char clamped = (expected < 0) ? 0 : (expected > 255 ? 255 : (unsigned char)expected);
            if (result[idx] != clamped) return false;
        }
    }
    return true;
}

void printImageCorner(unsigned char *img, int width, int height, const char *label) {
    printf("%s (top-left 8x8 corner):\n", label);
    for (int row = 0; row < 8 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 8 && col < width; col++) {
            printf("%3d ", img[row * width + col]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 3: Brightness Adjustment ===\n\n");

    // Test brightness adjustment
    const int W = 256, H = 256;
    const float DELTA = 50.0f;
    unsigned char *h_image, *d_image;
    h_image = (unsigned char*)malloc(W * H);
    initImage(h_image, W, H);

    cudaMalloc(&d_image, W * H);
    cudaMemcpy(d_image, h_image, W * H, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    printf("Launching adjustBrightness kernel...\n");
    printf("  Image size: %d x %d pixels\n", W, H);
    printf("  Block size: %d x %d threads\n", block.x, block.y);
    printf("  Grid size: %d x %d blocks\n", grid.x, grid.y);
    printf("  Brightness delta: +%.1f\n\n", DELTA);

    printImageCorner(h_image, W, H, "Original image");

    adjustBrightness<<<grid, block>>>(d_image, W, H, DELTA);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return 1;
    }

    cudaMemcpy(h_image, d_image, W * H, cudaMemcpyDeviceToHost);

    printImageCorner(h_image, W, H, "Adjusted image");

    if (verifyBrightness(h_image, W, H, DELTA)) {
        printf("\n✓ Brightness adjustment PASSED\n");
    } else {
        printf("\n✗ Brightness adjustment FAILED - Check pixel indexing\n");
    }

    // Cleanup
    free(h_image);
    cudaFree(d_image);

    printf("\n=== Level 3.1 Complete ===\n");
    printf("Next: Try level3_multidim_data_02_swapRBChannels.cu for RGB channel swap\n");

    return 0;
}
