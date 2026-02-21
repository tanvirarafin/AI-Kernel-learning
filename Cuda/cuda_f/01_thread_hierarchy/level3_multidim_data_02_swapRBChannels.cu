/*
 * Level 3: Multi-Dimensional Data Processing - Kernel 2: RGB Channel Swap
 *
 * This kernel swaps the red and blue channels in an interleaved RGB image
 * (RGB -> BGR conversion).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 2: RGB Image Channel Swap (RGB -> BGR)
// Swap red and blue channels in an interleaved RGB image
// ============================================================================
__global__ void swapRBChannels(unsigned char *image, int width, int height) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Add bounds check
    if (x < width && y < height) {
        // Calculate base index for this pixel (3 channels per pixel)
        int idx = (y * width + x) * 3;

        // Swap R (channel 0) and B (channel 2)
        unsigned char temp = image[idx];
        image[idx] = image[idx + 2];
        image[idx + 2] = temp;
    }
}

// Utility functions
void initRGBImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
        img[i] = (i % 256);
    }
}

bool verifySwapRB(unsigned char *result, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int base = (y * width + x) * 3;
            // Original: R=base%256, G=(base+1)%256, B=(base+2)%256
            // After swap: B should be at base, R at base+2
            unsigned char origR = (base) % 256;
            unsigned char origB = (base + 2) % 256;
            if (result[base] != origB || result[base + 2] != origR) return false;
        }
    }
    return true;
}

void printPixelCorner(unsigned char *img, int width, int height, const char *label) {
    printf("%s (top-left 4x4 pixels, RGB values):\n", label);
    for (int row = 0; row < 4 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 4 && col < width; col++) {
            int base = (row * width + col) * 3;
            printf("R%3dG%3dB%3d ", img[base], img[base+1], img[base+2]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 3: RGB Channel Swap ===\n\n");

    // Test RGB channel swap
    const int W = 64, H = 64;
    unsigned char *h_rgb, *d_rgb;
    h_rgb = (unsigned char*)malloc(W * H * 3);
    initRGBImage(h_rgb, W, H);

    cudaMalloc(&d_rgb, W * H * 3);
    cudaMemcpy(d_rgb, h_rgb, W * H * 3, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    printf("Launching swapRBChannels kernel...\n");
    printf("  Image size: %d x %d pixels (%d bytes)\n", W, H, W * H * 3);
    printf("  Block size: %d x %d threads\n", block.x, block.y);
    printf("  Grid size: %d x %d blocks\n\n", grid.x, grid.y);

    printPixelCorner(h_rgb, W, H, "Original RGB image");

    swapRBChannels<<<grid, block>>>(d_rgb, W, H);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_rgb);
        free(h_rgb);
        return 1;
    }

    cudaMemcpy(h_rgb, d_rgb, W * H * 3, cudaMemcpyDeviceToHost);

    printPixelCorner(h_rgb, W, H, "After R<->B swap (BGR)");

    if (verifySwapRB(h_rgb, W, H)) {
        printf("\n✓ RGB channel swap PASSED\n");
    } else {
        printf("\n✗ RGB channel swap FAILED - Check channel indexing\n");
    }

    // Cleanup
    free(h_rgb);
    cudaFree(d_rgb);

    printf("\n=== Level 3.2 Complete ===\n");
    printf("Next: Try level3_multidim_data_03_volumeBlur.cu for 3D volume processing\n");

    return 0;
}
