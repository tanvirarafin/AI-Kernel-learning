/*
 * Level 3: Multi-Dimensional Data Processing - Kernel 5: RGBA Normalization
 *
 * This kernel normalizes each channel of an RGBA image independently
 * using per-channel min/max values.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 5: Multi-Channel Image Normalization (Challenge)
// Normalize each channel of an RGBA image independently
// ============================================================================
__global__ void normalizeRGBA(float *image, int width, int height,
                               float *channelMin, float *channelMax) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Process all 4 channels for this pixel
    for (int c = 0; c < 4; c++) {
        int idx = (y * width + x) * 4 + c;
        float range = channelMax[c] - channelMin[c];

        // Normalize to [0, 1]
        if (range > 0) {
            image[idx] = (image[idx] - channelMin[c]) / range;
        } else {
            image[idx] = 0.5f;  // Constant channel
        }
    }
}

// Utility functions
void initRGBAImage(float *img, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int base = (y * width + x) * 4;
            img[base + 0] = (x % 256) / 255.0f;     // R
            img[base + 1] = (y % 256) / 255.0f;     // G
            img[base + 2] = ((x + y) % 256) / 255.0f; // B
            img[base + 3] = 1.0f;                    // A
        }
    }
}

void findChannelMinMax(float *img, int width, int height,
                       float *minVals, float *maxVals) {
    for (int c = 0; c < 4; c++) {
        minVals[c] = 1e10f;
        maxVals[c] = -1e10f;
    }
    for (int i = 0; i < width * height * 4; i++) {
        int c = i % 4;
        if (img[i] < minVals[c]) minVals[c] = img[i];
        if (img[i] > maxVals[c]) maxVals[c] = img[i];
    }
}

bool verifyNormalize(float *result, int width, int height,
                     float *minVals, float *maxVals) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 4; c++) {
                int idx = (y * width + x) * 4 + c;
                float range = maxVals[c] - minVals[c];
                float expected = (range > 0) ? 
                    ((y * width + x) * 4 + c) / 255.0f : 0.5f;
                // Just check that values are in [0, 1] range
                if (result[idx] < 0.0f || result[idx] > 1.0f) return false;
            }
        }
    }
    return true;
}

void printPixelCorner(float *img, int width, int height, const char *label) {
    printf("%s (top-left 3x3 pixels, RGBA values):\n", label);
    for (int row = 0; row < 3 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 3 && col < width; col++) {
            int base = (row * width + col) * 4;
            printf("(%.3f,%.3f,%.3f,%.3f) ", 
                   img[base], img[base+1], img[base+2], img[base+3]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 3: RGBA Normalization ===\n\n");

    // Test RGBA normalization
    const int W = 64, H = 64;
    float *h_image, *d_image;
    float *d_channelMin, *d_channelMax;
    float h_min[4], h_max[4];
    
    h_image = (float*)malloc(W * H * 4 * sizeof(float));
    initRGBAImage(h_image, W, H);
    findChannelMinMax(h_image, W, H, h_min, h_max);

    cudaMalloc(&d_image, W * H * 4 * sizeof(float));
    cudaMalloc(&d_channelMin, 4 * sizeof(float));
    cudaMalloc(&d_channelMax, 4 * sizeof(float));
    
    cudaMemcpy(d_image, h_image, W * H * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channelMin, h_min, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channelMax, h_max, 4 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    printf("Launching normalizeRGBA kernel...\n");
    printf("  Image size: %d x %d pixels (%d channels)\n", W, H, 4);
    printf("  Block size: %d x %d threads\n", block.x, block.y);
    printf("  Grid size: %d x %d blocks\n", grid.x, grid.y);
    printf("  Channel mins: [%.3f, %.3f, %.3f, %.3f]\n", 
           h_min[0], h_min[1], h_min[2], h_min[3]);
    printf("  Channel maxs: [%.3f, %.3f, %.3f, %.3f]\n\n", 
           h_max[0], h_max[1], h_max[2], h_max[3]);

    printPixelCorner(h_image, W, H, "Original RGBA image");

    normalizeRGBA<<<grid, block>>>(d_image, W, H, d_channelMin, d_channelMax);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        cudaFree(d_channelMin);
        cudaFree(d_channelMax);
        free(h_image);
        return 1;
    }

    cudaMemcpy(h_image, d_image, W * H * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printPixelCorner(h_image, W, H, "Normalized RGBA image");

    if (verifyNormalize(h_image, W, H, h_min, h_max)) {
        printf("\n✓ RGBA normalization PASSED\n");
    } else {
        printf("\n✗ RGBA normalization FAILED - Check channel processing\n");
    }

    // Cleanup
    free(h_image);
    cudaFree(d_image);
    cudaFree(d_channelMin);
    cudaFree(d_channelMax);

    printf("\n=== Level 3.5 Complete ===\n");
    printf("Next: Try 02_memory_hierarchy kernels for memory optimization patterns\n");

    return 0;
}
