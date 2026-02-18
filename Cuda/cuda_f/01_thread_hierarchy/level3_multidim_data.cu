/*
 * Level 3: Multi-Dimensional Data Processing
 * 
 * EXERCISE: Process 2D images and 3D volumes with proper boundary
 * handling and efficient thread-to-data mapping.
 * 
 * SKILLS PRACTICED:
 * - Image processing patterns
 * - Volume data handling
 * - Boundary conditions (halo regions)
 * - Channel-aware processing
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Grayscale Image Brightness Adjustment
// Apply brightness adjustment to a 2D grayscale image
// ============================================================================
__global__ void adjustBrightness(unsigned char *image, int width, int height, float delta) {
    // TODO: Calculate pixel coordinates
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check
    if (/* YOUR CODE HERE */) {
        int idx = y * width + x;
        float adjusted = image[idx] + delta;
        // Clamp to valid range [0, 255]
        image[idx] = (unsigned char)(adjusted < 0 ? 0 : (adjusted > 255 ? 255 : adjusted));
    }
}

// ============================================================================
// KERNEL 2: RGB Image Channel Swap (RGB -> BGR)
// Swap red and blue channels in an interleaved RGB image
// ============================================================================
__global__ void swapRBChannels(unsigned char *image, int width, int height) {
    // TODO: Calculate pixel coordinates
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check
    if (/* YOUR CODE HERE */) {
        // TODO: Calculate base index for this pixel (3 channels per pixel)
        int idx = /* YOUR CODE HERE */ 0;
        
        // Swap R (channel 0) and B (channel 2)
        unsigned char temp = image[idx];
        image[idx] = image[idx + 2];
        image[idx + 2] = temp;
    }
}

// ============================================================================
// KERNEL 3: 3D Volume Gaussian Blur (Simplified)
// Apply a simple 3x3x3 blur kernel to volume data
// ============================================================================
__global__ void volumeBlur(float *input, float *output, int width, int height, int depth) {
    // TODO: Calculate voxel coordinates
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    int z = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check (skip boundary voxels for simplicity)
    if (/* YOUR CODE HERE */) {
        float sum = 0.0f;
        int count = 0;
        
        // TODO: Loop over 3x3x3 neighborhood and accumulate
        // for (int dz = -1; dz <= 1; dz++) {
        //     for (int dy = -1; dy <= 1; dy++) {
        //         for (int dx = -1; dx <= 1; dx++) {
        //             int nx = x + dx;
        //             int ny = y + dy;
        //             int nz = z + dz;
        //             int nidx = /* YOUR 3D INDEX HERE */;
        //             sum += input[nidx];
        //             count++;
        //         }
        //     }
        // }
        
        output[z * width * height + y * width + x] = sum / count;
    }
}

// ============================================================================
// KERNEL 4: Image Convolution with Halo/Boundary Handling
// Apply convolution while properly handling image boundaries
// ============================================================================
__global__ void imageConvolution(float *input, float *output, int width, int height, 
                                  float *kernel, int kernelSize) {
    // TODO: Calculate output pixel coordinates
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    
    // TODO: Add bounds check
    if (/* YOUR CODE HERE */) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        
        // TODO: Apply convolution kernel
        // for (int ky = 0; ky < kernelSize; ky++) {
        //     for (int kx = 0; kx < kernelSize; kx++) {
        //         int px = /* YOUR CODE HERE: handle boundary */;
        //         int py = /* YOUR CODE HERE: handle boundary */;
        //         sum += input[py * width + px] * kernel[ky * kernelSize + kx];
        //     }
        // }
        
        output[y * width + x] = sum;
    }
}

// ============================================================================
// KERNEL 5: Multi-Channel Image Normalization (Challenge)
// Normalize each channel of an RGBA image independently
// ============================================================================
__global__ void normalizeRGBA(float *image, int width, int height, 
                               float *channelMin, float *channelMax) {
    // TODO: Calculate pixel and channel indices
    int x = /* YOUR CODE HERE */ 0;
    int y = /* YOUR CODE HERE */ 0;
    int c = /* YOUR CODE HERE: use threadIdx.z or other method */ 0;
    
    // TODO: Add bounds check for both spatial and channel dimensions
    if (/* YOUR CODE HERE */) {
        int idx = (y * width + x) * 4 + c;
        float range = channelMax[c] - channelMin[c];
        
        // Normalize to [0, 1]
        image[idx] = (image[idx] - channelMin[c]) / (range > 0 ? range : 1.0f);
    }
}

// Utility functions
void initImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        img[i] = (i % 256);
    }
}

void initRGBImage(unsigned char *img, int width, int height) {
    for (int i = 0; i < width * height * 3; i++) {
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

int main() {
    printf("=== Thread Hierarchy Level 3: Multi-Dimensional Data ===\n\n");
    
    // Test 1: Brightness adjustment
    printf("Testing brightness adjustment...\n");
    const int W = 256, H = 256;
    unsigned char *h_image, *d_image;
    h_image = (unsigned char*)malloc(W * H);
    initImage(h_image, W, H);
    
    cudaMalloc(&d_image, W * H);
    cudaMemcpy(d_image, h_image, W * H, cudaMemcpyHostToDevice);
    
    dim3 block(W / 16, H / 16);
    adjustBrightness<<<16, 16>>>(d_image, W, H, 50.0f);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_image, d_image, W * H, cudaMemcpyDeviceToHost);
    
    if (verifyBrightness(h_image, W, H, 50.0f)) {
        printf("✓ Brightness adjustment PASSED\n");
    } else {
        printf("✗ Brightness adjustment FAILED - Check pixel indexing\n");
    }
    
    // Test 2: RGB channel swap
    printf("\nTesting RGB channel swap...\n");
    unsigned char *h_rgb, *d_rgb;
    h_rgb = (unsigned char*)malloc(W * H * 3);
    initRGBImage(h_rgb, W, H);
    
    cudaMalloc(&d_rgb, W * H * 3);
    cudaMemcpy(d_rgb, h_rgb, W * H * 3, cudaMemcpyHostToDevice);
    
    dim3 blockRGB(16, 16);
    dim3 gridRGB((W + 15) / 16, (H + 15) / 16);
    swapRBChannels<<<gridRGB, blockRGB>>>(d_rgb, W, H);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_rgb, d_rgb, W * H * 3, cudaMemcpyDeviceToHost);
    
    if (verifySwapRB(h_rgb, W, H)) {
        printf("✓ RGB channel swap PASSED\n");
    } else {
        printf("✗ RGB channel swap FAILED - Check channel indexing\n");
    }
    
    // Cleanup
    free(h_image);
    free(h_rgb);
    cudaFree(d_image);
    cudaFree(d_rgb);
    
    printf("\n=== Level 3 Complete ===\n");
    printf("Next: Try level4_warp_aware.cu for warp-level optimization\n");
    
    return 0;
}
