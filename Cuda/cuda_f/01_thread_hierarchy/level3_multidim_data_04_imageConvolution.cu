/*
 * Level 3: Multi-Dimensional Data Processing - Kernel 4: Image Convolution
 *
 * This kernel applies a convolution filter to an image with proper
 * boundary/halo handling.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 4: Image Convolution with Halo/Boundary Handling
// Apply convolution while properly handling image boundaries
// ============================================================================
__global__ void imageConvolution(float *input, float *output, int width, int height,
                                  float *kernel, int kernelSize) {
    // Calculate output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Add bounds check
    if (x < width && y < height) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;

        // Apply convolution kernel
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                // Handle boundary by clamping coordinates
                int px = x + kx - halfKernel;
                int py = y + ky - halfKernel;
                
                // Clamp to image boundaries
                if (px < 0) px = 0;
                if (px >= width) px = width - 1;
                if (py < 0) py = 0;
                if (py >= height) py = height - 1;
                
                sum += input[py * width + px] * kernel[ky * kernelSize + kx];
            }
        }

        output[y * width + x] = sum;
    }
}

// Utility functions
void initImage(float *img, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = (x + y) * 0.1f;
        }
    }
}

void initGaussianKernel(float *kernel, int size) {
    // Simple 3x3 Gaussian-like kernel
    float kernel3x3[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
    for (int i = 0; i < size * size && i < 9; i++) {
        kernel[i] = kernel3x3[i];
    }
}

bool verifyConvolution(float *result, float *input, float *kernel,
                       int width, int height, int kernelSize) {
    int halfKernel = kernelSize / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float expected = 0.0f;
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int px = x + kx - halfKernel;
                    int py = y + ky - halfKernel;
                    if (px < 0) px = 0;
                    if (px >= width) px = width - 1;
                    if (py < 0) py = 0;
                    if (py >= height) py = height - 1;
                    expected += input[py * width + px] * kernel[ky * kernelSize + kx];
                }
            }
            if (fabsf(result[y * width + x] - expected) > 1e-4f) return false;
        }
    }
    return true;
}

void printImageCorner(float *img, int width, int height, const char *label) {
    printf("%s (top-left 5x5):\n", label);
    for (int row = 0; row < 5 && row < height; row++) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 5 && col < width; col++) {
            printf("%7.4f ", img[row * width + col]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 3: Image Convolution ===\n\n");

    // Test image convolution
    const int W = 64, H = 64;
    const int KERNEL_SIZE = 3;
    float *h_input, *h_output, *h_kernel;
    float *d_input, *d_output, *d_kernel;
    
    h_input = (float*)malloc(W * H * sizeof(float));
    h_output = (float*)malloc(W * H * sizeof(float));
    h_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    
    initImage(h_input, W, H);
    initGaussianKernel(h_kernel, KERNEL_SIZE);

    cudaMalloc(&d_input, W * H * sizeof(float));
    cudaMalloc(&d_output, W * H * sizeof(float));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    
    cudaMemcpy(d_input, h_input, W * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    printf("Launching imageConvolution kernel...\n");
    printf("  Image size: %d x %d pixels\n", W, H);
    printf("  Kernel size: %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("  Block size: %d x %d threads\n", block.x, block.y);
    printf("  Grid size: %d x %d blocks\n\n", grid.x, grid.y);

    printImageCorner(h_input, W, H, "Original image");

    imageConvolution<<<grid, block>>>(d_input, d_output, W, H, d_kernel, KERNEL_SIZE);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        free(h_input);
        free(h_output);
        free(h_kernel);
        return 1;
    }

    cudaMemcpy(h_output, d_output, W * H * sizeof(float), cudaMemcpyDeviceToHost);

    printImageCorner(h_output, W, H, "Convolved image");

    if (verifyConvolution(h_output, h_input, h_kernel, W, H, KERNEL_SIZE)) {
        printf("\n✓ Image convolution PASSED\n");
    } else {
        printf("\n✗ Image convolution FAILED - Check boundary handling\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    printf("\n=== Level 3.4 Complete ===\n");
    printf("Next: Try level3_multidim_data_05_normalizeRGBA.cu for multi-channel normalization\n");

    return 0;
}
