/**
 * Convolution 1D - Kernel 3 from level4_advanced_patterns.cu
 * 
 * This kernel demonstrates 1D convolution using shared memory.
 * Halo regions loaded for boundary handling.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define KERNEL_SIZE 5

__global__ void convolution1D(float *input, float *output, float *kernel,
                              int n, int kernelSize) {
    const int haloSize = (kernelSize - 1) / 2;
    // Shared memory with halo regions
    __shared__ float sharedData[BLOCK_SIZE + kernelSize - 1];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data with halo regions for boundary handling
    if (idx < n) {
        sharedData[tid + haloSize] = input[idx];
    }

    // Load left halo (boundary threads)
    if (tid < haloSize) {
        int leftIdx = blockIdx.x * blockDim.x - haloSize + tid;
        sharedData[tid] = (leftIdx >= 0) ? input[leftIdx] : 0.0f;
    }

    // Load right halo (boundary threads)
    if (tid < haloSize) {
        int rightIdx = (blockIdx.x + 1) * blockDim.x + tid;
        sharedData[BLOCK_SIZE + tid] = (rightIdx < n) ? input[rightIdx] : 0.0f;
    }

    __syncthreads();

    // Apply convolution
    if (idx < n) {
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            sum += kernel[k] * sharedData[tid + k];
        }
        output[idx] = sum;
    }
}

void initRandom(float *arr, int n, float maxVal) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % (int)(maxVal * 100)) / 100.0f;
    }
}

void initKernel(float *kernel, int size) {
    // Simple averaging kernel
    for (int i = 0; i < size; i++) {
        kernel[i] = 1.0f / size;
    }
}

bool verifyConvolution(float *result, float *input, float *kernel,
                       int n, int kernelSize) {
    int haloSize = (kernelSize - 1) / 2;
    for (int i = 0; i < n; i++) {
        float expected = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            int idx = i + k - haloSize;
            float val = (idx >= 0 && idx < n) ? input[idx] : 0.0f;
            expected += kernel[k] * val;
        }
        if (fabsf(result[i] - expected) > 1e-4f) return false;
    }
    return true;
}

int main() {
    printf("=== 1D Convolution with Shared Memory ===\n\n");

    const int CONV_N = 10000;
    float *h_convIn = (float*)malloc(CONV_N * sizeof(float));
    float *h_convOut = (float*)malloc(CONV_N * sizeof(float));
    float *h_convKernel = (float*)malloc(KERNEL_SIZE * sizeof(float));

    initRandom(h_convIn, CONV_N, 10.0f);
    initKernel(h_convKernel, KERNEL_SIZE);

    float *d_convIn, *d_convOut, *d_convKernel;
    cudaMalloc(&d_convIn, CONV_N * sizeof(float));
    cudaMalloc(&d_convOut, CONV_N * sizeof(float));
    cudaMalloc(&d_convKernel, KERNEL_SIZE * sizeof(float));
    cudaMemcpy(d_convIn, h_convIn, CONV_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convKernel, h_convKernel, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int convBlocks = (CONV_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Launching convolution kernel...\n");
    convolution1D<<<convBlocks, BLOCK_SIZE>>>(d_convIn, d_convOut, d_convKernel,
                                              CONV_N, KERNEL_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_convOut, d_convOut, CONV_N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyConvolution(h_convOut, h_convIn, h_convKernel, CONV_N, KERNEL_SIZE)) {
        printf("Convolution PASSED\n");
    } else {
        printf("Convolution FAILED\n");
    }

    // Cleanup
    free(h_convIn);
    free(h_convOut);
    free(h_convKernel);
    cudaFree(d_convIn);
    cudaFree(d_convOut);
    cudaFree(d_convKernel);

    printf("\n=== Key Takeaways ===\n");
    printf("- Halo regions handle convolution boundaries\n");
    printf("- Shared memory enables efficient stencil access\n");
    printf("- Each thread loads one element, edge threads load halo\n");

    return 0;
}
