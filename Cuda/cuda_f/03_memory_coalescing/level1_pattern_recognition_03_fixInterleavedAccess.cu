/**
 * Fix Interleaved Access - Kernel 3 from level1_pattern_recognition.cu
 * 
 * This kernel demonstrates processing interleaved data (e.g., RGB pixels).
 * Each thread handles all channels for consecutive pixels.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 10000
#define CHANNELS 3

__global__ void fixInterleavedAccess(float *data, int n, int channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all channels for consecutive pixels
    // Each thread handles one or more complete pixels
    if (tid * channels < n) {
        int pixelIdx = tid;
        for (int c = 0; c < channels; c++) {
            int idx = pixelIdx * channels + c;
            if (idx < n) {
                data[idx] = data[idx] * 1.5f;
            }
        }
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

int main() {
    printf("=== Fix Interleaved Access (RGB pixels) ===\n\n");

    const int DATA_SIZE = N * CHANNELS;

    float *h_data = (float*)malloc(DATA_SIZE * sizeof(float));
    float *h_expected = (float*)malloc(DATA_SIZE * sizeof(float));
    initArray(h_data, DATA_SIZE);

    // Compute expected
    for (int i = 0; i < DATA_SIZE; i++) {
        h_expected[i] = h_data[i] * 1.5f;
    }

    float *d_data;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    fixInterleavedAccess<<<gridSize, blockSize>>>(d_data, DATA_SIZE, CHANNELS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyArray(h_data, h_expected, DATA_SIZE)) {
        printf("Interleaved access fix PASSED\n");
    } else {
        printf("Interleaved access fix FAILED\n");
    }

    // Cleanup
    free(h_data);
    free(h_expected);
    cudaFree(d_data);

    printf("\n=== Key Takeaways ===\n");
    printf("- Process all channels for consecutive pixels\n");
    printf("- Each thread handles complete pixels (all channels)\n");
    printf("- Avoid strided access across channels\n");

    return 0;
}
