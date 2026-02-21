/**
 * Dynamic Shared Padding - Kernel 5 from level3_bank_conflicts.cu
 * 
 * This kernel demonstrates dynamic shared memory with runtime padding.
 * Padding determined at kernel launch time.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define DYN_SIZE 10000
#define NUM_BANKS 32

__global__ void dynamicSharedPadding(float *input, float *output, int size, int padding) {
    // Dynamic shared memory - size determined at kernel launch
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load with padding consideration
    if (idx < size) {
        int paddedIdx = tid + (tid / (NUM_BANKS - 1)) * padding;
        sharedData[paddedIdx] = input[idx];
    }
    __syncthreads();

    // Process with padded access
    if (idx < size) {
        int accessIdx = tid + (tid / (NUM_BANKS - 1)) * padding;
        output[idx] = sharedData[accessIdx] * 2.0f;
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.1f;
    }
}

int main() {
    printf("=== Dynamic Shared Memory with Padding ===\n\n");

    float *h_dynIn = (float*)malloc(DYN_SIZE * sizeof(float));
    float *h_dynOut = (float*)malloc(DYN_SIZE * sizeof(float));
    initArray(h_dynIn, DYN_SIZE);

    float *d_dynIn, *d_dynOut;
    cudaMalloc(&d_dynIn, DYN_SIZE * sizeof(float));
    cudaMalloc(&d_dynOut, DYN_SIZE * sizeof(float));
    cudaMemcpy(d_dynIn, h_dynIn, DYN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int dynBlockSize = 256;
    int dynGridSize = (DYN_SIZE + dynBlockSize - 1) / dynBlockSize;
    int sharedMemSize = (DYN_SIZE + dynBlockSize / NUM_BANKS) * sizeof(float);

    printf("Launching dynamic shared memory kernel...\n");
    dynamicSharedPadding<<<dynGridSize, dynBlockSize, sharedMemSize>>>(
        d_dynIn, d_dynOut, DYN_SIZE, dynBlockSize / NUM_BANKS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dynOut, d_dynOut, DYN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    bool dynPass = true;
    for (int i = 0; i < DYN_SIZE && dynPass; i++) {
        if (fabsf(h_dynOut[i] - h_dynIn[i] * 2.0f) > 1e-5f) dynPass = false;
    }
    if (dynPass) {
        printf("Dynamic shared memory PASSED\n");
    } else {
        printf("Dynamic shared memory FAILED\n");
    }

    // Cleanup
    free(h_dynIn);
    free(h_dynOut);
    cudaFree(d_dynIn);
    cudaFree(d_dynOut);

    printf("\n=== Key Takeaways ===\n");
    printf("- extern __shared__ declares dynamic shared memory\n");
    printf("- Size specified at kernel launch (third parameter)\n");
    printf("- Padding can be computed at runtime\n");

    return 0;
}
