/**
 * Basic Shared Load - Kernel 1 from level1_basic_shared.cu
 * 
 * This kernel demonstrates basic shared memory load and store.
 * Load data to shared memory, synchronize, then process.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048
#define BLOCK_SIZE 256

__global__ void basicSharedLoad(float *input, float *output, int n) {
    // Shared memory array
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global to shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    // Synchronize threads before reading shared memory
    __syncthreads();

    // Process: multiply by 2 after synchronization
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f;
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

int main() {
    printf("=== Basic Shared Memory Load ===\n\n");

    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));

    initArray(h_in, N);

    // Compute expected
    for (int i = 0; i < N; i++) h_expected[i] = h_in[i] * 2.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaMemset(d_out, 0, N * sizeof(float));
    basicSharedLoad<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyArray(h_out, h_expected, N)) {
        printf("Basic shared memory PASSED\n");
    } else {
        printf("Basic shared memory FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    free(h_expected);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- __shared__ declares shared memory\n");
    printf("- __syncthreads() synchronizes all threads in block\n");
    printf("- Shared memory is fast, block-level storage\n");

    return 0;
}
