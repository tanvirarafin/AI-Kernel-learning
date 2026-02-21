/**
 * Shared Reverse - Kernel 2 from level1_basic_shared.cu
 * 
 * This kernel demonstrates reversing data within a block using shared memory.
 * Load data, reverse within block, store result.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048
#define BLOCK_SIZE 256

__global__ void sharedReverse(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // Read from shared memory in reverse order within block
    if (idx < n) {
        int reverseIdx = BLOCK_SIZE - 1 - tid;
        // Handle boundary case where block is not full
        int blockEnd = min((blockIdx.x + 1) * BLOCK_SIZE, n);
        int blockStart = blockIdx.x * BLOCK_SIZE;
        int validReverseIdx = blockStart + (blockEnd - 1 - idx + blockStart);
        
        if (validReverseIdx < n && validReverseIdx >= blockStart) {
            output[idx] = sdata[BLOCK_SIZE - 1 - tid];
        } else {
            output[idx] = 0.0f;
        }
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

bool verifyReverse(float *result, float *input, int n, int blockSize) {
    for (int block = 0; block * blockSize < n; block++) {
        int blockStart = block * blockSize;
        int blockEnd = min(blockStart + blockSize, n);

        for (int i = blockStart; i < blockEnd; i++) {
            int reverseIdx = blockStart + (blockEnd - 1 - i);
            if (fabsf(result[i] - input[reverseIdx]) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Shared Memory Reverse ===\n\n");

    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));

    initArray(h_in, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaMemset(d_out, 0, N * sizeof(float));
    sharedReverse<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyReverse(h_out, h_in, N, blockSize)) {
        printf("Shared memory reverse PASSED\n");
    } else {
        printf("Shared memory reverse FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory enables block-level data rearrangement\n");
    printf("- Reverse indexing: BLOCK_SIZE - 1 - tid\n");
    printf("- Handle boundary cases for incomplete blocks\n");

    return 0;
}
