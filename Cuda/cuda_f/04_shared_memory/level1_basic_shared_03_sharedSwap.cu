/**
 * Shared Swap - Kernel 3 from level1_basic_shared.cu
 * 
 * This kernel demonstrates swapping adjacent elements using shared memory.
 * Even threads swap with next odd thread, odd threads swap with previous.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048
#define BLOCK_SIZE 256

__global__ void sharedSwap(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // Swap with adjacent element
    if (idx < n) {
        int swapIdx = (tid % 2 == 0) ? tid + 1 : tid - 1;

        // Handle boundary - last element in odd-sized block
        int blockEnd = min((blockIdx.x + 1) * BLOCK_SIZE, n);
        int localIdx = idx - blockIdx.x * BLOCK_SIZE;
        
        if (swapIdx < BLOCK_SIZE && (blockIdx.x * BLOCK_SIZE + swapIdx) < n) {
            output[idx] = sdata[swapIdx];
        } else {
            output[idx] = sdata[tid];  // Keep original if no swap partner
        }
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

bool verifySwap(float *result, float *input, int n, int blockSize) {
    for (int block = 0; block * blockSize < n; block++) {
        int blockStart = block * blockSize;
        int blockEnd = min(blockStart + blockSize, n);

        for (int i = blockStart; i < blockEnd; i++) {
            int localIdx = i - blockStart;
            int swapLocalIdx = (localIdx % 2 == 0) ? localIdx + 1 : localIdx - 1;
            int swapIdx = (swapLocalIdx >= 0 && swapLocalIdx < (blockEnd - blockStart))
                          ? blockStart + swapLocalIdx : i;
            if (fabsf(result[i] - input[swapIdx]) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Shared Memory Swap ===\n\n");

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
    sharedSwap<<<gridSize, blockSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifySwap(h_out, h_in, N, blockSize)) {
        printf("Shared memory swap PASSED\n");
    } else {
        printf("Shared memory swap FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- Even threads swap with tid+1, odd threads with tid-1\n");
    printf("- Handle boundary cases for last element\n");
    printf("- Shared memory enables efficient intra-block communication\n");

    return 0;
}
