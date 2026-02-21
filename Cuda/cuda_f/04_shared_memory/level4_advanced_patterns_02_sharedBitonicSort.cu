/**
 * Shared Bitonic Sort - Kernel 2 from level4_advanced_patterns.cu
 * 
 * This kernel demonstrates sorting elements within a block using bitonic sort.
 * Sort network implemented in shared memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void sharedBitonicSort(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    if (idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 999999.0f;  // Sentinel value
    }
    __syncthreads();

    // Bitonic sort stages
    for (int k = 2; k <= BLOCK_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;  // XOR to find compare partner
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    // Ascending: swap if sharedData[tid] > sharedData[ixj]
                    if (sharedData[tid] > sharedData[ixj]) {
                        float temp = sharedData[tid];
                        sharedData[tid] = sharedData[ixj];
                        sharedData[ixj] = temp;
                    }
                } else {
                    // Descending: swap if sharedData[tid] < sharedData[ixj]
                    if (sharedData[tid] < sharedData[ixj]) {
                        float temp = sharedData[tid];
                        sharedData[tid] = sharedData[ixj];
                        sharedData[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Store sorted data
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 1000);
    }
}

bool verifySort(float *result, int n) {
    for (int i = 1; i < n; i++) {
        if (result[i] < result[i-1]) return false;
    }
    return true;
}

int main() {
    printf("=== Shared Memory Bitonic Sort ===\n\n");

    const int N = 256;  // Must be power of 2 for bitonic sort
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));

    initArray(h_in, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    sharedBitonicSort<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifySort(h_out, N)) {
        printf("Bitonic sort PASSED\n");
    } else {
        printf("Bitonic sort FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- Bitonic sort is a sorting network\n");
    printf("- Works well in shared memory for block-sized data\n");
    printf("- Requires power-of-2 sized input\n");

    return 0;
}
