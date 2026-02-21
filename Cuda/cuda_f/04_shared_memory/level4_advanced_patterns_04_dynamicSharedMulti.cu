/**
 * Dynamic Shared Multi - Kernel 4 from level4_advanced_patterns.cu
 * 
 * This kernel demonstrates dynamic shared memory for multiple algorithms.
 * Algorithm selection at runtime via parameter.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void dynamicSharedMulti(float *input, float *output, int n,
                                   int algorithm, int param) {
    // Dynamic shared memory - size determined at launch
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    switch (algorithm) {
        case 0:  // Prefix sum (scan)
            // Load
            if (idx < n) {
                sharedData[tid] = input[idx];
            } else {
                sharedData[tid] = 0.0f;
            }
            __syncthreads();

            // Upsweep phase
            for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
                int index = (tid + 1) * 2 * stride - 1;
                if (index < BLOCK_SIZE) {
                    sharedData[index] += sharedData[index - stride];
                }
                __syncthreads();
            }

            // Downsweep phase
            for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
                int index = (tid + 1) * 2 * stride - 1;
                if (index < BLOCK_SIZE) {
                    float t = sharedData[index - stride];
                    sharedData[index - stride] = sharedData[index];
                    sharedData[index] += t;
                }
                __syncthreads();
            }

            if (idx < n) {
                output[idx] = sharedData[tid];
            }
            break;

        case 1:  // Reduction (sum)
            // Load with grid-stride
            float sum = 0.0f;
            for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
                sum += input[i];
            }
            sharedData[tid] = sum;
            __syncthreads();

            // Tree reduction
            for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sharedData[tid] += sharedData[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[blockIdx.x] = sharedData[0];
            }
            break;

        case 2:  // Broadcast (first element to all)
            if (tid == 0 && idx < n) {
                sharedData[0] = input[idx];
            }
            __syncthreads();

            if (idx < n) {
                output[idx] = sharedData[0];
            }
            break;
    }
}

void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) {
        arr[i] = val;
    }
}

int main() {
    printf("=== Dynamic Shared Memory Multi-Algorithm ===\n\n");

    const int N = 10000;
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));

    initArray(h_in, N, 1.0f);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = 64;
    int sharedMemSize = blockSize * sizeof(float);

    // Test reduction (algorithm 1)
    printf("Testing reduction algorithm...\n");
    dynamicSharedMulti<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_out, N, 1, 0);
    cudaDeviceSynchronize();

    // Sum partial results
    float *h_temp = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_temp, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float result = 0.0f;
    for (int i = 0; i < gridSize; i++) result += h_temp[i];

    printf("  Reduction result: %.0f (expected: %d)\n", result, N);
    if ((int)result == N) {
        printf("  PASSED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    free(h_temp);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- extern __shared__ for dynamic sizing\n");
    printf("- Algorithm selection via parameter\n");
    printf("- Same shared memory for different patterns\n");

    return 0;
}
