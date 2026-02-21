/**
 * Multi-Stage Shared - Kernel 4 from level1_basic_shared.cu
 * 
 * This kernel demonstrates multi-stage processing with shared memory.
 * Load -> Transform -> Sync -> Transform -> Sync -> Store
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 2048
#define BLOCK_SIZE 256

__global__ void multiStageShared(float *input, float *output, int n, float multiplier) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Stage 1: Load
    if (idx < n) {
        sdata[tid] = input[idx];
    }

    __syncthreads();

    // Stage 2: First transformation (multiply)
    if (idx < n) {
        sdata[tid] = sdata[tid] * multiplier;
    }

    __syncthreads();

    // Stage 3: Second transformation (add thread index)
    if (idx < n) {
        sdata[tid] = sdata[tid] + tid;
    }

    __syncthreads();

    // Stage 4: Store result
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

int main() {
    printf("=== Multi-Stage Shared Memory Processing ===\n\n");

    const float MULT = 3.0f;

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
    multiStageShared<<<gridSize, blockSize>>>(d_in, d_out, N, MULT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: expected = (input * MULT) + threadIdx.x
    bool pass = true;
    for (int block = 0; block < gridSize && pass; block++) {
        for (int t = 0; t < blockSize; t++) {
            int idx = block * blockSize + t;
            if (idx < N) {
                float expected = (h_in[idx] * MULT) + t;
                if (fabsf(h_out[idx] - expected) > 1e-5f) {
                    pass = false;
                    break;
                }
            }
        }
    }

    if (pass) {
        printf("Multi-stage processing PASSED\n");
    } else {
        printf("Multi-stage processing FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- Multiple __syncthreads() enable staged processing\n");
    printf("- Each stage can transform data cooperatively\n");
    printf("- All threads must reach __syncthreads() together\n");

    return 0;
}
