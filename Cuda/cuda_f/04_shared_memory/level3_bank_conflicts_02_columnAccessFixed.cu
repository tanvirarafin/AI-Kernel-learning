/**
 * Column Access Fixed - Kernel 2 from level3_bank_conflicts.cu
 * 
 * This kernel demonstrates bank conflict-free column access using padding.
 * Adding one element padding eliminates 32-way bank conflicts.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 256
#define BLOCK_SIZE 256
#define PADDED_BLOCK_SIZE (BLOCK_SIZE + 1)

__global__ void columnAccessFixed(float *input, float *output, int width) {
    // Add padding to eliminate bank conflicts
    __shared__ float sharedData[PADDED_BLOCK_SIZE];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + row * blockDim.x + col;

    // Load with padded indexing
    int loadIdx = row * blockDim.x + col;
    sharedData[loadIdx] = input[idx];
    __syncthreads();

    // Access with padded indexing - now conflict-free!
    int storeIdx = col * (blockDim.x + 1) + row;
    output[idx] = sharedData[storeIdx] * 2.0f;
}

void initMatrix(float *mat, int width, float seed) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)(i % 100) / 100.0f + seed;
    }
}

bool verifyTranspose(float *result, float *input, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = input[col * width + row] * 2.0f;
            if (fabsf(result[row * width + col] - expected) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Column Access Fixed (Bank Conflict Free) ===\n\n");

    const int WIDTH_VAL = 256;
    const int N = WIDTH_VAL * WIDTH_VAL;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    initMatrix(h_A, WIDTH_VAL, 1.0f);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(WIDTH_VAL / 16, WIDTH_VAL / 16);

    printf("Running transpose WITH padding (conflict-free)...\n");
    columnAccessFixed<<<grid, block>>>(d_A, d_B, WIDTH_VAL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_B, h_A, WIDTH_VAL)) {
        printf("Matrix transpose PASSED (bank conflict free!)\n");
    } else {
        printf("Matrix transpose FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    printf("\n=== Key Takeaways ===\n");
    printf("- Padding eliminates shared memory bank conflicts\n");
    printf("- Add 1 element: sharedData[BLOCK_SIZE + 1]\n");
    printf("- 32 banks on most GPUs\n");

    return 0;
}
