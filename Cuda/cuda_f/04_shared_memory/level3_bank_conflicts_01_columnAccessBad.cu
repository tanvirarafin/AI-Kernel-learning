/**
 * Column Access Bad - Kernel 1 from level3_bank_conflicts.cu
 * 
 * This kernel demonstrates bank conflicts with column access pattern.
 * Creates 32-way bank conflicts - used to show the problem.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 256
#define BLOCK_SIZE 256
#define NUM_BANKS 32

__global__ void columnAccessBad(float *input, float *output, int width) {
    // No padding - causes 32-way bank conflicts!
    __shared__ float sharedData[BLOCK_SIZE];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + row * blockDim.x + col;

    // Load data - each thread loads one element
    sharedData[row * blockDim.x + col] = input[idx];
    __syncthreads();

    // Column access pattern causes bank conflicts!
    // Each thread reads from a different row but same column
    output[idx] = sharedData[col * blockDim.x + row] * 2.0f;
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
    printf("=== Column Access (With Bank Conflicts) ===\n\n");

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

    printf("Running transpose WITHOUT padding (has bank conflicts)...\n");
    columnAccessBad<<<grid, block>>>(d_A, d_B, WIDTH_VAL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_B, h_A, WIDTH_VAL)) {
        printf("Matrix transpose PASSED (but has bank conflicts!)\n");
    } else {
        printf("Matrix transpose FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    printf("\n=== Key Takeaways ===\n");
    printf("- Column access causes 32-way bank conflicts\n");
    printf("- All threads in a warp access the same bank\n");
    printf("- Add padding to eliminate conflicts\n");

    return 0;
}
