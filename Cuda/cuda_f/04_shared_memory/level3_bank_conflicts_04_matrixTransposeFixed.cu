/**
 * Matrix Transpose Fixed - Kernel 4 from level3_bank_conflicts.cu
 * 
 * This kernel demonstrates conflict-free matrix transpose using padding.
 * Padding shared memory eliminates bank conflicts during transpose.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 512
#define TILE_WIDTH 32
#define PADDED_TILE_WIDTH (TILE_WIDTH + 1)

__global__ void matrixTransposeFixed(float *input, float *output, int width) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float sharedData[TILE_WIDTH][PADDED_TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowIn = by * TILE_WIDTH + ty;
    int colIn = bx * TILE_WIDTH + tx;
    int idxIn = rowIn * width + colIn;

    // Load with padded column index
    sharedData[ty][tx] = input[idxIn];
    __syncthreads();

    // Transpose: read from [tx][ty], write to transposed position
    int rowOut = by * TILE_WIDTH + tx;
    int colOut = bx * TILE_WIDTH + ty;
    int idxOut = rowOut * width + colOut;

    // Store with padded row index (was column before transpose)
    output[idxOut] = sharedData[tx][ty];
}

void initMatrix(float *mat, int width, float seed) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)(i % 100) / 100.0f + seed;
    }
}

bool verifyTranspose(float *result, float *input, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = input[col * width + row];
            if (fabsf(result[row * width + col] - expected) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Matrix Transpose Fixed (Bank Conflict Free) ===\n\n");

    const int WIDTH_VAL = 512;
    const int N = WIDTH_VAL * WIDTH_VAL;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    initMatrix(h_A, WIDTH_VAL, 1.0f);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(WIDTH_VAL / TILE_WIDTH, WIDTH_VAL / TILE_WIDTH);

    printf("Running transpose WITH padding (conflict-free)...\n");
    matrixTransposeFixed<<<grid, block>>>(d_A, d_B, WIDTH_VAL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_B, h_A, WIDTH_VAL)) {
        printf("Matrix transpose PASSED\n");
    } else {
        printf("Matrix transpose FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    printf("\n=== Key Takeaways ===\n");
    printf("- Padding shared memory eliminates bank conflicts\n");
    printf("- Use tile[TILE_WIDTH][TILE_WIDTH + 1] for transpose\n");
    printf("- Transpose without padding has severe bank conflicts\n");

    return 0;
}
