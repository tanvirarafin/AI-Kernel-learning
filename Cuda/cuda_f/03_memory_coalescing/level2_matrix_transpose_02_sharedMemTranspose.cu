/**
 * Shared Memory Transpose - Kernel 2 from level2_matrix_transpose.cu
 * 
 * This kernel demonstrates matrix transpose using shared memory.
 * Achieves coalesced reads AND writes through tile-based approach.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 1024
#define HEIGHT 1024
#define TILE_SIZE 32

__global__ void sharedMemTranspose(float *input, float *output, int width, int height) {
    // Shared memory tile
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data into shared memory (coalesced read)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // Calculate transposed coordinates for writing
    int transposedX = blockIdx.y * blockDim.x + threadIdx.x;
    int transposedY = blockIdx.x * blockDim.y + threadIdx.y;

    // Read from shared memory and write to output (coalesced write)
    if (transposedX < height && transposedY < width) {
        output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

void initMatrix(float *mat, int n) {
    for (int i = 0; i < n; i++) {
        mat[i] = i * 0.5f;
    }
}

bool verifyTranspose(float *result, float *expected, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

void transposeCPU(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

int main() {
    printf("=== Shared Memory Matrix Transpose ===\n\n");

    const int N = WIDTH * HEIGHT;

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));

    initMatrix(h_input, N);
    transposeCPU(h_input, h_expected, WIDTH, HEIGHT);

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE,
              (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    cudaMemset(d_output, 0, N * sizeof(float));
    sharedMemTranspose<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_output, h_expected, N)) {
        printf("Shared memory transpose correctness PASSED\n");
    } else {
        printf("Shared memory transpose correctness FAILED\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory enables coalesced reads AND writes\n");
    printf("- Load tile cooperatively, transpose in shared memory, write coalesced\n");
    printf("- Requires __syncthreads() between load and store phases\n");

    return 0;
}
