/**
 * Transpose 1D Block - Kernel 4 from level2_matrix_transpose.cu
 * 
 * This kernel demonstrates matrix transpose using 1D block configuration.
 * Can provide better occupancy on some GPUs.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 1024
#define HEIGHT 1024
#define TILE_SIZE 32

__global__ void transpose1DBlock(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // 1D to 2D mapping
    int x = blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE);
    int y = blockIdx.y * TILE_SIZE + (threadIdx.x / TILE_SIZE);

    // Load data into shared memory
    if (x < width && y < height) {
        tile[threadIdx.x / TILE_SIZE][threadIdx.x % TILE_SIZE] = input[y * width + x];
    }

    __syncthreads();

    // Calculate transposed coordinates
    int transposedX = blockIdx.y * TILE_SIZE + (threadIdx.x % TILE_SIZE);
    int transposedY = blockIdx.x * TILE_SIZE + (threadIdx.x / TILE_SIZE);

    // Write transposed data
    if (transposedX < height && transposedY < width) {
        output[transposedY * height + transposedX] = 
            tile[threadIdx.x % TILE_SIZE][threadIdx.x / TILE_SIZE];
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
    printf("=== 1D Block Matrix Transpose ===\n\n");

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

    dim3 block(TILE_SIZE * TILE_SIZE);  // 1D block
    dim3 grid((WIDTH + TILE_SIZE - 1) / TILE_SIZE,
              (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    cudaMemset(d_output, 0, N * sizeof(float));
    transpose1DBlock<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_output, h_expected, N)) {
        printf("1D block transpose correctness PASSED\n");
    } else {
        printf("1D block transpose correctness FAILED\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- 1D block configuration can improve occupancy\n");
    printf("- Map 1D thread index to 2D coordinates\n");
    printf("- Same shared memory tiling approach applies\n");

    return 0;
}
