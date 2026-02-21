/**
 * Naive Transpose - Kernel 1 from level2_matrix_transpose.cu
 * 
 * This kernel demonstrates naive matrix transpose.
 * Read is coalesced but write is uncoalesced - used as baseline.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 1024
#define HEIGHT 1024

__global__ void naiveTranspose(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Read is coalesced (row-major), but WRITE is uncoalesced!
        // output[x][y] = input[y][x]
        output[x * height + y] = input[y * width + x];
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
    printf("=== Naive Matrix Transpose (Baseline) ===\n\n");

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

    dim3 block(32, 32);
    dim3 grid((WIDTH + 31) / 32, (HEIGHT + 31) / 32);

    cudaMemset(d_output, 0, N * sizeof(float));
    naiveTranspose<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyTranspose(h_output, h_expected, N)) {
        printf("Naive transpose correctness PASSED\n");
    } else {
        printf("Naive transpose correctness FAILED\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Naive transpose has coalesced reads but uncoalesced writes\n");
    printf("- Performance is limited by uncoalesced write pattern\n");
    printf("- Use shared memory tiling for better performance\n");

    return 0;
}
