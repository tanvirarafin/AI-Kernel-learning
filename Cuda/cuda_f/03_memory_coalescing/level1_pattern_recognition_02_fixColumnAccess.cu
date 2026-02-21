/**
 * Fix Column Access - Kernel 2 from level1_pattern_recognition.cu
 * 
 * This kernel demonstrates correct row-major matrix access.
 * Original had column-major access pattern which is inefficient for row-major data.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 256
#define HEIGHT 256

__global__ void fixColumnAccess(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        // Row-major: index = row * width + col
        // Consecutive threads in x-dimension access consecutive memory
        int idx = row * width + col;
        matrix[idx] = matrix[idx] * 2.0f;
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

bool verifyMatrix(float *result, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            float expected = (idx * 0.5f) * 2.0f;
            if (fabsf(result[idx] - expected) > 1e-5f) return false;
        }
    }
    return true;
}

int main() {
    printf("=== Fix Column Access (Row-Major) ===\n\n");

    const int N_MAT = WIDTH * HEIGHT;

    float *h_mat = (float*)malloc(N_MAT * sizeof(float));
    initArray(h_mat, N_MAT);

    float *d_mat;
    cudaMalloc(&d_mat, N_MAT * sizeof(float));
    cudaMemcpy(d_mat, h_mat, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    fixColumnAccess<<<grid, block>>>(d_mat, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_mat, d_mat, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyMatrix(h_mat, WIDTH, HEIGHT)) {
        printf("Column access fix PASSED\n");
    } else {
        printf("Column access fix FAILED\n");
    }

    // Cleanup
    free(h_mat);
    cudaFree(d_mat);

    printf("\n=== Key Takeaways ===\n");
    printf("- Row-major storage: index = row * width + col\n");
    printf("- Consecutive threads in x-dimension access consecutive memory\n");
    printf("- Column-major access causes uncoalesced transactions\n");

    return 0;
}
