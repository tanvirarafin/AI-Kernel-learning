/**
 * Tiled Matrix Multiply - Kernel 2 from level2_tiled_matmul.cu
 * 
 * This kernel demonstrates tiled matrix multiplication using shared memory.
 * Reduces global memory accesses by reusing data in shared memory tiles.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 512
#define TILE_WIDTH 32

__global__ void tiledMatMul(float *A, float *B, float *C, int width) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (row < width && aCol < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = t * TILE_WIDTH + threadIdx.y;
        if (bRow < width && col < width) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * width + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize before using shared memory
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void initMatrix(float *mat, int width, float val) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = val;
    }
}

bool verifyMatMul(float *result, float *A, float *B, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0.0f;
            for (int k = 0; k < width; k++) {
                expected += A[row * width + k] * B[k * width + col];
            }
            if (fabsf(result[row * width + col] - expected) > 1e-2f * width) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    printf("=== Tiled Matrix Multiplication ===\n\n");

    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    initMatrix(h_A, WIDTH, 1.0f);
    initMatrix(h_B, WIDTH, 2.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
              (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("Tiled matmul correctness PASSED\n");
    } else {
        printf("Tiled matmul correctness FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n=== Key Takeaways ===\n");
    printf("- Tiling reduces global memory bandwidth\n");
    printf("- Each tile is loaded once, used TILE_WIDTH times\n");
    printf("- Synchronization is critical between phases\n");

    return 0;
}
