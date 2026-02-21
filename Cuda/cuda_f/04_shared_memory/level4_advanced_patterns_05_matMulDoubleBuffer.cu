/**
 * Matrix Multiply Double Buffer - Kernel 5 from level4_advanced_patterns.cu
 * 
 * This kernel demonstrates double-buffered matrix multiplication.
 * Overlaps memory loads with computation using two buffer sets.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 256
#define TILE_WIDTH 16

__global__ void matMulDoubleBuffer(float *A, float *B, float *C, int width) {
    // Double-buffered shared memory (2 sets of tiles)
    __shared__ float As[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[2][TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    // Prefetch first tile
    int t = 0;
    int aCol = t * TILE_WIDTH + threadIdx.x;
    int bRow = t * TILE_WIDTH + threadIdx.y;

    if (row < width && aCol < width) {
        As[0][threadIdx.y][threadIdx.x] = A[row * width + aCol];
    } else {
        As[0][threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (bRow < width && col < width) {
        Bs[0][threadIdx.y][threadIdx.x] = B[bRow * width + col];
    } else {
        Bs[0][threadIdx.y][threadIdx.x] = 0.0f;
    }

    for (t = 0; t < numTiles; t++) {
        int curBuf = t % 2;
        int nextBuf = (t + 1) % 2;

        __syncthreads();

        // Compute with current buffer
        for (int k = 0; k < TILE_WIDTH; k++) {
            accumulator += As[curBuf][threadIdx.y][k] * Bs[curBuf][k][threadIdx.x];
        }

        __syncthreads();

        // Prefetch next tile (if not last iteration)
        if (t + 1 < numTiles) {
            aCol = (t + 1) * TILE_WIDTH + threadIdx.x;
            bRow = (t + 1) * TILE_WIDTH + threadIdx.y;

            if (row < width && aCol < width) {
                As[nextBuf][threadIdx.y][threadIdx.x] = A[row * width + aCol];
            } else {
                As[nextBuf][threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (bRow < width && col < width) {
                Bs[nextBuf][threadIdx.y][threadIdx.x] = B[bRow * width + col];
            } else {
                Bs[nextBuf][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
    }

    if (row < width && col < width) {
        C[row * width + col] = accumulator;
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
    printf("=== Double-Buffered Matrix Multiplication ===\n\n");

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

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
                 (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulDoubleBuffer<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("Double-buffered matmul PASSED\n");
    } else {
        printf("Double-buffered matmul FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n=== Key Takeaways ===\n");
    printf("- Double-buffering overlaps compute with memory\n");
    printf("- While computing with buffer[t%2], prefetch buffer[(t+1)%2]\n");
    printf("- Hides memory latency through pipelining\n");

    return 0;
}
