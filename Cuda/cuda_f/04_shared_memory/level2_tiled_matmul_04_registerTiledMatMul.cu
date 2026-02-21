/**
 * Register Tiled Matrix Multiply - Kernel 4 from level2_tiled_matmul.cu
 * 
 * This kernel demonstrates register tiling where each thread computes multiple outputs.
 * Higher arithmetic intensity through increased register usage.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 512
#define TILE_WIDTH 16
#define THREAD_TILE 2

__global__ void registerTiledMatMul(float *A, float *B, float *C, int width) {
    // Each thread computes a 2x2 block of the output
    __shared__ float tileA[TILE_WIDTH * THREAD_TILE][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH * THREAD_TILE];

    int row = blockIdx.y * (TILE_WIDTH * THREAD_TILE) + threadIdx.y * THREAD_TILE;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // 2x2 accumulator in registers
    float accum[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles cooperatively
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y * THREAD_TILE;

        // Load A tile (each thread loads 2 rows for its 2x2 block)
        for (int i = 0; i < THREAD_TILE; i++) {
            int loadRow = row + i;
            if (loadRow < width && aCol < width) {
                tileA[threadIdx.y * THREAD_TILE + i][threadIdx.x] = A[loadRow * width + aCol];
            } else {
                tileA[threadIdx.y * THREAD_TILE + i][threadIdx.x] = 0.0f;
            }
        }

        // Load B tile (each thread loads 2 columns)
        for (int i = 0; i < THREAD_TILE; i++) {
            int loadCol = col + i;
            if (bRow < width && loadCol < width) {
                tileB[threadIdx.y][threadIdx.x * THREAD_TILE + i] = B[bRow * width + loadCol];
            } else {
                tileB[threadIdx.y][threadIdx.x * THREAD_TILE + i] = 0.0f;
            }
        }

        __syncthreads();

        // Compute 2x2 output block
        for (int k = 0; k < TILE_WIDTH; k++) {
            for (int i = 0; i < THREAD_TILE; i++) {
                for (int j = 0; j < THREAD_TILE; j++) {
                    accum[i][j] += tileA[threadIdx.y * THREAD_TILE + i][k] *
                                   tileB[k][threadIdx.x * THREAD_TILE + j];
                }
            }
        }

        __syncthreads();
    }

    // Store 2x2 output block
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            int storeRow = row + i;
            int storeCol = col + j;
            if (storeRow < width && storeCol < width) {
                C[storeRow * width + storeCol] = accum[i][j];
            }
        }
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
    printf("=== Register Tiled Matrix Multiplication ===\n\n");

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
                 (WIDTH + TILE_WIDTH * THREAD_TILE - 1) / (TILE_WIDTH * THREAD_TILE));

    registerTiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("Register tiled matmul correctness PASSED\n");
    } else {
        printf("Register tiled matmul correctness FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n=== Key Takeaways ===\n");
    printf("- Each thread computes multiple outputs (2x2 = 4 elements)\n");
    printf("- Higher arithmetic intensity (more compute per memory load)\n");
    printf("- Uses more registers per thread\n");

    return 0;
}
