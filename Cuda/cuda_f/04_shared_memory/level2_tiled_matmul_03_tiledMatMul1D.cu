/**
 * Tiled Matrix Multiply 1D - Kernel 3 from level2_tiled_matmul.cu
 * 
 * This kernel demonstrates tiled matrix multiplication with 1D block configuration.
 * Uses 1D thread indexing for potentially better occupancy.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 512
#define TILE_WIDTH 32

__global__ void tiledMatMul1D(float *A, float *B, float *C, int width) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Map 1D thread index to 2D tile coordinates
    int tid = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + (tid / TILE_WIDTH);
    int col = blockIdx.x * TILE_WIDTH + (tid % TILE_WIDTH);

    float sum = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles with 1D thread indexing
        int aCol = t * TILE_WIDTH + (tid % TILE_WIDTH);
        int bRow = t * TILE_WIDTH + (tid / TILE_WIDTH);

        // Load A tile
        if (row < width && aCol < width) {
            tileA[tid / TILE_WIDTH][tid % TILE_WIDTH] = A[row * width + aCol];
        } else {
            tileA[tid / TILE_WIDTH][tid % TILE_WIDTH] = 0.0f;
        }

        // Load B tile
        if (bRow < width && col < width) {
            tileB[tid / TILE_WIDTH][tid % TILE_WIDTH] = B[bRow * width + col];
        } else {
            tileB[tid / TILE_WIDTH][tid % TILE_WIDTH] = 0.0f;
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[tid / TILE_WIDTH][k] * tileB[k][tid % TILE_WIDTH];
        }

        __syncthreads();
    }

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
    printf("=== Tiled Matrix Multiplication (1D Block) ===\n\n");

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

    dim3 block(TILE_WIDTH * TILE_WIDTH);  // 1D block
    dim3 grid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
              (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatMul1D<<<grid, block>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("Tiled matmul 1D correctness PASSED\n");
    } else {
        printf("Tiled matmul 1D correctness FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n=== Key Takeaways ===\n");
    printf("- 1D block can improve occupancy on some GPUs\n");
    printf("- Map 1D thread index to 2D coordinates\n");
    printf("- Same tiling algorithm applies\n");

    return 0;
}
