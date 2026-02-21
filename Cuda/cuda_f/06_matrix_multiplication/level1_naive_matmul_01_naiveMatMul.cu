/*
 * Matrix Multiplication Level 1: Naive Implementation - Kernel 1
 *
 * This kernel demonstrates basic matrix multiplication on GPU
 * with 2D thread indexing.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 512

// ============================================================================
// KERNEL 1: Naive Matrix Multiplication
// C = A * B where all matrices are in row-major format
// ============================================================================
__global__ void naiveMatMul(float *A, float *B, float *C, int width) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds checking
    if (row < width && col < width) {
        // Compute dot product of row from A and column from B
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        // Store result in C (row-major layout)
        C[row * width + col] = sum;
    }
}

// ============================================================================
// KERNEL 2: Naive Matrix Multiplication 1D
// Use 1D grid indexing for potentially simpler launch configuration
// ============================================================================
__global__ void naiveMatMul1D(float *A, float *B, float *C, int width) {
    // Map 1D thread index to 2D matrix coordinates
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate row and column from 1D thread index
    int row = tid / width;
    int col = tid % width;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Utility functions
void initMatrix(float *mat, int width, float val) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = val;
    }
}

bool verifyMatMul(float *C, float *A, float *B, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0.0f;
            for (int k = 0; k < width; k++) {
                expected += A[row * width + k] * B[k * width + col];
            }
            float diff = fabsf(C[row * width + col] - expected);
            if (diff > 1e-2f * width) {
                return false;
            }
        }
    }
    return true;
}

void printMatrixCorner(float *mat, int width, const char *label) {
    printf("%s (top-left 5x5):\n", label);
    for (int i = 0; i < 5; i++) {
        printf("  ");
        for (int j = 0; j < 5; j++) {
            printf("%10.2f ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Matrix Multiplication Level 1: Naive Implementation ===\n\n");

    const int WIDTH = MATRIX_SIZE;
    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices: A = all 1s, B = all 2s
    initMatrix(h_A, WIDTH, 1.0f);
    initMatrix(h_B, WIDTH, 2.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch configuration for 2D grid
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + 31) / 32, (WIDTH + 31) / 32);

    printf("Matrix size: %d x %d\n", WIDTH, WIDTH);
    printf("Grid: %d x %d blocks, Block: %d x %d threads\n\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Test 1: Naive matrix multiplication (2D)
    printf("Test 1: Naive matrix multiplication (2D grid)\n");
    naiveMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
        printMatrixCorner(h_C, WIDTH, "Result C (5x5 corner)");
    } else {
        printf("  ✗ FAILED - Check thread indexing and dot product\n");
    }

    // Test 2: Naive with 1D grid
    printf("\nTest 2: Naive matrix multiplication (1D grid)\n");
    int totalThreads = WIDTH * WIDTH;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    naiveMatMul1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (verifyMatMul(h_C, h_A, h_B, WIDTH)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Check 1D to 2D mapping\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n=== Key Takeaways ===\n");
    printf("- Matrix multiplication is O(n³) - great for parallelization\n");
    printf("- Row-major layout: element (row, col) at index row*width + col\n");
    printf("- 2D thread mapping matches 2D data naturally\n");

    return 0;
}
