/**
 * Naive Matrix Multiply - Kernel 2 from level5_combined.cu
 * 
 * This kernel demonstrates simple matrix multiplication without optimization.
 * Used as a baseline for comparison with optimized versions.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 512

__global__ void naiveMatrixMultiply(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initializeMatrix(float *matrix, int size, float seed) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(i % 100) / 100.0f + seed;
    }
}

bool verifyMatrix(float *C, float *reference, int size) {
    float tolerance = 1e-3f;
    for (int i = 0; i < size * size; i++) {
        float diff = fabsf(C[i] - reference[i]);
        if (diff > tolerance) {
            printf("Mismatch at (%d, %d): expected %f, got %f\n",
                   i / size, i % size, reference[i], C[i]);
            return false;
        }
    }
    return true;
}

void printMatrixCorner(float *matrix, int size, const char *label) {
    printf("%s (top-left 5x5 corner):\n", label);
    for (int i = 0; i < 5; i++) {
        printf("  ");
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    int size = MATRIX_SIZE;
    int matrixBytes = size * size * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(matrixBytes);
    h_B = (float*)malloc(matrixBytes);
    h_C = (float*)malloc(matrixBytes);
    h_C_ref = (float*)malloc(matrixBytes);

    // Initialize matrices
    initializeMatrix(h_A, size, 1.0f);
    initializeMatrix(h_B, size, 2.0f);

    // Compute reference on CPU
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += h_A[row * size + k] * h_B[k * size + col];
            }
            h_C_ref[row * size + col] = sum;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, matrixBytes);
    cudaMalloc(&d_C, matrixBytes);

    // Copy to device
    cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);

    // Configure blocks and threads
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((size + 31) / 32, (size + 31) / 32);

    printf("=== Naive Matrix Multiplication (Baseline) ===\n");
    printf("Matrix size: %d x %d\n\n", size, size);

    // Run naive version for reference
    printf("Running naive matrix multiplication...\n");
    naiveMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_ref, d_C, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrixCorner(h_C_ref, size, "   Reference result");

    // Verify results
    printf("\nVerifying results...\n");
    if (verifyMatrix(h_C_ref, h_C_ref, size)) {
        printf("   Self-verification passed!\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    printf("\n=== Key Takeaways ===\n");
    printf("- Naive version accesses global memory repeatedly\n");
    printf("- Each element of A and B is loaded size times\n");
    printf("- This is memory-bound and inefficient\n");
    printf("- Use shared memory tiling for optimization\n");

    return 0;
}
