/*
 * Solution: Thread Hierarchy Level 1 - Basic Indexing
 * 
 * This is a REFERENCE SOLUTION. Try the exercise first!
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// SOLUTION: Complete kernel with proper indexing
__global__ void basicIndexing(float *input, float *output, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// SOLUTION: Grid-stride version
__global__ void gridStrideIndexing(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop handles any input size
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * 2.0f;
    }
}

// SOLUTION: 2D indexing
__global__ void indexing2D(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        output[idx] = input[idx] * 2.0f;
    }
}

void verifyResults(float *output, int n, float expected) {
    bool success = true;
    for (int i = 0; i < n && i < 10; i++) {
        if (output[i] != expected) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, output[i]);
            success = false;
            break;
        }
    }
    if (success) printf("✓ Verification passed!\n");
}

int main() {
    printf("=== Solution: Thread Hierarchy Level 1 ===\n\n");
    
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Test 1: Basic indexing
    printf("Test 1: Basic indexing\n");
    basicIndexing<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_output, N, 2.0f);
    
    // Test 2: Grid-stride
    printf("\nTest 2: Grid-stride indexing\n");
    gridSize = 64;  // Fewer blocks to test striding
    gridStrideIndexing<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_output, N, 2.0f);
    
    // Test 3: 2D indexing
    printf("\nTest 3: 2D indexing\n");
    int width = 1024, height = 1024;
    dim3 block2D(32, 32);
    dim3 grid2D((width + 31) / 32, (height + 31) / 32);
    indexing2D<<<grid2D, block2D>>>(d_input, d_output, width, height);
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_output, width * height, 2.0f);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("\n=== Key Points ===\n");
    printf("1. Global index = blockIdx * blockDim + threadIdx\n");
    printf("2. Always check bounds before accessing memory\n");
    printf("3. Grid-stride loops handle inputs larger than thread count\n");
    printf("4. 2D indexing: row * width + col for row-major layout\n");
    
    return 0;
}
