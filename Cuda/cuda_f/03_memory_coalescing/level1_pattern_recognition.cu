/*
 * Memory Coalescing Level 1: Pattern Recognition
 * 
 * EXERCISE: Fix uncoalesced memory access patterns to achieve
 * efficient memory throughput.
 * 
 * SKILLS PRACTICED:
 * - Identifying strided access patterns
 * - Converting to coalesced access
 * - Understanding memory transaction efficiency
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 1: Strided Access Problem
 * This kernel has BAD memory access pattern - fix it!
// Current: Thread 0 accesses index 0, Thread 1 accesses index stride, etc.
// Goal: Thread 0 accesses index 0, Thread 1 accesses index 1, etc.
// ============================================================================
__global__ void fixStridedAccess(float *input, float *output, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Fix this strided access pattern
    // Current (BAD): Each thread accesses memory stride apart
    // for (int i = tid * stride; i < n; i += stride) {
    //     output[i] = input[i] * 2.0f;
    // }
    
    // TODO: Implement coalesced access
    // Each thread should process consecutive elements
    // Hint: Use grid-stride loop instead
}

// ============================================================================
// KERNEL 2: Column-Major vs Row-Major Access
// Fix the access pattern for row-major stored matrix
// ============================================================================
__global__ void fixColumnAccess(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TODO: This accesses memory in column-major order (BAD for row-major data)
    // Current (BAD): matrix[row][col] accessed as matrix[col][row]
    // Fix: Ensure consecutive threads access consecutive memory
    
    if (row < height && col < width) {
        // TODO: Fix the indexing for row-major storage
        // Row-major: index = row * width + col
        int idx = /* YOUR CODE HERE */ 0;
        matrix[idx] = matrix[idx] * 2.0f;
    }
}

// ============================================================================
// KERNEL 3: Interleaved Access Pattern
// Fix access pattern for interleaved data (e.g., RGB pixels)
// ============================================================================
__global__ void fixInterleavedAccess(float *data, int n, int channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: This kernel processes one channel at a time with strided access
    // Current (BAD): Processing channel 0: indices 0, channels, 2*channels...
    // Goal: Process all channels for consecutive pixels
    
    // TODO: Restructure to have consecutive threads process consecutive pixels
    // Each thread handles all channels for one or more pixels
    if (tid * channels < n) {
        int pixelIdx = tid;
        for (int c = 0; c < channels; c++) {
            int idx = pixelIdx * channels + c;
            if (idx < n) {
                data[idx] = data[idx] * 1.5f;
            }
        }
    }
}

// ============================================================================
// KERNEL 4: Gather Pattern to Coalesced
// Convert gather pattern (indirect access) to coalesced where possible
// ============================================================================
__global__ void optimizeGather(float *input, float *output, int *indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // TODO: This is a gather pattern - inherently uncoalesced
        // However, we can optimize the WRITE to be coalesced
        // Current: Both read and write may be uncoalesced
        // Goal: At minimum, ensure coalesced WRITE pattern
        
        int srcIdx = indices[tid];
        // TODO: Ensure output write is coalesced (it should be with tid)
        output[tid] = input[srcIdx] * 2.0f;
    }
}

// Utility functions
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

void initIndices(int *indices, int n, int maxVal) {
    for (int i = 0; i < n; i++) {
        indices[i] = rand() % maxVal;
    }
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
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
    printf("=== Memory Coalescing Level 1: Pattern Recognition ===\n\n");
    
    // Test 1: Fix strided access
    printf("Testing strided access fix...\n");
    const int N = 10000;
    const int STRIDE = 256;
    
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));
    initArray(h_in, N);
    
    // Compute expected result
    for (int i = 0; i < N; i++) {
        h_expected[i] = h_in[i] * 2.0f;
    }
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cudaMemset(d_out, 0, N * sizeof(float));
    fixStridedAccess<<<gridSize, blockSize>>>(d_in, d_out, N, STRIDE);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyArray(h_out, h_expected, N)) {
        printf("✓ Strided access fix PASSED\n");
    } else {
        printf("✗ Strided access fix FAILED - Implement coalesced grid-stride loop\n");
    }
    
    // Test 2: Fix column access
    printf("\nTesting column access fix...\n");
    const int W = 256, H = 256;
    const int N_MAT = W * H;
    
    float *h_mat = (float*)malloc(N_MAT * sizeof(float));
    initArray(h_mat, N_MAT);
    
    float *d_mat;
    cudaMalloc(&d_mat, N_MAT * sizeof(float));
    cudaMemcpy(d_mat, h_mat, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    
    fixColumnAccess<<<grid, block>>>(d_mat, W, H);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_mat, d_mat, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (verifyMatrix(h_mat, W, H)) {
        printf("✓ Column access fix PASSED\n");
    } else {
        printf("✗ Column access fix FAILED - Use row-major indexing\n");
    }
    
    // Cleanup
    free(h_in);
    free(h_out);
    free(h_expected);
    free(h_mat);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mat);
    
    printf("\n=== Level 1 Complete ===\n");
    printf("Next: Try level2_matrix_transpose.cu for shared memory optimization\n");
    
    return 0;
}
