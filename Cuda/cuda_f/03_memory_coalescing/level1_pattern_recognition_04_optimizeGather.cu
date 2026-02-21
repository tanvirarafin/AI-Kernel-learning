/**
 * Optimize Gather - Kernel 4 from level1_pattern_recognition.cu
 * 
 * This kernel demonstrates gather pattern optimization.
 * While reads may be uncoalesced (indirect), writes are coalesced.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 10000
#define MAX_IDX 1000

__global__ void optimizeGather(float *input, float *output, int *indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // Gather pattern - read is inherently uncoalesced (indirect access)
        int srcIdx = indices[tid];
        
        // But write is coalesced - consecutive threads write consecutive locations
        output[tid] = input[srcIdx] * 2.0f;
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

void initIndices(int *indices, int n, int maxVal) {
    for (int i = 0; i < n; i++) {
        indices[i] = i % maxVal;  // Deterministic for verification
    }
}

bool verifyArray(float *result, float *input, int *indices, int n) {
    for (int i = 0; i < n; i++) {
        float expected = input[indices[i]] * 2.0f;
        if (fabsf(result[i] - expected) > 1e-5f) return false;
    }
    return true;
}

int main() {
    printf("=== Optimize Gather Pattern ===\n\n");

    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    int *h_indices = (int*)malloc(N * sizeof(int));

    initArray(h_in, N);
    initIndices(h_indices, N, MAX_IDX);

    float *d_in, *d_out;
    int *d_indices;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    optimizeGather<<<gridSize, blockSize>>>(d_in, d_out, d_indices, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyArray(h_out, h_in, h_indices, N)) {
        printf("Gather optimization PASSED\n");
    } else {
        printf("Gather optimization FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    free(h_indices);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices);

    printf("\n=== Key Takeaways ===\n");
    printf("- Gather pattern: indirect reads are inherently uncoalesced\n");
    printf("- Ensure writes are coalesced (consecutive threads -> consecutive addresses)\n");
    printf("- Consider transposing data layout if gather is frequent\n");

    return 0;
}
