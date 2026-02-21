/**
 * Fix Strided Access - Kernel 1 from level1_pattern_recognition.cu
 * 
 * This kernel demonstrates coalesced memory access using grid-stride loop.
 * Original had strided access pattern which is inefficient.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 10000

__global__ void fixStridedAccess(float *input, float *output, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Coalesced access using grid-stride loop
    // Each thread processes consecutive elements
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * 2.0f;
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i * 0.5f;
}

bool verifyArray(float *result, float *expected, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

int main() {
    printf("=== Fix Strided Access (Coalesced) ===\n\n");

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
        printf("Strided access fix PASSED\n");
    } else {
        printf("Strided access fix FAILED\n");
    }

    // Cleanup
    free(h_in);
    free(h_out);
    free(h_expected);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n=== Key Takeaways ===\n");
    printf("- Use grid-stride loop for coalesced access\n");
    printf("- Consecutive threads should access consecutive memory\n");
    printf("- Strided access causes uncoalesced memory transactions\n");

    return 0;
}
