/*
 * Level 2: Grid-Stride Loop Pattern - Kernel 3: Grid-Stride Accumulation
 *
 * This kernel demonstrates accumulation using grid-stride loops where
 * each thread accumulates a sum of elements it visits.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 3: Grid-Stride with Accumulation
// Each thread accumulates values across its stride iterations
// ============================================================================
__global__ void gridStrideAccumulate(float *input, float *output, int n) {
    // Calculate thread index and stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each output element corresponds to one thread
    // Each thread accumulates sum of elements it visits
    float sum = 0.0f;

    // Implement accumulation loop
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    output[idx] = sum;
}

// Utility functions
void initArray(float *arr, int n, float val) {
    for (int i = 0; i < n; i++) arr[i] = val;
}

bool verifyAccumulate(float *result, float *input, int n, int stride) {
    for (int t = 0; t < stride; t++) {
        float expected = 0.0f;
        for (int i = t; i < n; i += stride) {
            expected += input[i];
        }
        if (result[t] != expected) return false;
    }
    return true;
}

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Thread Hierarchy Level 2: Grid-Stride Accumulation ===\n\n");

    // Test grid-stride accumulation
    const int N_ACC = 1000;
    float *h_input = (float*)malloc(N_ACC * sizeof(float));
    for (int i = 0; i < N_ACC; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N_ACC * sizeof(float));
    cudaMemcpy(d_input, h_input, N_ACC * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N_ACC + blockSize - 1) / blockSize;
    int outputSize = gridSize;

    cudaMalloc(&d_output, gridSize * sizeof(float));

    printf("Launching gridStrideAccumulate kernel...\n");
    printf("  Input size: %d elements (all 1.0f)\n", N_ACC);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Grid size: %d blocks\n", gridSize);
    printf("  Each thread accumulates ~%d elements\n\n", 
           (N_ACC + blockSize * gridSize - 1) / (blockSize * gridSize));

    gridStrideAccumulate<<<gridSize, blockSize>>>(d_input, d_output, N_ACC);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return 1;
    }

    float *h_output = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    printFirstElements(h_output, gridSize, "First 10 thread sums");

    // Verify: each thread should have summed all elements it visited
    bool passAcc = true;
    for (int t = 0; t < gridSize && passAcc; t++) {
        float expected = 0.0f;
        for (int i = t; i < N_ACC; i += gridSize * blockSize) {
            expected += 1.0f;
        }
        if (h_output[t] != expected) passAcc = false;
    }

    if (passAcc) {
        printf("\n✓ Grid-stride accumulation PASSED\n");
    } else {
        printf("\n✗ Grid-stride accumulation FAILED - Check your accumulation logic\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Level 2.3 Complete ===\n");
    printf("Next: Try level2_grid_stride_04_flexibleGridStride.cu for flexible configuration\n");

    return 0;
}
