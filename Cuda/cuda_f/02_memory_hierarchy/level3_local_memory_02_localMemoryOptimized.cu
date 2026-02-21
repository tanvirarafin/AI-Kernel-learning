/**
 * Local Memory Optimized - Kernel 2 from level3_local_memory.cu
 * 
 * This kernel demonstrates an optimized version that avoids local memory
 * by processing data on-the-fly without storing in large arrays.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000
#define ARRAY_SIZE 100

__global__ void localMemoryOptimized(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Instead of storing all values, process them on-the-fly
    // Use a single accumulator variable (stays in register)
    float sum = 0.0f;

    // Loop through ARRAY_SIZE iterations but don't store all values
    // Just accumulate the sum directly - no local memory needed!
    for (int i = 0; i < ARRAY_SIZE; i++) {
        int dataIdx = idx * ARRAY_SIZE + i;
        if (dataIdx < n) {
            sum += input[dataIdx];  // Accumulate directly without storing in array
        }
    }

    output[idx] = sum;
}

void printArrayStats(float *arr, int n, const char *label) {
    float sum = 0.0f, min = arr[0], max = arr[0];
    for (int i = 0; i < n && i < 100; i++) {
        sum += arr[i];
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    printf("%s: sum=%.2f, min=%.2f, max=%.2f (first 100 elements)\n", label, sum, min, max);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    int dataSize = N * ARRAY_SIZE;

    // Allocate host memory
    h_input = (float*)malloc(dataSize * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("=== Local Memory Optimized (register-friendly) ===\n\n");

    // Test optimized version
    printf("Running optimized version (register-friendly)...\n");
    localMemoryOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printArrayStats(h_output, N, "   Results");

    // Verify results - each output should be ARRAY_SIZE (sum of 100 ones)
    bool passed = true;
    for (int i = 0; i < N && i < 100; i++) {
        if (h_output[i] != (float)ARRAY_SIZE) {
            passed = false;
            break;
        }
    }
    printf("\nVerification: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Process data on-the-fly when possible to stay in registers\n");
    printf("- Avoid large local arrays that spill to local memory\n");
    printf("- Use -Xptxas=-v to see register usage\n");

    return 0;
}
