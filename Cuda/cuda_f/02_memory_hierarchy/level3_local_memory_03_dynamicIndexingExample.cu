/**
 * Dynamic Indexing Example - Kernel 3 from level3_local_memory.cu
 * 
 * This kernel demonstrates how dynamic indexing (index determined at runtime)
 * prevents compiler from optimizing to registers, causing local memory usage.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000
#define ARRAY_SIZE 100

__global__ void dynamicIndexingExample(float *input, float *output, int n, int *indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Local array for temporary storage
    float temp[ARRAY_SIZE];

    // Use dynamic indexing (index determined at runtime)
    // This prevents compiler from optimizing to registers
    int dynamicIdx = indices[idx % 10];  // Runtime-determined index

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // Dynamic indexing causes local memory usage
        int dataIdx = idx * ARRAY_SIZE + (i + dynamicIdx) % ARRAY_SIZE;
        if (dataIdx < n) {
            temp[i] = input[dataIdx];
        } else {
            temp[i] = 0.0f;
        }
    }

    // Compute sum and store to output
    float sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += temp[i];
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
    int *d_indices;

    int dataSize = N * ARRAY_SIZE;

    // Allocate host memory
    h_input = (float*)malloc(dataSize * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));
    int h_indices[10] = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};

    // Initialize
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_indices, 10 * sizeof(int));

    // Copy to device
    cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, 10 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("=== Dynamic Indexing Example (local memory) ===\n\n");

    // Test dynamic indexing version
    printf("Running dynamic indexing version (local memory)...\n");
    dynamicIndexingExample<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_indices);
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
    cudaFree(d_indices);
    free(h_input);
    free(h_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Dynamic indexing prevents register allocation\n");
    printf("- Index determined at runtime causes local memory usage\n");
    printf("- Local memory is as slow as global memory!\n");

    return 0;
}
