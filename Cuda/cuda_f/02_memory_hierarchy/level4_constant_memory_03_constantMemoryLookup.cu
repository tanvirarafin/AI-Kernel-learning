/**
 * Constant Memory Lookup - Kernel 3 from level4_constant_memory.cu
 * 
 * This kernel demonstrates using constant memory for a lookup table.
 * Precomputed sine table stored in constant memory for fast access.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define TABLE_SIZE 256

// Constant memory for sine lookup table
__constant__ float sinTable[TABLE_SIZE];

__global__ void constantMemoryLookup(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Access the table based on input value (normalized to 0-255)
    // Input is expected to be in range [0, 1]
    int tableIdx = (int)(input[idx] * (TABLE_SIZE - 1)) % TABLE_SIZE;
    
    // Ensure non-negative index
    if (tableIdx < 0) tableIdx = 0;
    
    output[idx] = sinTable[tableIdx];
}

void initializeSinTable(float *table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        table[i] = sinf(2.0f * M_PI * i / TABLE_SIZE);
    }
}

void printFirstElements(float *arr, int n, const char *label) {
    printf("%s: ", label);
    for (int i = 0; i < 5 && i < n; i++) {
        printf("%.4f ", arr[i]);
    }
    printf("\n");
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    float h_sinTable[TABLE_SIZE];

    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize input data (values between 0 and 1)
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // Initialize sine lookup table
    initializeSinTable(h_sinTable);

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy sine table to constant memory
    cudaMemcpyToSymbol(sinTable, h_sinTable, TABLE_SIZE * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("=== Constant Memory Lookup Table ===\n\n");

    // Test constant memory lookup version
    printf("Running constant memory lookup version...\n");
    constantMemoryLookup<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");

    // Verify first few results
    printf("\nVerification (first 5 elements):\n");
    bool passed = true;
    for (int i = 0; i < 5; i++) {
        int tableIdx = (int)(h_input[i] * (TABLE_SIZE - 1));
        if (tableIdx < 0) tableIdx = 0;
        if (tableIdx >= TABLE_SIZE) tableIdx = TABLE_SIZE - 1;
        float expected = h_sinTable[tableIdx];
        float diff = fabsf(h_output[i] - expected);
        if (diff > 1e-5f) {
            printf("  Element %d: expected %.6f, got %.6f - MISMATCH\n", i, expected, h_output[i]);
            passed = false;
        }
    }
    printf("Verification: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Constant memory is ideal for lookup tables\n");
    printf("- All threads reading same table entry = broadcast optimization\n");
    printf("- Precomputed tables avoid expensive runtime calculations\n");
    printf("- 64KB limit applies to all constant memory combined\n");

    return 0;
}
