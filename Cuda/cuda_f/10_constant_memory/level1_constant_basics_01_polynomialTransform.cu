/*
 * Constant Memory Level 1: Constant Basics - Kernel 1
 *
 * This file demonstrates constant memory usage for read-only
 * uniform data access.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define COEFF_COUNT 10
#define TABLE_SIZE 256

// Declare constant memory for coefficients
__constant__ float d_coeffs[COEFF_COUNT];

// Declare constant memory for lookup table
__constant__ float sinTable[TABLE_SIZE];

// ============================================================================
// KERNEL 1: Polynomial Transform using Constant Memory
// Apply polynomial transformation using constant memory coefficients
// ============================================================================
__global__ void polynomialTransform(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];
        float result = 0.0f;

        // Use constant memory coefficients
        // Apply polynomial: result += d_coeffs[i] * powf(x, i)
        float x_power = 1.0f;
        for (int i = 0; i < COEFF_COUNT; i++) {
            result += d_coeffs[i] * x_power;
            x_power *= x;
        }

        output[idx] = result;
    }
}

// ============================================================================
// KERNEL 2: Lookup Transform using Constant Memory
// Use constant memory lookup table for sine approximation
// ============================================================================
__global__ void lookupTransform(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Use constant memory lookup table
        int tableIdx = (int)(input[idx] * (TABLE_SIZE - 1)) % TABLE_SIZE;
        output[idx] = sinTable[tableIdx];
    }
}

// ============================================================================
// KERNEL 3: Constant vs Global Memory Comparison
// Compare performance between constant and global memory access
// ============================================================================
__global__ void constantVsGlobal(float *input, float *output,
                                  float *globalCoeffs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];
        float resultConstant = 0.0f;
        float resultGlobal = 0.0f;

        // Compute using constant memory (faster for broadcast reads)
        float x_power = 1.0f;
        for (int i = 0; i < COEFF_COUNT; i++) {
            resultConstant += d_coeffs[i] * x_power;
            x_power *= x;
        }

        // Compute using global memory (slower)
        x_power = 1.0f;
        for (int i = 0; i < COEFF_COUNT; i++) {
            resultGlobal += globalCoeffs[i] * x_power;
            x_power *= x;
        }

        output[idx] = resultConstant;  // Use constant memory result
    }
}

void initCoefficients(float *coeffs) {
    for (int i = 0; i < COEFF_COUNT; i++) {
        coeffs[i] = 1.0f / (i + 1);  // 1, 1/2, 1/3, 1/4, ...
    }
}

void initSinTable(float *table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        table[i] = sinf(2.0f * M_PI * i / TABLE_SIZE);
    }
}

int main() {
    printf("=== Constant Memory Level 1: Constant Basics ===\n\n");

    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float h_coeffs[COEFF_COUNT];
    float h_sinTable[TABLE_SIZE];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    initCoefficients(h_coeffs);
    initSinTable(h_sinTable);

    float *d_input, *d_output, *d_globalCoeffs;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_globalCoeffs, COEFF_COUNT * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_globalCoeffs, h_coeffs, COEFF_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    // Copy coefficients to constant memory
    cudaMemcpyToSymbol(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float));
    
    // Copy sin table to constant memory
    cudaMemcpyToSymbol(sinTable, h_sinTable, TABLE_SIZE * sizeof(float));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Test 1: Polynomial transform with constant memory
    printf("Test 1: Polynomial transform (constant memory)\n");
    printf("  Array size: %d elements\n", N);
    printf("  Coefficients: %d values in constant memory\n\n", COEFF_COUNT);
    
    polynomialTransform<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    printf("  ✓ Completed\n");

    // Test 2: Lookup table transform
    printf("\nTest 2: Lookup table transform (constant memory)\n");
    lookupTransform<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    printf("  ✓ Completed\n");

    // Test 3: Constant vs global comparison
    printf("\nTest 3: Constant vs Global memory comparison\n");
    constantVsGlobal<<<gridSize, blockSize>>>(d_input, d_output, d_globalCoeffs, N);
    cudaDeviceSynchronize();
    printf("  ✓ Completed\n");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_globalCoeffs);

    printf("\n=== Key Takeaways ===\n");
    printf("- __constant__ declares constant memory\n");
    printf("- cudaMemcpyToSymbol copies data to constant memory\n");
    printf("- Constant memory is cached and optimized for broadcast\n");
    printf("- 64KB total constant memory limit\n");
    printf("- Fast when all threads read same address\n");

    return 0;
}
