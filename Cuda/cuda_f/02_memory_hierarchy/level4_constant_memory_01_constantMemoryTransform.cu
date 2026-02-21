/**
 * Constant Memory Transform - Kernel 1 from level4_constant_memory.cu
 * 
 * This kernel demonstrates using constant memory for read-only coefficients.
 * Constant memory is cached and optimized for broadcast reads (all threads read same address).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define COEFF_COUNT 10

// Constant memory for coefficients
__constant__ float d_coeffs[COEFF_COUNT];

__global__ void constantMemoryTransform(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float x = input[idx];
    float result = 0.0f;

    // Loop through coefficients in constant memory
    // Apply polynomial: result += d_coeffs[j] * pow(x, j)
    // All threads read the same coefficients - broadcast optimization!
    float x_pow = 1.0f;  // x^0 = 1
    for (int j = 0; j < COEFF_COUNT; j++) {
        result += d_coeffs[j] * x_pow;
        x_pow *= x;  // Next power of x
    }

    output[idx] = result;
}

void initializeCoefficients(float *coeffs) {
    // Initialize polynomial coefficients: 1, 1/2, 1/3, 1/4, ...
    for (int i = 0; i < COEFF_COUNT; i++) {
        coeffs[i] = 1.0f / (i + 1);
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
    float h_coeffs[COEFF_COUNT];

    // Allocate host memory
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize input data (values between 0 and 1)
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // Initialize coefficients
    initializeCoefficients(h_coeffs);

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy coefficients to constant memory
    cudaMemcpyToSymbol(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("=== Constant Memory Transform ===\n\n");

    // Test constant memory version
    printf("Running constant memory version...\n");
    constantMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");

    // Verify first few results
    printf("\nVerification (first 5 elements):\n");
    bool passed = true;
    for (int i = 0; i < 5; i++) {
        float x = h_input[i];
        float expected = 0.0f;
        float x_pow = 1.0f;
        for (int j = 0; j < COEFF_COUNT; j++) {
            expected += h_coeffs[j] * x_pow;
            x_pow *= x;
        }
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
    printf("- Constant memory is cached and optimized for broadcast reads\n");
    printf("- When all threads read same address, constant memory is very fast\n");
    printf("- Use cudaMemcpyToSymbol to copy data to constant memory\n");
    printf("- Limited to 64KB total constant memory\n");

    return 0;
}
