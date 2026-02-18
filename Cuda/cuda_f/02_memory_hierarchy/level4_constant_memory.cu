/**
 * Level 4: Constant Memory Optimization
 * 
 * GOAL: Learn to use constant memory for read-only uniform data access.
 * 
 * CONCEPTS:
 * - Constant memory declaration (__constant__)
 * - Constant memory caching
 * - Broadcast optimization (all threads read same address)
 * - 64KB constant memory limit
 * 
 * EXERCISE:
 * 1. Declare and copy data to constant memory
 * 2. Use constant memory in kernel for coefficients/parameters
 * 3. Compare performance with global memory
 * 
 * HINTS:
 * - Constant memory is fastest when all threads read same location
 * - Use cudaMemcpyToSymbol for copying to constant memory
 * - Constant memory is cached with special constant cache
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define COEFF_COUNT 10

// TODO: Declare constant memory for coefficients
// Hint: __constant__ float d_coeffs[COEFF_COUNT];
/* YOUR CODE HERE */

// TODO: Complete this kernel using constant memory for coefficients
// Task:
//   1. Apply polynomial transformation using constant memory coefficients
//   2. Formula: output[i] = sum(coeffs[j] * pow(input[i], j)) for j = 0 to COEFF_COUNT-1
//   3. All threads read the same coefficients (broadcast optimization)
__global__ void constantMemoryTransform(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float x = input[idx];
    float result = 0.0f;
    
    // TODO: Loop through coefficients in constant memory
    // Apply polynomial: result += d_coeffs[j] * pow(x, j)
    /* YOUR CODE HERE */
    
    output[idx] = result;
}

// Compare with global memory version
__global__ void globalMemoryTransform(float *input, float *output, int n, float *coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float x = input[idx];
    float result = 0.0f;
    
    // Same computation but coefficients in global memory
    for (int j = 0; j < COEFF_COUNT; j++) {
        result += coeffs[j] * powf(x, j);
    }
    
    output[idx] = result;
}

// TODO: Complete this kernel that uses constant memory for a lookup table
// Task: Implement a sine approximation using constant memory lookup table
__global__ void constantMemoryLookup(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // TODO: Assume constant memory has a precomputed sine table
    // Access the table based on input value (normalized to 0-255)
    // Hint: int tableIdx = (int)(input[idx] * 255.0f) % 256;
    //       output[idx] = sinTable[tableIdx];
    /* YOUR CODE HERE */
}

void initializeCoefficients(float *coeffs) {
    // Initialize polynomial coefficients
    for (int i = 0; i < COEFF_COUNT; i++) {
        coeffs[i] = 1.0f / (i + 1);  // 1, 1/2, 1/3, 1/4, ...
    }
}

void initializeSinTable(float *table) {
    for (int i = 0; i < 256; i++) {
        table[i] = sinf(2.0f * M_PI * i / 256.0f);
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
    float *d_coeffs;
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
    cudaMalloc(&d_coeffs, COEFF_COUNT * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    
    // TODO: Copy coefficients to constant memory
    // Hint: cudaMemcpyToSymbol(symbol_name, source, size)
    // Example: cudaMemcpyToSymbol(d_coeffs, h_coeffs, COEFF_COUNT * sizeof(float));
    /* YOUR CODE HERE */
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("=== Constant Memory Exercise ===\n\n");
    
    // Test constant memory version
    printf("1. Running constant memory version...\n");
    constantMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");
    
    // Test global memory version (for comparison)
    printf("\n2. Running global memory version...\n");
    globalMemoryTransform<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_coeffs);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    printFirstElements(h_output, 5, "   Results (first 5)");
    
    // TODO: Add timing code to compare performance
    // Hint: Use cudaEventRecord for timing
    /* YOUR CODE HERE - Optional: Add timing comparison */
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_coeffs);
    free(h_input);
    free(h_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Constant memory is cached and optimized for broadcast reads\n");
    printf("- When all threads read same address, constant memory is very fast\n");
    printf("- Use cudaMemcpyToSymbol to copy data to constant memory\n");
    printf("- Limited to 64KB total constant memory\n");
    
    return 0;
}
