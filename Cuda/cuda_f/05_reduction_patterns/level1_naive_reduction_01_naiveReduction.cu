/*
 * Reduction Patterns Level 1: Naive Reduction - Kernel 1
 *
 * This kernel demonstrates the basic tree reduction pattern
 * using global memory (slow baseline).
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// ============================================================================
// KERNEL 1: Naive Global Memory Reduction
// Each block reduces a portion, then atomic add to global result
// ============================================================================
__global__ void naiveReduction(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread starts with one element (or 0 if out of bounds)
    float partialSum = (idx < n) ? input[idx] : 0.0f;

    // Implement strided reduction loop
    // Note: This naive version reads from global memory in the loop
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            int partnerIdx = idx + stride;
            if (partnerIdx < n) {
                partialSum += input[partnerIdx];
            }
        }
        __syncthreads();
    }

    // Thread 0 of each block writes partial sum to output
    if (tid == 0) {
        atomicAdd(output, partialSum);
    }
}

// ============================================================================
// KERNEL 2: Sequential Addressing Reduction
// Uses sequential addressing for better memory access pattern
// ============================================================================
__global__ void sequentialReduction(float *input, float *output, int n) {
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Implement sequential addressing reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block's result
    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

float reduceCPU(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    printf("=== Reduction Patterns Level 1: Naive Reduction ===\n\n");

    const int N = 1000000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(sizeof(float));

    // Initialize: all 1s, so sum should equal N
    initArray(h_input, N, 1.0f);
    float expected = (float)N;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    printf("Input size: %d elements\n", N);
    printf("Expected sum: %.0f\n\n", expected);

    // Test 1: Naive reduction
    printf("Test 1: Naive global memory reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    naiveReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED (within 1%% tolerance)\n");
    } else {
        printf("  ✗ FAILED - Check reduction loop and atomicAdd\n");
    }

    // Test 2: Sequential addressing reduction
    printf("\nTest 2: Sequential addressing reduction\n");
    cudaMemset(d_output, 0, sizeof(float));
    sequentialReduction<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  Result: %.0f\n", *h_output);
    if (fabsf(*h_output - expected) < expected * 0.01f) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Check sequential reduction loop\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== Key Takeaways ===\n");
    printf("- Tree reduction: halve working set each iteration\n");
    printf("- Sequential addressing: better thread utilization\n");
    printf("- AtomicAdd: combines partial results from blocks\n");

    return 0;
}
