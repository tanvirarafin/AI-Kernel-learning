/*
 * CUDA Streams Level 1: Stream Basics - Function 1
 *
 * This file demonstrates basic stream creation, management, and usage.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// Simple kernel for stream testing
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// FUNCTION 1: Create and Use Stream
// Complete stream creation and kernel launch
// ============================================================================
void basicStreamUsage(float *d_a, float *d_b, float *d_c, int n) {
    cudaStream_t stream;

    // Create stream
    cudaStreamCreate(&stream);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel in stream
    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);

    // Synchronize stream
    cudaStreamSynchronize(stream);

    // Destroy stream
    cudaStreamDestroy(stream);
}

// ============================================================================
// FUNCTION 2: Multiple Sequential Streams
// Launch kernels in multiple streams sequentially
// ============================================================================
void multipleStreams(float *d_data, int n, int numStreams) {
    cudaStream_t *streams = new cudaStream_t[numStreams];

    // Create all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel in each stream
    for (int i = 0; i < numStreams; i++) {
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_data, d_data, d_data, n);
    }

    // Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Destroy all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    delete[] streams;
}

// ============================================================================
// FUNCTION 3: Stream with Error Checking
// Proper error handling for stream operations
// ============================================================================
cudaError_t safeStreamUsage(float *d_a, float *d_b, float *d_c, int n) {
    cudaStream_t stream;
    cudaError_t err;

    // Create stream with error checking
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        return err;
    }

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel and check for errors
    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    // Synchronize and check for execution errors
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return err;
    }

    // Destroy stream
    err = cudaStreamDestroy(stream);

    return err;
}

int main() {
    printf("=== CUDA Streams Level 1: Stream Basics ===\n\n");

    const int N = 1000000;
    size_t size = N * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 0.25f;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count: %d\n\n", prop.asyncEngineCount);

    // Test 1: Basic stream usage
    printf("Test 1: Basic stream usage\n");
    basicStreamUsage(d_a, d_b, d_c, N);
    printf("  ✓ Basic stream completed\n");

    // Verify
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("  Sample result: c[0] = %.2f (expected %.2f)\n",
           h_c[0], h_a[0] + h_b[0]);

    // Test 2: Multiple streams
    printf("\nTest 2: Multiple sequential streams\n");
    const int NUM_STREAMS = 4;
    multipleStreams(d_a, N / NUM_STREAMS, NUM_STREAMS);
    printf("  ✓ Multiple streams completed\n");

    // Test 3: Safe stream usage with error checking
    printf("\nTest 3: Stream with error checking\n");
    cudaError_t err = safeStreamUsage(d_a, d_b, d_c, N);
    if (err == cudaSuccess) {
        printf("  ✓ Safe stream usage completed\n");
    } else {
        printf("  ✗ Error: %s\n", cudaGetErrorString(err));
    }

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\n=== Key Takeaways ===\n");
    printf("- cudaStreamCreate: Create a new stream\n");
    printf("- cudaStreamDestroy: Clean up stream resources\n");
    printf("- cudaStreamSynchronize: Wait for stream to complete\n");
    printf("- Kernels in same stream execute sequentially\n");

    return 0;
}
