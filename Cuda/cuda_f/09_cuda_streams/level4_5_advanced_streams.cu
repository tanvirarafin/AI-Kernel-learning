/*
 * CUDA Streams Level 4-5: Callbacks and Advanced Patterns
 *
 * EXERCISE: Use stream callbacks and advanced stream patterns.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000

// Callback function
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    int *callbackData = (int*)userData;
    printf("  Callback: Stream completed, data = %d\n", *callbackData);
}

__global__ void simpleKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// ============================================================================
// FUNCTION 1: Stream Callback
 * Add callback to stream for completion notification
 * TODO: Complete the callback pattern
// ============================================================================
void streamWithCallback(float *d_data, int n, int callbackValue) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int *callbackData = new int;
    *callbackData = callbackValue;
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    simpleKernel<<<gridSize, blockSize, 0, stream>>>(d_data, d_data, n);
    
    // TODO: Add callback
    // cudaStreamAddCallback(stream, myCallback, callbackData, 0);
    /* YOUR CODE HERE */
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    delete callbackData;
}

// ============================================================================
// FUNCTION 2: Stream Priority
 * Create streams with different priorities
 * TODO: Complete the priority stream pattern
// ============================================================================
void priorityStreams(float *d_data, int n) {
    cudaStream_t highPriorityStream, lowPriorityStream;
    
    // TODO: Query priority range
    // int leastPriority, greatestPriority;
    // cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    // TODO: Create streams with priorities
    // cudaStreamCreateWithPriority(&highPriorityStream, 
    //     cudaStreamDefault, greatestPriority);
    // cudaStreamCreateWithPriority(&lowPriorityStream,
    //     cudaStreamDefault, leastPriority);
    
    /* YOUR CODE HERE */
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch in both streams
    simpleKernel<<<gridSize, blockSize, 0, highPriorityStream>>>(d_data, d_data, n);
    simpleKernel<<<gridSize, blockSize, 0, lowPriorityStream>>>(d_data, d_data, n);
    
    // Synchronize
    cudaStreamSynchronize(highPriorityStream);
    cudaStreamSynchronize(lowPriorityStream);
    
    // Destroy
    cudaStreamDestroy(highPriorityStream);
    cudaStreamDestroy(lowPriorityStream);
}

int main() {
    printf("=== CUDA Streams Level 4-5: Callbacks & Advanced ===\n\n");
    
    const int N = 100000;
    size_t size = N * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Test 1: Stream callback
    printf("Test 1: Stream callback\n");
    streamWithCallback(d_data, N, 42);
    printf("  ✓ Callback test completed\n");
    
    // Test 2: Priority streams
    printf("\nTest 2: Priority streams\n");
    priorityStreams(d_data, N);
    printf("  ✓ Priority streams completed\n");
    
    // Cleanup
    free(h_data);
    cudaFree(d_data);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- cudaStreamAddCallback: Notify when stream completes\n");
    printf("- Stream priority: Hint to scheduler (hardware dependent)\n");
    printf("- Callbacks run on host, can trigger more work\n");
    printf("- Priority range varies by GPU\n");
    printf("\n=== CUDA Streams Module Complete ===\n");
    
    return 0;
}
