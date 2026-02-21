/*
 * CUDA Streams Level 3: Async Memory Transfers
 *
 * EXERCISE: Overlap computation with data transfer using async memcpy.
 *
 * CONCEPTS:
 * - cudaMemcpyAsync
 * - Pinned (page-locked) memory
 * - Transfer/compute overlap
 * - Full-duplex transfers
 *
 * SKILLS PRACTICED:
 * - cudaMallocHost for pinned memory
 * - cudaMemcpyAsync
 * - Transfer/compute overlap pattern
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define NUM_CHUNKS 4

__global__ void processKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// FUNCTION 1: Async Transfer with Pinned Memory
 * Use pinned memory for async transfers
 * TODO: Complete the async transfer pattern
// ============================================================================
void asyncTransfer(float *h_data, float *d_data, int n) {
    float *h_pinned;
    
    // TODO: Allocate pinned host memory
    // cudaMallocHost(&h_pinned, n * sizeof(float));
    /* YOUR CODE HERE */
    
    // Copy to pinned memory
    memcpy(h_pinned, h_data, n * sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // TODO: Async transfer to device
    // cudaMemcpyAsync(d_data, h_pinned, n * sizeof(float),
    //                 cudaMemcpyHostToDevice, stream);
    /* YOUR CODE HERE */
    
    cudaStreamSynchronize(stream);
    
    // Copy result back
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // TODO: Free pinned memory
    // cudaFreeHost(h_pinned);
    /* YOUR CODE HERE */
    
    cudaStreamDestroy(stream);
}

// ============================================================================
// FUNCTION 2: Overlap Transfer and Compute
 * Transfer next chunk while processing current chunk
 * TODO: Complete the overlap pattern
// ============================================================================
void overlapTransferCompute(float **h_data, float **d_data, int n, int numChunks) {
    cudaStream_t *streams = new cudaStream_t[numChunks];
    float **h_pinned = new float*[numChunks];
    
    // TODO: Allocate pinned memory for each chunk
    for (int i = 0; i < numChunks; i++) {
        // cudaMallocHost(&h_pinned[i], n * sizeof(float));
        /* YOUR CODE HERE */
    }
    
    // TODO: Create streams
    for (int i = 0; i < numChunks; i++) {
        // cudaStreamCreate(&streams[i]);
        /* YOUR CODE HERE */
    }
    
    int blockSize = 256;
    
    // TODO: Pipeline: transfer and compute for each chunk
    for (int i = 0; i < numChunks; i++) {
        // Copy to pinned memory
        memcpy(h_pinned[i], h_data[i], n * sizeof(float));
        
        // Async transfer H2D
        // cudaMemcpyAsync(d_data[i], h_pinned[i], n * sizeof(float),
        //                 cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel
        // int gridSize = (n + blockSize - 1) / blockSize;
        // processKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_data[i], d_data[i], n);
        
        // Async transfer D2H
        // cudaMemcpyAsync(h_data[i], d_data[i], n * sizeof(float),
        //                 cudaMemcpyDeviceToHost, streams[i]);
        
        /* YOUR CODE HERE */
    }
    
    // TODO: Synchronize all streams
    for (int i = 0; i < numChunks; i++) {
        // cudaStreamSynchronize(streams[i]);
        /* YOUR CODE HERE */
    }
    
    // Cleanup
    for (int i = 0; i < numChunks; i++) {
        // cudaFreeHost(h_pinned[i]);
        // cudaStreamDestroy(streams[i]);
        /* YOUR CODE HERE */
    }
    
    delete[] streams;
    delete[] h_pinned;
}

// ============================================================================
// FUNCTION 3: Bidirectional Async Transfers
 * Simultaneous H2D and D2H transfers
 * TODO: Complete the bidirectional transfer
// ============================================================================
void bidirectionalTransfer(float *h_in, float *h_out, 
                           float *d_in, float *d_out, int n) {
    float *h_pinnedIn, *h_pinnedOut;
    
    // TODO: Allocate pinned memory
    /* YOUR CODE HERE */
    
    cudaStream_t streamH2D, streamD2H;
    // TODO: Create separate streams for H2D and D2H
    /* YOUR CODE HERE */
    
    // Prepare data
    memcpy(h_pinnedIn, h_in, n * sizeof(float));
    
    // TODO: Launch async transfers in separate streams
    // cudaMemcpyAsync(d_in, h_pinnedIn, n * sizeof(float),
    //                 cudaMemcpyHostToDevice, streamH2D);
    // cudaMemcpyAsync(h_pinnedOut, d_out, n * sizeof(float),
    //                 cudaMemcpyDeviceToHost, streamD2H);
    /* YOUR CODE HERE */
    
    // TODO: Synchronize both streams
    /* YOUR CODE HERE */
    
    // Copy result
    memcpy(h_out, h_pinnedOut, n * sizeof(float));
    
    // TODO: Free pinned memory and streams
    /* YOUR CODE HERE */
}

int main() {
    printf("=== CUDA Streams Level 3: Async Memory Transfers ===\n\n");
    
    const int N = 1000000;
    size_t size = N * sizeof(float);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("Can overlap transfer/compute: %s\n\n",
           prop.deviceOverlap ? "Yes" : "No");
    
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }
    
    // Test 1: Async transfer with pinned memory
    printf("Test 1: Async transfer with pinned memory\n");
    asyncTransfer(h_data, d_data, N);
    printf("  ✓ Async transfer completed\n");
    printf("  Sample: h_data[0] = %.4f\n", h_data[0]);
    
    // Test 2: Overlap transfer and compute
    printf("\nTest 2: Overlap transfer and compute\n");
    const int NUM_CHUNKS = 4;
    float **h_chunks = new float*[NUM_CHUNKS];
    float **d_chunks = new float*[NUM_CHUNKS];
    
    for (int i = 0; i < NUM_CHUNKS; i++) {
        h_chunks[i] = (float*)malloc(size / NUM_CHUNKS);
        cudaMalloc(&d_chunks[i], size / NUM_CHUNKS);
        for (int j = 0; j < N / NUM_CHUNKS; j++) {
            h_chunks[i][j] = j * 0.001f;
        }
    }
    
    overlapTransferCompute(h_chunks, d_chunks, N / NUM_CHUNKS, NUM_CHUNKS);
    printf("  ✓ Overlap transfer/compute completed\n");
    
    // Cleanup
    for (int i = 0; i < NUM_CHUNKS; i++) {
        free(h_chunks[i]);
        cudaFree(d_chunks[i]);
    }
    delete[] h_chunks;
    delete[] d_chunks;
    
    free(h_data);
    cudaFree(d_data);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Pinned memory (cudaMallocHost) enables async transfers\n");
    printf("- cudaMemcpyAsync returns immediately, doesn't wait\n");
    printf("- Use streams to overlap transfer with compute\n");
    printf("- Separate streams for H2D and D2H can run concurrently\n");
    printf("- Pinned memory is limited - free when done!\n");
    printf("\nNext: Try level4_stream_callbacks.cu for callbacks\n");
    
    return 0;
}
