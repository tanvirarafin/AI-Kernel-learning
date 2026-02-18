/*
 * CUDA Streams Level 2: Concurrent Kernel Execution
 *
 * EXERCISE: Execute multiple kernels concurrently using streams.
 *
 * CONCEPTS:
 * - Kernel concurrency
 * - GPU resource sharing
 * - Concurrent execution requirements
 * - Performance benefits
 *
 * SKILLS PRACTICED:
 * - Multi-stream kernel launches
 * - Concurrent execution verification
 * - Workload partitioning
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define NUM_STREAMS 4

// Kernel with configurable workload
__global__ void workloadKernel(float *input, float *output, int n, 
                                int iterations, int streamId) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
        }
        output[idx] = val + streamId * 0.1f;
    }
}

// Simple kernel for concurrent execution
__global__ void simpleKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// ============================================================================
// FUNCTION 1: Launch Kernels in Multiple Streams
 * Launch independent kernels concurrently
 * TODO: Complete the concurrent launch
// ============================================================================
void concurrentKernels(float **d_inputs, float **d_outputs, int n, int numStreams) {
    cudaStream_t *streams = new cudaStream_t[numStreams];
    
    // TODO: Create streams
    for (int i = 0; i < numStreams; i++) {
        // cudaStreamCreate(&streams[i]);
        /* YOUR CODE HERE */
    }
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // TODO: Launch kernels concurrently in different streams
    // Each stream processes different data
    for (int i = 0; i < numStreams; i++) {
        // workloadKernel<<<gridSize, blockSize, 0, streams[i]>>>(
        //     d_inputs[i], d_outputs[i], n, 100, i);
        /* YOUR CODE HERE */
    }
    
    // TODO: Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        // cudaStreamSynchronize(streams[i]);
        /* YOUR CODE HERE */
    }
    
    // TODO: Destroy streams
    for (int i = 0; i < numStreams; i++) {
        // cudaStreamDestroy(streams[i]);
        /* YOUR CODE HERE */
    }
    
    delete[] streams;
}

// ============================================================================
// FUNCTION 2: Partitioned Workload
 * Split large workload across multiple streams
 * TODO: Complete the partitioned execution
// ============================================================================
void partitionedWorkload(float *d_input, float *d_output, int n, int numStreams) {
    cudaStream_t *streams = new cudaStream_t[numStreams];
    int chunkSize = n / numStreams;
    
    // TODO: Create streams
    /* YOUR CODE HERE */
    
    int blockSize = 256;
    
    // TODO: Launch kernel for each chunk in separate stream
    for (int i = 0; i < numStreams; i++) {
        int chunkStart = i * chunkSize;
        int currentChunkSize = (i == numStreams - 1) ? (n - chunkStart) : chunkSize;
        int gridSize = (currentChunkSize + blockSize - 1) / blockSize;
        
        // simpleKernel<<<gridSize, blockSize, 0, streams[i]>>>(
        //     d_input + chunkStart, d_output + chunkStart, currentChunkSize);
        /* YOUR CODE HERE */
    }
    
    // TODO: Synchronize all streams
    /* YOUR CODE HERE */
    
    // TODO: Destroy streams
    /* YOUR CODE HERE */
    
    delete[] streams;
}

// ============================================================================
// FUNCTION 3: Mixed Workload (Different Kernel Types)
 * Launch different kernel types concurrently
 * TODO: Complete the mixed workload execution
// ============================================================================
void mixedWorkload(float *d_a, float *d_b, float *d_c, float *d_d,
                   int n1, int n2, int n3) {
    cudaStream_t stream1, stream2, stream3;
    
    // TODO: Create three streams
    /* YOUR CODE HERE */
    
    int blockSize = 256;
    
    // TODO: Launch different kernels in different streams
    // Stream 1: Vector add
    // int gridSize1 = (n1 + blockSize - 1) / blockSize;
    // vectorAdd<<<gridSize1, blockSize, 0, stream1>>>(d_a, d_b, d_c, n1);
    
    // Stream 2: Simple kernel
    /* YOUR CODE HERE */
    
    // Stream 3: Workload kernel
    /* YOUR CODE HERE */
    
    // TODO: Synchronize all streams
    /* YOUR CODE HERE */
    
    // TODO: Destroy streams
    /* YOUR CODE HERE */
}

// Simple vector add for mixed workload
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("=== CUDA Streams Level 2: Concurrent Kernels ===\n\n");
    
    const int N = 1000000;
    size_t size = N * sizeof(float);
    
    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("Multiprocessor count: %d\n\n", prop.multiProcessorCount);
    
    if (!prop.concurrentKernels) {
        printf("WARNING: This GPU may not support concurrent kernel execution!\n\n");
    }
    
    // Allocate arrays for concurrent kernels
    float **h_inputs, **h_outputs;
    float **d_inputs, **d_outputs;
    
    h_inputs = new float*[NUM_STREAMS];
    h_outputs = new float*[NUM_STREAMS];
    d_inputs = new float*[NUM_STREAMS];
    d_outputs = new float*[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        h_inputs[i] = (float*)malloc(size / NUM_STREAMS);
        h_outputs[i] = (float*)malloc(size / NUM_STREAMS);
        cudaMalloc(&d_inputs[i], size / NUM_STREAMS);
        cudaMalloc(&d_outputs[i], size / NUM_STREAMS);
        
        // Initialize
        for (int j = 0; j < N / NUM_STREAMS; j++) {
            h_inputs[i][j] = j * 0.001f + i * 0.1f;
        }
    }
    
    // Copy to device
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpy(d_inputs[i], h_inputs[i], size / NUM_STREAMS, 
                   cudaMemcpyHostToDevice);
    }
    
    // Test 1: Concurrent kernels
    printf("Test 1: Concurrent kernel execution\n");
    printf("  Launching %d kernels in parallel...\n", NUM_STREAMS);
    
    concurrentKernels(d_inputs, d_outputs, N / NUM_STREAMS, NUM_STREAMS);
    
    printf("  ✓ Concurrent kernels completed\n");
    
    // Verify first stream
    cudaMemcpy(h_outputs[0], d_outputs[0], size / NUM_STREAMS, cudaMemcpyDeviceToHost);
    printf("  Sample output[0] = %.4f\n", h_outputs[0][0]);
    
    // Test 2: Partitioned workload
    printf("\nTest 2: Partitioned workload across streams\n");
    float *d_partitionIn, *d_partitionOut;
    cudaMalloc(&d_partitionIn, size);
    cudaMalloc(&d_partitionOut, size);
    
    float *h_partitionIn = (float*)malloc(size);
    float *h_partitionOut = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_partitionIn[i] = i * 0.001f;
    }
    cudaMemcpy(d_partitionIn, h_partitionIn, size, cudaMemcpyHostToDevice);
    
    partitionedWorkload(d_partitionIn, d_partitionOut, N, NUM_STREAMS);
    
    cudaMemcpy(h_partitionOut, d_partitionOut, size, cudaMemcpyDeviceToHost);
    printf("  ✓ Partitioned workload completed\n");
    printf("  Sample output[0] = %.4f\n", h_partitionOut[0]);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        free(h_inputs[i]);
        free(h_outputs[i]);
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
    }
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] d_inputs;
    delete[] d_outputs;
    
    free(h_partitionIn);
    free(h_partitionOut);
    cudaFree(d_partitionIn);
    cudaFree(d_partitionOut);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Multiple streams enable concurrent kernel execution\n");
    printf("- Partition workloads across streams for better utilization\n");
    printf("- Different kernel types can run concurrently\n");
    printf("- GPU resources (SMs, memory bandwidth) are shared\n");
    printf("- Concurrent execution requires compute capability 3.0+\n");
    printf("\nNext: Try level3_async_memcpy.cu for async transfers\n");
    
    return 0;
}
