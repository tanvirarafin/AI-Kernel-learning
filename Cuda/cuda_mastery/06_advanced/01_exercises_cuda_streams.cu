// ============================================================================
// Exercise 6.1: CUDA Streams - Concurrent Execution
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to learn CUDA streams and concurrency.
//   Streams enable overlapping execution for better performance!
//   Compile with: nvcc -o ex6.1 01_exercises_cuda_streams.cu
//   Run with: ./ex6.1
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernel for stream exercises
__global__ void vectorKernel(float *A, float *B, float *C, int n, int workAmount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < workAmount; i++) {
            sum += A[idx] * B[idx];
        }
        C[idx] = sum;
    }
}

// ============================================================================
// EXERCISE 1: Create and Use Streams
// Create 4 streams and launch kernels concurrently
// ============================================================================
void exercise1_Streams() {
    printf("=== Exercise 1: Create and Use Streams ===\n\n");
    
    int n = 1 << 18;  // 256K elements per stream
    size_t size = n * sizeof(float);
    const int NUM_STREAMS = 4;
    
    // TODO: Declare array of streams
    // cudaStream_t streams[NUM_STREAMS];
    
    // TODO: Create streams
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaStreamCreate(&streams[i]);
    // }
    
    // Host arrays
    float *h_A = (float *)malloc(size * NUM_STREAMS);
    float *h_B = (float *)malloc(size * NUM_STREAMS);
    float *h_C = (float *)malloc(size * NUM_STREAMS);
    
    for (int i = 0; i < n * NUM_STREAMS; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size * NUM_STREAMS));
    CUDA_CHECK(cudaMalloc(&d_B, size * NUM_STREAMS));
    CUDA_CHECK(cudaMalloc(&d_C, size * NUM_STREAMS));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size * NUM_STREAMS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size * NUM_STREAMS, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerStream = (n + threadsPerBlock - 1) / threadsPerBlock;
    int workAmount = 500;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // SEQUENTIAL EXECUTION (for comparison)
    // ========================================================================
    printf("Sequential Execution (4 kernels one after another):\n");
    
    cudaEventRecord(start);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * n;
        vectorKernel<<<blocksPerStream, threadsPerBlock>>>(
            d_A + offset, d_B + offset, d_C + offset, n, workAmount);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sequentialTime;
    cudaEventElapsedTime(&sequentialTime, start, stop);
    printf("  Time: %.3f ms\n\n", sequentialTime);
    
    // ========================================================================
    // CONCURRENT EXECUTION WITH STREAMS
    // ========================================================================
    printf("Concurrent Execution (4 streams):\n");
    
    // TODO: Launch kernels in different streams
    // cudaEventRecord(start);
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     int offset = i * n;
    //     vectorKernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(
    //         d_A + offset, d_B + offset, d_C + offset, n, workAmount);
    // }
    
    // TODO: Synchronize all streams
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaStreamSynchronize(streams[i]);
    // }
    
    // TODO: Record and measure time
    
    // printf("  Time: %.3f ms\n", concurrentTime);
    // printf("  Speedup: %.2fx\n\n", sequentialTime / concurrentTime);
    
    // TODO: Destroy streams
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaStreamDestroy(streams[i]);
    // }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

// ============================================================================
// EXERCISE 2: Async Memory Transfers
// Overlap data transfers with kernel execution using streams
// ============================================================================
void exercise2_AsyncTransfers() {
    printf("=== Exercise 2: Async Memory Transfers ===\n\n");
    
    int n = 1 << 18;
    size_t size = n * sizeof(float);
    const int NUM_STREAMS = 4;
    
    // TODO: Create streams
    
    // TODO: Allocate PINNED (page-locked) host memory
    // Pinned memory is required for async transfers!
    // float *h_A_pinned, *h_C_pinned;
    // cudaMallocHost(&h_A_pinned, size * NUM_STREAMS);
    // cudaMallocHost(&h_C_pinned, size * NUM_STREAMS);
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size * NUM_STREAMS));
    CUDA_CHECK(cudaMalloc(&d_B, size * NUM_STREAMS));
    CUDA_CHECK(cudaMalloc(&d_C, size * NUM_STREAMS));
    
    int threadsPerBlock = 256;
    int blocksPerStream = (n + threadsPerBlock - 1) / threadsPerBlock;
    int workAmount = 500;
    
    printf("Async Transfers with Kernel Execution:\n");
    printf("  (Requires pinned host memory)\n\n");
    
    // TODO: Implement overlapping transfer and compute
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     int offset = i * n;
    //     
    //     // Async H2D transfer in stream
    //     cudaMemcpyAsync(h_A_pinned + offset, d_A + offset, size,
    //                     cudaMemcpyDeviceToHost, streams[i]);
    //     
    //     // Kernel in same stream
    //     vectorKernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(
    //         d_A + offset, d_B + offset, d_C + offset, n, workAmount);
    // }
    
    // TODO: Synchronize and measure
    
    // TODO: Free pinned memory
    // cudaFreeHost(h_A_pinned);
    // cudaFreeHost(h_C_pinned);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// EXERCISE 3: Stream Dependencies with Events
// Use events to create dependencies between streams
// ============================================================================
void exercise3_StreamDependencies() {
    printf("=== Exercise 3: Stream Dependencies with Events ===\n\n");
    
    int n = 1 << 18;
    size_t size = n * sizeof(float);
    
    // TODO: Create two streams
    // cudaStream_t stream1, stream2;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));
    
    // TODO: Create event
    // cudaEvent_t event;
    // cudaEventCreate(&event);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Producer-Consumer Pattern:\n");
    printf("  Stream 1: Kernel A -> Event\n");
    printf("  Stream 2: Wait for Event -> Kernel B\n\n");
    
    // TODO: Implement dependency
    // 1. Launch Kernel A in stream1
    // 2. Record event in stream1
    // 3. Make stream2 wait for event
    // 4. Launch Kernel B in stream2
    
    // Example:
    // vectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, n, 100);
    // cudaEventRecord(event, stream1);
    // cudaStreamWaitEvent(stream2, event, 0);
    // vectorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_C, d_B, d_D, n, 100);
    
    // TODO: Synchronize and cleanup
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
}

// ============================================================================
// EXERCISE 4: Stream Priorities
// Use stream priorities for QoS
// ============================================================================
void exercise4_StreamPriorities() {
    printf("=== Exercise 4: Stream Priorities ===\n\n");
    
    int leastPriority, greatestPriority;
    
    // TODO: Query priority range
    // cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    printf("Stream Priority Range:\n");
    printf("  Least priority:    %d\n", leastPriority);
    printf("  Greatest priority: %d\n", greatestPriority);
    printf("  (Lower number = higher priority)\n\n");
    
    // TODO: Create high and low priority streams
    // cudaStream_t highPriority, lowPriority;
    // cudaStreamCreateWithPriority(&highPriority, cudaStreamDefault, greatestPriority);
    // cudaStreamCreateWithPriority(&lowPriority, cudaStreamDefault, leastPriority);
    
    printf("Use case: Critical tasks in high-priority streams\n");
    printf("          Background tasks in low-priority streams\n\n");
    
    // TODO: Cleanup
}

// ============================================================================
// EXERCISE 5: Multi-Stream Matrix Multiplication (Challenge!)
// Split a large matrix multiply across multiple streams
// ============================================================================
void exercise5_MatrixMultiply() {
    printf("=== Exercise 5: Multi-Stream Matrix Multiply (Challenge!) ===\n\n");
    
    int width = 2048;
    size_t size = width * width * sizeof(float);
    const int NUM_STREAMS = 4;
    int rowsPerStream = width / NUM_STREAMS;
    
    printf("Matrix size: %dx%d\n", width, width);
    printf("Streams: %d (%d rows per stream)\n\n", NUM_STREAMS, rowsPerStream);
    
    // TODO: Allocate and initialize matrices A, B, C
    // float *h_A = (float *)malloc(size);
    // float *h_B = (float *)malloc(size);
    // float *h_C = (float *)malloc(size);
    
    // TODO: Create streams
    
    // TODO: Allocate device memory
    
    // TODO: Copy to device
    
    // TODO: Launch matrix multiply kernel in each stream
    // Each stream processes rows [streamId * rowsPerStream : (streamId+1) * rowsPerStream]
    
    // TODO: Synchronize and verify
    
    // TODO: Cleanup
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== CUDA Streams Exercises ===\n\n");
    
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count: %d\n\n", prop.asyncEngineCount);
    
    if (!prop.concurrentKernels) {
        printf("WARNING: Your GPU may not support concurrent kernel execution!\n");
        printf("         Stream exercises will still work, but without concurrency.\n\n");
    }
    
    exercise1_Streams();
    exercise2_AsyncTransfers();
    exercise3_StreamDependencies();
    exercise4_StreamPriorities();
    exercise5_MatrixMultiply();
    
    printf("=== Streams Summary ===\n");
    printf("| Feature            | Default Stream | Non-Default Stream |\n");
    printf("|--------------------|----------------|--------------------|\n");
    printf("| Synchronization    | Implicit       | Explicit           |\n");
    printf("| Concurrent kernels | No             | Yes                |\n");
    printf("| Async memcpy       | No             | Yes (with pinned)  |\n");
    printf("| Priority           | Normal         | Configurable       |\n");
    printf("\n=== Best Practices ===\n");
    printf("1. Use cudaStreamCreateWithFlags(cudaStreamNonBlocking)\n");
    printf("2. Pinned memory required for async cudaMemcpyAsync\n");
    printf("3. Use cudaStreamWaitEvent() for inter-stream dependencies\n");
    printf("4. Profile with Nsight Systems to visualize concurrency\n");
    printf("5. More streams != always better (resource contention)\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Stream creation:
//    cudaStream_t stream;
//    cudaStreamCreate(&stream);
//    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//
// 2. Kernel launch in stream:
//    kernel<<<grid, block, sharedMem, stream>>>(args);
//
// 3. Async memory transfer:
//    cudaMemcpyAsync(dst, src, size, kind, stream);
//    // Requires pinned (page-locked) host memory!
//
// 4. Stream synchronization:
//    cudaStreamSynchronize(stream);  // Wait for specific stream
//    cudaDeviceSynchronize();         // Wait for all streams
//
// 5. Events for dependencies:
//    cudaEventRecord(event, stream1);
//    cudaStreamWaitEvent(stream2, event, 0);
//
// 6. Stream destruction:
//    cudaStreamDestroy(stream);
// ============================================================================
