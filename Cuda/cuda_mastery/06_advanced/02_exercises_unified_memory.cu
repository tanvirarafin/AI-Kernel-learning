// ============================================================================
// Exercise 6.2: Unified Memory - Simplified Memory Management
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to learn CUDA Unified Memory.
//   Unified Memory simplifies code with automatic data migration!
//   Compile with: nvcc -o ex6.2 02_exercises_unified_memory.cu
//   Run with: ./ex6.2
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

// ============================================================================
// EXERCISE 1: Basic Unified Memory
// Replace cudaMalloc/cudaMemcpy with cudaMallocManaged
// ============================================================================
__global__ void unifiedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void exercise1_BasicUnifiedMemory() {
    printf("=== Exercise 1: Basic Unified Memory ===\n\n");
    
    int n = 1000000;
    
    // TODO: Allocate unified memory
    // float *data;
    // cudaMallocManaged(&data, n * sizeof(float));
    
    // TODO: Initialize on CPU (no cudaMemcpy needed!)
    // for (int i = 0; i < n; i++) {
    //     data[i] = 1.0f;
    // }
    
    // TODO: Launch kernel (data is accessible on GPU automatically)
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // unifiedKernel<<<blocksPerGrid, threadsPerBlock>>>(data, n);
    
    // TODO: Access on CPU again (automatic migration back)
    // printf("First 5 elements: ");
    // for (int i = 0; i < 5; i++) {
    //     printf("%.1f ", data[i]);
    // }
    // printf("\n (Expected: 3.0 = 1*2+1)\n");
    
    // TODO: Free unified memory
    // cudaFree(data);
}

// ============================================================================
// EXERCISE 2: CPU-GPU Data Sharing
// Same data accessed alternately by CPU and GPU
// ============================================================================
__global__ void gpuProcess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

void cpuProcess(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] + 10.0f;
    }
}

void exercise2_CPUGPUSharing() {
    printf("=== Exercise 2: CPU-GPU Data Sharing ===\n\n");
    
    int n = 10000;
    
    // TODO: Allocate unified memory
    
    // TODO: Initialize on CPU
    
    // TODO: Process on GPU (x2)
    
    // TODO: Process on CPU (+10)
    
    // TODO: Verify: result should be (1 * 2) + 10 = 12
    
    // TODO: Cleanup
}

// ============================================================================
// EXERCISE 3: Memory Prefetching
// Use cudaMemPrefetchAsync to proactively move data
// ============================================================================
__global__ void computeKernel(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += A[idx] * B[idx];
        }
        C[idx] = sum;
    }
}

void exercise3_Prefetching() {
    printf("=== Exercise 3: Memory Prefetching ===\n\n");
    
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // TODO: Allocate unified memory for A, B, C
    
    // TODO: Initialize on CPU
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // WITHOUT prefetch
    printf("Without prefetch:\n");
    
    // TODO: Launch kernel without prefetch and measure time
    
    // WITH prefetch
    printf("With prefetch:\n");
    
    // TODO: Prefetch data to GPU before kernel
    // int deviceId = 0;
    // cudaMemPrefetchAsync(A, size, deviceId);
    // cudaMemPrefetchAsync(B, size, deviceId);
    // cudaMemPrefetchAsync(C, size, deviceId);
    
    // TODO: Launch kernel and measure time
    
    // TODO: Compare times and print speedup
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // TODO: Cleanup
}

// ============================================================================
// EXERCISE 4: Memory Advice
// Use cudaMemAdvise to give hints about access patterns
// ============================================================================
void exercise4_MemoryAdvice() {
    printf("=== Exercise 4: Memory Advice ===\n\n");
    
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, size));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f;
    }
    
    int deviceId = 0;
    
    // TODO: Set ReadMostly advice (data is read frequently, rarely modified)
    // cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);
    
    printf("Memory Advice Options:\n");
    printf("  1. cudaMemAdviseSetReadMostly\n");
    printf("     - Data is read frequently, rarely modified\n");
    printf("     - Keeps copies on both CPU and GPU\n\n");
    
    printf("  2. cudaMemAdviseUnsetReadMostly\n");
    printf("     - Reset to default behavior\n\n");
    
    printf("  3. cudaMemAdviseSetPreferredLocation\n");
    printf("     - Hint where memory should physically reside\n");
    printf("     - Use cudaCpuDeviceId for CPU location\n\n");
    
    printf("  4. cudaMemAdviseSetAccessedBy\n");
    printf("     - Hint which processor will access most\n\n");
    
    // TODO: Launch kernel and verify
    
    // TODO: Cleanup
    cudaFree(data);
}

// ============================================================================
// EXERCISE 5: Irregular Access Pattern (Challenge!)
// Unified Memory excels with unpredictable access patterns
// ============================================================================
__global__ void irregularAccessKernel(float *data, int *indices, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Indirect access - hard to predict!
        int accessIdx = indices[idx] % n;
        output[idx] = data[accessIdx] * 2.0f;
    }
}

void exercise5_IrregularAccess() {
    printf("=== Exercise 5: Irregular Access Pattern (Challenge!) ===\n\n");
    
    int n = 100000;
    
    // TODO: Allocate unified memory for data, indices, output
    
    // TODO: Initialize data array
    
    // TODO: Initialize indices with random values (0 to n-1)
    
    // TODO: Launch kernel with irregular access pattern
    
    // TODO: Verify results
    
    // TODO: Compare with explicit cudaMemcpy approach
    // (Unified Memory should be easier to code, possibly faster)
    
    // TODO: Cleanup
}

// ============================================================================
// EXERCISE 6: Compare UM vs Explicit Memory (Benchmark)
// ============================================================================
void exercise6_PerformanceComparison() {
    printf("=== Exercise 6: UM vs Explicit Memory Comparison ===\n\n");
    
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    printf("Benchmark: Vector addition (1M elements, 100 iterations)\n\n");
    
    // UNIFIED MEMORY VERSION
    printf("Unified Memory:\n");
    
    float *um_A, *um_B, *um_C;
    CUDA_CHECK(cudaMallocManaged(&um_A, size));
    CUDA_CHECK(cudaMallocManaged(&um_B, size));
    CUDA_CHECK(cudaMallocManaged(&um_C, size));
    
    for (int i = 0; i < n; i++) {
        um_A[i] = 1.0f;
        um_B[i] = 2.0f;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(um_A, um_B, um_C, n);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float umTime;
    cudaEventElapsedTime(&umTime, start, stop);
    printf("  Time: %.3f ms\n\n", umTime);
    
    // EXPLICIT MEMORY VERSION
    printf("Explicit Memory (cudaMalloc + cudaMemcpy):\n");
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float explicitTime;
    cudaEventElapsedTime(&explicitTime, start, stop);
    printf("  Time: %.3f ms\n\n", explicitTime);
    
    printf("Comparison:\n");
    printf("  Unified Memory is %.1fx %s\n",
           umTime / explicitTime,
           umTime > explicitTime ? "slower" : "faster");
    printf("  (UM has overhead but simpler code)\n\n");
    
    // Cleanup
    cudaFree(um_A);
    cudaFree(um_B);
    cudaFree(um_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Helper kernel
__global__ void vectorAddKernel(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Unified Memory Exercises ===\n\n");
    
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Managed Memory Support: %s\n\n",
           prop.managedMemory ? "Yes" : "No");
    
    if (!prop.managedMemory) {
        printf("WARNING: Your GPU doesn't support Unified Memory!\n");
        printf("         Some exercises may not work.\n\n");
    }
    
    exercise1_BasicUnifiedMemory();
    exercise2_CPUGPUSharing();
    exercise3_Prefetching();
    exercise4_MemoryAdvice();
    exercise5_IrregularAccess();
    exercise6_PerformanceComparison();
    
    printf("=== Unified Memory Summary ===\n");
    printf("| Feature           | Traditional CUDA    | Unified Memory      |\n");
    printf("|-------------------|---------------------|---------------------|\n");
    printf("| Allocation        | cudaMalloc          | cudaMallocManaged   |\n");
    printf("| Data transfer     | cudaMemcpy          | Automatic           |\n");
    printf("| CPU access        | Requires copy back  | Direct access       |\n");
    printf("| Optimization      | Manual              | Prefetch/Advice     |\n");
    printf("| Code complexity   | Higher              | Lower               |\n");
    printf("\n=== When to Use Unified Memory ===\n");
    printf("GOOD FIT:\n");
    printf("  - Irregular/unpredictable access patterns\n");
    printf("  - Rapid prototyping\n");
    printf("  - CPU-GPU shared data structures\n");
    printf("  - Oversized datasets (automatic paging)\n\n");
    printf("NOT IDEAL:\n");
    printf("  - Performance-critical code\n");
    printf("  - Predictable, regular access patterns\n");
    printf("  - When you need fine-grained control\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Unified memory allocation:
//    float *data;
//    cudaMallocManaged(&data, size);
//
// 2. No cudaMemcpy needed!
//    // Access on CPU
//    data[i] = value;
//    // Access on GPU
//    kernel<<<>>>(data, ...);
//    // Access on CPU again
//    printf("%f", data[i]);
//
// 3. Prefetching:
//    cudaMemPrefetchAsync(ptr, size, deviceId);
//
// 4. Memory advice:
//    cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, deviceId);
//
// 5. Free unified memory:
//    cudaFree(data);  // Same as regular memory
// ============================================================================
