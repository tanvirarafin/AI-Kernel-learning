// ============================================================================
// Exercise 5.1: Occupancy Tuning - Maximize GPU Utilization
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to learn occupancy optimization.
//   Higher occupancy = better latency hiding = better performance!
//   Compile with: nvcc -o ex5.1 01_exercises_occupancy_tuning.cu
//   Run with: ./ex5.1
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
// EXERCISE 1: Analyze Register Usage
// Use compiler flags to see and control register usage
// Compile with: nvcc -ptxas-options=-v 01_exercises_occupancy_tuning.cu
// ============================================================================
__global__ void highRegisterKernel(float *data, int n) {
    // TODO: This kernel uses many registers - try to reduce them!
    register float r0, r1, r2, r3, r4, r5, r6, r7;
    register float r8, r9, r10, r11, r12, r13, r14, r15;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        r0 = data[idx];
        r1 = r0 * 2.0f;
        r2 = r1 + 1.0f;
        r3 = r2 * 3.0f;
        r4 = r3 - 2.0f;
        r5 = r4 * 4.0f;
        r6 = r5 + 3.0f;
        r7 = r6 * 5.0f;
        r8 = r7 - 4.0f;
        r9 = r8 * 6.0f;
        r10 = r9 + 5.0f;
        r11 = r10 * 7.0f;
        r12 = r11 - 6.0f;
        r13 = r12 * 8.0f;
        r14 = r13 + 7.0f;
        r15 = r14 * 9.0f;
        
        data[idx] = r15;
    }
}

// TODO: Rewrite this kernel to use fewer registers
// Hint: Reuse variables instead of creating new ones!
__global__ void optimizedRegisterKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Rewrite using fewer variables
        // Example: float result = data[idx];
        //          result = result * 2.0f + 1.0f;
        //          result = result * 3.0f - 2.0f;
        //          ...
        
        data[idx] = data[idx] * 2.0f;  // Placeholder - implement full calculation
    }
}

// ============================================================================
// EXERCISE 2: Find Optimal Block Size
// Use CUDA Occupancy API to find the best configuration
// ============================================================================
void analyzeOccupancy(void (*kernel)(float*, int), const char *name, int n) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int minGridSize, blockSize;
    
    // TODO: Use cudaOccupancyMaxPotentialBlockSize() to find optimal block size
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    
    printf("=== %s ===\n", name);
    printf("  Device: %s\n", prop.name);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Optimal block size (occupancy API): %d\n", blockSize);
    printf("  Min blocks per SM: %d\n\n", minGridSize);
}

// ============================================================================
// EXERCISE 3: Shared Memory Limited Kernel
// High shared memory usage limits occupancy
// ============================================================================
__global__ void sharedMemLimitedKernel(float *data, int n) {
    // TODO: This uses 4KB of shared memory per block
    // Try reducing to increase occupancy
    __shared__ float sharedData[1024];  // 4KB
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sharedData[threadIdx.x] = data[idx];
        __syncthreads();
        
        data[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

// TODO: Create a version that uses less shared memory
__global__ void optimizedSharedMemKernel(float *data, int n) {
    // TODO: Reduce shared memory usage
    // Hint: Do you really need 1024 floats?
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// ============================================================================
// EXERCISE 4: Measure Occupancy Impact on Performance
// Compare performance at different occupancies
// ============================================================================
__global__ void configurableKernel(float *data, int n, int workAmount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < workAmount; i++) {
            sum += i * 0.001f;
        }
        data[idx] = sum;
    }
}

float timeKernel(void (*kernel)(float*, int, int),
                 float *d_data, int n, int workAmount,
                 int blocks, int threads, int iterations) {
    kernel<<<blocks, threads>>>(d_data, n, workAmount);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_data, n, workAmount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed;
}

// ============================================================================
// EXERCISE 5: Calculate Theoretical Occupancy
// Implement occupancy calculation based on resource limits
// ============================================================================
float calculateOccupancy(int threadsPerBlock, int regsPerThread, 
                         int sharedMemPerBlock, cudaDeviceProp *prop) {
    // TODO: Calculate occupancy based on resource limits
    
    // 1. Calculate max blocks limited by threads
    // int maxBlocksByThreads = prop->maxThreadsPerMultiProcessor / threadsPerBlock;
    
    // 2. Calculate max blocks limited by registers
    // int totalRegsNeeded = threadsPerBlock * regsPerThread;
    // int maxBlocksByRegs = prop->regsPerMultiprocessor / totalRegsNeeded;
    
    // 3. Calculate max blocks limited by shared memory
    // int maxBlocksByShared = prop->sharedMemPerMultiprocessor / sharedMemPerBlock;
    
    // 4. Occupancy is limited by the most constrained resource
    // int activeBlocks = min(maxBlocksByThreads, min(maxBlocksByRegs, maxBlocksByShared));
    
    // 5. Calculate occupancy as ratio
    // float occupancy = (float)activeBlocks / maxPossibleBlocks;
    
    return 1.0f;  // Placeholder
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Occupancy Tuning Exercises ===\n\n");
    
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (SMs): %d\n\n", prop.multiProcessorCount);
    
    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(float);
    
    float *h_data = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f;
    }
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    // ========================================================================
    // TEST 1: Analyze Register Usage
    // ========================================================================
    printf("Exercise 1: Register Usage Analysis\n\n");
    
    analyzeOccupancy(highRegisterKernel, "High Register Kernel", n);
    analyzeOccupancy(optimizedRegisterKernel, "Optimized Register Kernel", n);
    
    // ========================================================================
    // TEST 2: Compare Performance
    // ========================================================================
    printf("Exercise 2: Performance Comparison\n\n");
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int workAmount = 100;
    int iterations = 100;
    
    printf("Configuration: %d blocks, %d threads/block\n", blocksPerGrid, threadsPerBlock);
    printf("Iterations: %d\n\n", iterations);
    
    float highRegTime = timeKernel(highRegisterKernel, d_data, n, workAmount,
                                    blocksPerGrid, threadsPerBlock, iterations);
    printf("High Register Kernel:    %.3f ms\n", highRegTime);
    
    float optRegTime = timeKernel(optimizedRegisterKernel, d_data, n, workAmount,
                                   blocksPerGrid, threadsPerBlock, iterations);
    printf("Optimized Register Kernel: %.3f ms\n", optRegTime);
    
    if (highRegTime > 0) {
        printf("Speedup: %.2fx\n\n", highRegTime / optRegTime);
    }
    
    // ========================================================================
    // TEST 3: Shared Memory Impact
    // ========================================================================
    printf("Exercise 3: Shared Memory Impact on Occupancy\n\n");
    
    analyzeOccupancy(sharedMemLimitedKernel, "4KB Shared Memory Kernel", n);
    analyzeOccupancy(optimizedSharedMemKernel, "Optimized Shared Memory Kernel", n);
    
    // ========================================================================
    // TEST 4: Block Size Sweep
    // ========================================================================
    printf("Exercise 4: Block Size Sweep\n\n");
    
    printf("| Block Size | Time (ms) | Notes                    |\n");
    printf("|------------|-----------|--------------------------|\n");
    
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numSizes = sizeof(blockSizes) / sizeof(int);
    
    for (int i = 0; i < numSizes; i++) {
        int bs = blockSizes[i];
        if (bs > prop.maxThreadsPerBlock) continue;
        
        int bg = (n + bs - 1) / bs;
        
        float time = timeKernel(configurableKernel, d_data, n, workAmount,
                                bg, bs, iterations);
        
        printf("| %4d       | %7.3f   | ", bs, time);
        
        // Calculate theoretical occupancy
        int minGrid, optBlock;
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &optBlock, configurableKernel, 0, 0);
        
        if (bs == optBlock) {
            printf("Optimal!         |\n");
        } else if (bs < optBlock) {
            printf("Under-utilized   |\n");
        } else {
            printf("Over block limit |\n");
        }
    }
    
    printf("\n");
    
    // ========================================================================
    // TEST 5: Compiler Flag Experiment
    // ========================================================================
    printf("Exercise 5: Compiler Flag Experiment\n\n");
    printf("Try compiling with different -maxrregcount values:\n");
    printf("  nvcc -maxrregcount=32 01_exercises_occupancy_tuning.cu\n");
    printf("  nvcc -maxrregcount=64 01_exercises_occupancy_tuning.cu\n");
    printf("  nvcc -maxrregcount=128 01_exercises_occupancy_tuning.cu\n\n");
    printf("Measure performance difference!\n\n");
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    printf("=== Occupancy Optimization Tips ===\n");
    printf("1. Use cudaOccupancyMaxPotentialBlockSize() for optimal config\n");
    printf("2. Reduce register usage with -maxrregcount flag\n");
    printf("3. Minimize shared memory per block\n");
    printf("4. Larger blocks (up to 512-1024) often give better occupancy\n");
    printf("5. Higher occupancy != always faster (depends on workload)\n");
    printf("6. Memory-bound kernels benefit less from high occupancy\n");
    printf("7. Compute-bound kernels benefit more from high occupancy\n");
    printf("\n=== Profiling Tools ===\n");
    printf("1. nvprof --metrics achieved_occupancy ./program\n");
    printf("2. ncu --metrics sm__warps_per_sm.max ./program\n");
    printf("3. Nsight Compute: Look for 'Occupancy Limiting Reason'\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Occupancy API:
//    int minGridSize, blockSize;
//    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
//
// 2. Register reduction:
//    - Reuse variables instead of creating new ones
//    - Use -maxrregcount=N compiler flag
//    - But: too few registers can cause spills to local memory!
//
// 3. Shared memory reduction:
//    - Only allocate what you really need
//    - Consider processing smaller tiles
//
// 4. Occupancy calculation:
//    - Limited by: threads, registers, shared memory
//    - occupancy = active_warps / max_warps
//
// 5. When occupancy matters:
//    - Compute-bound: YES, higher occupancy helps
//    - Memory-bound: MAYBE, depends on latency hiding needs
// ============================================================================
