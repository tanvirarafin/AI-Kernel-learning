/*
 * Occupancy Optimization - Complete Exercise File
 * 
 * This file contains exercises for occupancy optimization.
 * Complete the TODO sections to learn occupancy tuning.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

// Kernel with configurable register usage
__global__ void registerHeavyKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Use many local variables (registers)
        float r0 = input[idx];
        float r1 = r0 * 1.1f;
        float r2 = r1 * 1.2f;
        float r3 = r2 * 1.3f;
        float r4 = r3 * 1.4f;
        float r5 = r4 * 1.5f;
        float r6 = r5 * 1.6f;
        float r7 = r6 * 1.7f;
        
        output[idx] = r7;
    }
}

// Kernel with configurable shared memory
__global__ void sharedMemoryHeavyKernel(float *input, float *output, 
                                         int n, int sharedSize) {
    extern __shared__ float sharedData[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
        __syncthreads();
        
        output[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

// ============================================================================
// FUNCTION 1: Query Occupancy
 * Calculate occupancy for given configuration
 * TODO: Complete the occupancy query
// ============================================================================
void queryOccupancy(int blockSize, int registersPerThread) {
    int minGridSize = 10;
    int blockSize_query = blockSize;
    
    // TODO: Query max potential block size
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_query,
    //     registerHeavyKernel, 0, 0);
    
    /* YOUR CODE HERE */
    
    printf("  Recommended block size: %d\n", blockSize_query);
    
    // TODO: Query max active blocks per SM
    // int numBlocks = 0;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
    //     registerHeavyKernel, blockSize_query, 0);
    
    /* YOUR CODE HERE */
    
    // printf("  Max active blocks per SM: %d\n", numBlocks);
}

// ============================================================================
// FUNCTION 2: Find Optimal Block Size
 * Search for optimal block size
 * TODO: Complete the block size optimization
// ============================================================================
void findOptimalBlockSize() {
    int possibleBlockSizes[] = {32, 64, 128, 256, 512, 1024};
    
    printf("  Block Size | Max Blocks/SM | Occupancy\n");
    printf("  -----------|---------------|----------\n");
    
    for (int i = 0; i < 6; i++) {
        int blockSize = possibleBlockSizes[i];
        
        // TODO: Query occupancy for each block size
        // int numBlocks = 0;
        // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
        //     registerHeavyKernel, blockSize, 0);
        
        // int occupancy = numBlocks * blockSize / 1024 * 100;
        
        /* YOUR CODE HERE */
        
        // printf("  %10d | %13d | %d%%\n", blockSize, numBlocks, occupancy);
    }
}

// ============================================================================
// FUNCTION 3: Shared Memory Impact on Occupancy
 * Test how shared memory affects occupancy
 * TODO: Complete the shared memory occupancy test
// ============================================================================
void sharedMemoryOccupancyTest() {
    int sharedSizes[] = {0, 1024, 4096, 16384, 49152};
    
    printf("  Shared Size | Max Blocks/SM | Occupancy\n");
    printf("  ------------|---------------|----------\n");
    
    int blockSize = 256;
    
    for (int i = 0; i < 5; i++) {
        int sharedSize = sharedSizes[i];
        
        // TODO: Query occupancy with shared memory
        // int numBlocks = 0;
        // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
        //     sharedMemoryHeavyKernel, blockSize, sharedSize);
        
        /* YOUR CODE HERE */
        
        // printf("  %11d | %13d | %d%%\n", sharedSize, numBlocks, occupancy);
    }
}

int main() {
    printf("=== Occupancy Optimization Exercises ===\n\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max registers per thread: %d\n", prop.regsPerBlock);
    printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Warp size: %d\n\n", prop.warpSize);
    
    // Test 1: Query occupancy
    printf("Test 1: Occupancy Query\n");
    queryOccupancy(256, 0);
    printf("\n");
    
    // Test 2: Find optimal block size
    printf("Test 2: Block Size Optimization\n");
    findOptimalBlockSize();
    printf("\n");
    
    // Test 3: Shared memory impact
    printf("Test 3: Shared Memory Impact on Occupancy\n");
    sharedMemoryOccupancyTest();
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Occupancy = Active warps / Maximum warps\n");
    printf("- Higher occupancy doesn't always mean better performance\n");
    printf("- Register usage limits occupancy\n");
    printf("- Shared memory usage limits occupancy\n");
    printf("- Use cudaOccupancyMaxPotentialBlockSize for auto-tuning\n");
    printf("- Sweet spot: often 64-256 threads per block\n");
    
    return 0;
}
