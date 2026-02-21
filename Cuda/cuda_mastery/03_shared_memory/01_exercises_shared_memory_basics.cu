// ============================================================================
// Exercise 3.1: Shared Memory Basics - On-Chip Communication
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to learn shared memory patterns.
//   Shared memory is ~100x faster than global memory!
//   Compile with: nvcc -o ex3.1 01_exercises_shared_memory_basics.cu
//   Run with: ./ex3.1
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
// EXERCISE 1: Basic Shared Memory Usage
// Load data to shared memory, then process with neighbor access
// ============================================================================
__global__ void sharedMemoryNeighbor(const float *input, float *output, int n) {
    // TODO: Declare 256 floats of shared memory
    // Hint: __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // TODO: Load from global memory to shared memory
        
        // TODO: Synchronize - ensure all threads have loaded
        
        // TODO: Each thread computes: output = self + left_neighbor
        // Handle boundary case (tid == 0 has no left neighbor)
    }
}

// ============================================================================
// EXERCISE 2: Block Reduction Using Shared Memory
// Compute the sum of all elements in a block
// Classic parallel reduction pattern
// ============================================================================
__global__ void blockReduction(const float *input, float *blockSums, int n) {
    // TODO: Declare shared memory for reduction
    // Hint: __shared__ float sdata[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // TODO: Load data to shared memory (handle bounds)
    
    // TODO: Synchronize
    
    // TODO: Perform tree-based reduction
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         sdata[tid] += sdata[tid + stride];
    //     }
    //     __syncthreads();
    // }
    
    // TODO: Thread 0 writes block's sum to output
}

// ============================================================================
// EXERCISE 3: Reverse Data Within Block
// Use shared memory to reverse the order of elements within each block
// ============================================================================
__global__ void reverseWithinBlock(const float *input, float *output, int n) {
    // TODO: Declare shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // TODO: Load to shared memory
        
        // TODO: Synchronize
        
        // TODO: Read from reversed position
        // reversed position = (blockDim.x - 1 - tid)
        
        // TODO: Write to output
    }
}

// ============================================================================
// EXERCISE 4: Dynamic Shared Memory
// Size determined at kernel launch time
// ============================================================================
__global__ void dynamicSharedMemory(const float *input, float *output, int n) {
    // TODO: Declare dynamic shared memory
    // Hint: extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // TODO: Load to shared memory
        
        // TODO: Synchronize
        
        // TODO: Process and write to output
        output[idx] = sdata[tid] * 2.0f;
    }
}

// ============================================================================
// EXERCISE 5: Sliding Window Average (Challenge!)
// Each output element is the average of itself and its 2 neighbors
// Handle boundary conditions carefully!
// ============================================================================
__global__ void slidingWindowAverage(const float *input, float *output, int n) {
    // TODO: Declare shared memory with HALO regions
    // Need 2 extra elements per block for left/right neighbors
    // Hint: __shared__ float sdata[256 + 2];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // TODO: Calculate left and right neighbor global indices
    // leftIdx = idx - 1, rightIdx = idx + 1
    
    // TODO: Load data including halo regions
    // Thread 0 loads left halo, last thread loads right halo
    // Handle global bounds!
    
    // TODO: Synchronize
    
    // TODO: Compute average: (left + self + right) / 3.0f
    
    // TODO: Write to output
}

// ============================================================================
// EXERCISE 6: Histogram Using Shared Memory (Challenge!)
// Count occurrences of values (0-255 range) using shared memory
// ============================================================================
__global__ void sharedHistogram(const unsigned char *input, unsigned int *histogram, 
                                int n, int numBins) {
    // TODO: Declare shared memory for bins (256 bins)
    // Hint: __shared__ unsigned int sharedHist[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // TODO: Initialize shared histogram to zero
    // Each thread initializes a few bins
    
    // TODO: Synchronize
    
    // TODO: Each thread processes its input element(s)
    // Use atomicAdd to update shared histogram
    // Consider grid-stride loop for large inputs
    
    // TODO: Synchronize
    
    // TODO: Thread 0 adds shared histogram to global histogram
    // Use atomicAdd for each bin
}

// ============================================================================
// VERIFICATION FUNCTION
// ============================================================================
bool verifyResults(const float *gpu, const float *cpu, int n, float tolerance, 
                   const char *testName) {
    for (int i = 0; i < n; i++) {
        float diff = gpu[i] - cpu[i];
        if (diff < 0) diff = -diff;
        if (diff > tolerance) {
            printf("  [FAIL] %s: Mismatch at %d: GPU=%.4f, CPU=%.4f\n",
                   testName, i, gpu[i], cpu[i]);
            return false;
        }
    }
    printf("  [PASS] %s: All %d elements match!\n", testName, n);
    return true;
}

// ============================================================================
// CPU REFERENCE: Block Reduction
// ============================================================================
float cpuReduction(const float *input, int start, int end) {
    float sum = 0;
    for (int i = start; i < end && i < 256; i++) {
        sum += input[i];
    }
    return sum;
}

// ============================================================================
// CPU REFERENCE: Sliding Window Average
// ============================================================================
void cpuSlidingWindow(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float sum = input[i];
        int count = 1;
        if (i > 0) { sum += input[i-1]; count++; }
        if (i < n-1) { sum += input[i+1]; count++; }
        output[i] = sum / count;
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Shared Memory Exercises ===\n\n");
    
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_expected = (float *)malloc(size);
    
    // Initialize with all 1s for easy verification
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    // Device arrays
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Array size: %d elements\n", n);
    printf("Blocks: %d, Threads per block: %d\n\n", blocksPerGrid, threadsPerBlock);
    
    // ========================================================================
    // TEST 1: Shared Memory Neighbor Access
    // ========================================================================
    printf("Exercise 1: Shared Memory Neighbor Access\n");
    printf("  Each thread: output = self + left_neighbor\n\n");
    
    sharedMemoryNeighbor<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute expected (first block only, thread 0 has no left neighbor)
    for (int i = 0; i < n; i++) {
        int tid = i % 256;
        if (tid == 0) {
            h_expected[i] = 1.0f;  // No left neighbor
        } else {
            h_expected[i] = 2.0f;  // self + left = 1 + 1
        }
    }
    
    verifyResults(h_output, h_expected, 512, 1e-5f, "Neighbor Access");
    printf("  Sample: output[0]=%.0f, output[1]=%.0f, output[256]=%.0f\n\n",
           h_output[0], h_output[1], h_output[256]);
    
    // ========================================================================
    // TEST 2: Block Reduction
    // ========================================================================
    printf("Exercise 2: Block Reduction (Sum within each block)\n");
    
    float *d_blockSums;
    cudaMalloc(&d_blockSums, blocksPerGrid * sizeof(float));
    
    blockReduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_blockSums, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float *h_blockSums = (float *)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, blocksPerGrid * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    printf("  Number of blocks: %d\n", blocksPerGrid);
    printf("  Sum per block (first 5): ");
    for (int i = 0; i < 5 && i < blocksPerGrid; i++) {
        printf("%.0f ", h_blockSums[i]);
    }
    printf("\n");
    
    // Calculate total sum
    float totalSum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        totalSum += h_blockSums[i];
    }
    printf("  Total sum: %.0f (expected: %d)\n", totalSum, n);
    
    // Verify each block sum (should be 256 for full blocks)
    bool reductionCorrect = true;
    for (int i = 0; i < blocksPerGrid - 1; i++) {  // Exclude last block (may be partial)
        if (h_blockSums[i] != 256.0f) {
            reductionCorrect = false;
            break;
        }
    }
    printf("  [%s] Block reduction\n\n", reductionCorrect ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 3: Reverse Within Block
    // ========================================================================
    printf("Exercise 3: Reverse Data Within Block\n");
    
    // Set up pattern: each thread's index
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 256;  // Thread ID within block
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    reverseWithinBlock<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify first block
    printf("  First block (first 5 threads): ");
    for (int i = 0; i < 5; i++) {
        printf("%.0f ", h_output[i]);
    }
    printf("\n  (Expected: 255, 254, 253, 252, 251 - reversed!)\n");
    
    bool reverseCorrect = true;
    for (int i = 0; i < 256; i++) {
        if (h_output[i] != (255 - i)) {
            reverseCorrect = false;
            break;
        }
    }
    printf("  [%s] Reverse within block\n\n", reverseCorrect ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 4: Dynamic Shared Memory
    // ========================================================================
    printf("Exercise 4: Dynamic Shared Memory\n");
    
    // Reset input
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    dynamicSharedMemory<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    bool dynamicCorrect = true;
    for (int i = 0; i < 100; i++) {
        if (h_output[i] != 2.0f) {
            dynamicCorrect = false;
            break;
        }
    }
    printf("  [%s] Dynamic shared memory (all elements = 2.0)\n\n", 
           dynamicCorrect ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 5: Sliding Window Average
    // ========================================================================
    printf("Exercise 5: Sliding Window Average (Challenge!)\n");
    
    // Set up input with known pattern
    for (int i = 0; i < n; i++) {
        h_input[i] = i * 1.0f;  // [0, 1, 2, 3, ...]
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // CPU reference
    cpuSlidingWindow(h_input, h_expected, n);
    
    slidingWindowAverage<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    verifyResults(h_output, h_expected, 1000, 0.01f, "Sliding Window Average");
    printf("  Sample: output[5]=%.2f (expected: %.2f)\n", h_output[5], h_expected[5]);
    printf("  (Middle elements: (i-1 + i + i+1)/3 = i)\n\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    free(h_input);
    free(h_output);
    free(h_expected);
    free(h_blockSums);
    
    printf("=== Shared Memory Summary ===\n");
    printf("| Feature      | Details                                    |\n");
    printf("|--------------|--------------------------------------------|\n");
    printf("| Location     | On-chip (very fast)                        |\n");
    printf("| Scope        | Thread block                               |\n");
    printf("| Size Limit   | ~48KB per SM                               |\n");
    printf("| Latency      | ~100x faster than global memory            |\n");
    printf("| Sync         | __syncthreads() required                   |\n");
    printf("\n=== Common Patterns ===\n");
    printf("1. Data reuse (load once, use many times)\n");
    printf("2. Thread communication (share data within block)\n");
    printf("3. Reduction operations (sum, min, max)\n");
    printf("4. Tiled algorithms (matrix multiply, convolution)\n");
    printf("5. Histogram and other aggregation\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Shared memory declaration:
//    __shared__ float sdata[256];  // Static
//    extern __shared__ float sdata[];  // Dynamic (size at launch)
//
// 2. Synchronization is CRITICAL:
//    __syncthreads();  // All threads must reach this!
//
// 3. Tree reduction pattern:
//    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//        if (tid < stride) sdata[tid] += sdata[tid + stride];
//        __syncthreads();
//    }
//
// 4. Reversing within block:
//    output[idx] = sdata[blockDim.x - 1 - tid];
//
// 5. Never put __syncthreads() in conditionals!
// ============================================================================
