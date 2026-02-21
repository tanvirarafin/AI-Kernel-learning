// ============================================================================
// Exercise 4.1: Synchronization and Atomics - Thread Coordination
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to learn synchronization and atomic operations.
//   These are essential for correct parallel programs!
//   Compile with: nvcc -o ex4.1 01_exercises_sync_atomics.cu
//   Run with: ./ex4.1
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// EXERCISE 1: Correct Synchronization Pattern
// Fix the WRONG example by ensuring all threads reach __syncthreads()
// ============================================================================
__global__ void correctSync(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: All threads must participate in synchronization
    // Move __syncthreads() OUTSIDE the conditional!
    
    if (idx < n) {
        // First operation
        data[idx] = 1.0f;
    }
    
    // TODO: Add __syncthreads() here (ALL threads execute this)
    
    if (idx < n) {
        // Second operation - safe to read neighbor's data now
        int neighborIdx = idx + 1;
        if (neighborIdx < n) {
            data[idx] += data[neighborIdx];
        }
    }
}

// ============================================================================
// EXERCISE 2: Atomic Addition for Global Sum
// Implement thread-safe summation using atomicAdd
// ============================================================================
__global__ void atomicSumKernel(const float *input, float *sum, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Atomically add input[idx] to *sum
    // Use atomicAdd(&sum, value)
}

// ============================================================================
// EXERCISE 3: Atomic Min/Max for Reduction
// Find the minimum and maximum values in an array atomically
// ============================================================================
__global__ void atomicMinMaxKernel(const float *input, float *minVal, float *maxVal, 
                                   int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    float val = input[idx];
    
    // TODO: Atomically update minimum
    // Note: atomicMin works on integers, so use __float2int_rn()
    // You'll need a loop with atomicCAS for floats, or use int representation
    
    // TODO: Atomically update maximum
}

// ============================================================================
// EXERCISE 4: Simple Spinlock Using atomicCAS
// Implement a basic lock using Compare-And-Swap
// ============================================================================
__global__ void spinlockKernel(int *lock, int *counter, int n) {
    // TODO: Calculate global thread index
    
    if (idx < n) {
        // TODO: Acquire lock using atomicCAS
        // while (atomicCAS(lock, 0, 1) != 0) { /* spin */ }
        // 0 = unlocked, 1 = locked
        
        // Critical section
        (*counter)++;
        
        // TODO: Release lock
        // atomicExch(lock, 0) or *lock = 0
    }
}

// ============================================================================
// EXERCISE 5: Warp Shuffle for Fast Reduction
// Use warp shuffle operations for intra-warp communication
// Much faster than shared memory!
// ============================================================================
__global__ void warpShuffleReduce(const float *input, float *blockResults, int n) {
    __shared__ float sharedData[1024];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int laneId = threadIdx.x % 32;  // Lane within warp
    
    // Load data to shared memory
    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Shared memory reduction to 32 elements
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    // TODO: Warp-level reduction using shuffle
    // Only threads 0-32 participate
    if (tid < 32) {
        float val = sharedData[tid];
        
        // TODO: Add shuffle down operations
        // val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        // val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        // val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        // val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        // val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        
        // TODO: Lane 0 writes result
    }
}

// ============================================================================
// EXERCISE 6: Parallel Prefix Sum (Scan) - Challenge!
// Compute inclusive scan: output[i] = sum(input[0]...input[i])
// Use Koggin-Stone algorithm
// ============================================================================
__global__ void parallelScan(float *input, float *output, int n) {
    __shared__ float sdata[1024];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // TODO: Load data to shared memory
    
    // TODO: Synchronize
    
    // TODO: Koggin-Stone scan algorithm
    // for (int stride = 1; stride < blockDim.x; stride *= 2) {
    //     float temp = sdata[tid];
    //     if (tid >= stride) {
    //         sdata[tid] += sdata[tid - stride];
    //     }
    //     __syncthreads();
    // }
    
    // TODO: Write result to output
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
// CPU REFERENCE: Scan
// ============================================================================
void cpuScan(const float *input, float *output, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Synchronization and Atomics Exercises ===\n\n");
    
    srand(time(NULL));
    
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_expected = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All 1s for easy sum verification
    }
    
    // Device arrays
    float *d_input, *d_output, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Array size: %d elements\n", n);
    printf("Blocks: %d, Threads per block: %d\n\n", blocksPerGrid, threadsPerBlock);
    
    // ========================================================================
    // TEST 1: Correct Synchronization
    // ========================================================================
    printf("Exercise 1: Correct Synchronization Pattern\n");
    
    CUDA_CHECK(cudaMemcpy(d_output, d_input, size, cudaMemcpyHostToDevice));
    
    correctSync<<<blocksPerGrid, threadsPerBlock>>>(d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    printf("  First 5 elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n  (Each = 1 + right_neighbor, boundary = 1)\n\n");
    
    // ========================================================================
    // TEST 2: Atomic Sum
    // ========================================================================
    printf("Exercise 2: Atomic Sum\n");
    
    float h_sum = 0;
    CUDA_CHECK(cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice));
    
    atomicSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_sum, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("  Atomic sum: %.0f (expected: %d)\n", h_sum, n);
    printf("  [%s] Atomic addition\n\n", (h_sum == n) ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 3: Spinlock
    // ========================================================================
    printf("Exercise 3: Spinlock Counter\n");
    
    int *d_lock, *d_counter;
    CUDA_CHECK(cudaMalloc(&d_lock, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_lock, &h_sum, sizeof(int), cudaMemcpyHostToDevice));  // 0
    CUDA_CHECK(cudaMemcpy(d_counter, &h_sum, sizeof(int), cudaMemcpyHostToDevice));  // 0
    
    spinlockKernel<<<blocksPerGrid, threadsPerBlock>>>(d_lock, d_counter, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("  Counter value: %d (expected: %d)\n", h_counter, n);
    printf("  [%s] Spinlock (note: may be slow with many threads)\n\n",
           (h_counter == n) ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 4: Warp Shuffle Reduction
    // ========================================================================
    printf("Exercise 4: Warp Shuffle Reduction\n");
    
    float *d_blockResults;
    CUDA_CHECK(cudaMalloc(&d_blockResults, blocksPerGrid * sizeof(float)));
    
    warpShuffleReduce<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_blockResults, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float *h_blockResults = (float *)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_blockResults, d_blockResults, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    float warpSum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        warpSum += h_blockResults[i];
    }
    
    printf("  Warp shuffle sum: %.0f (expected: %d)\n", warpSum, n);
    printf("  [%s] Warp shuffle reduction\n\n", (warpSum == n) ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 5: Parallel Scan (Challenge!)
    // ========================================================================
    printf("Exercise 5: Parallel Prefix Sum (Scan) - Challenge!\n");
    
    // Small array for scan (single block)
    int scanN = 256;
    size_t scanSize = scanN * sizeof(float);
    
    float *h_scanInput = (float *)malloc(scanSize);
    float *h_scanOutput = (float *)malloc(scanSize);
    float *h_scanExpected = (float *)malloc(scanSize);
    
    for (int i = 0; i < scanN; i++) {
        h_scanInput[i] = 1.0f;
    }
    
    float *d_scanInput, *d_scanOutput;
    CUDA_CHECK(cudaMalloc(&d_scanInput, scanSize));
    CUDA_CHECK(cudaMalloc(&d_scanOutput, scanSize));
    
    CUDA_CHECK(cudaMemcpy(d_scanInput, h_scanInput, scanSize, cudaMemcpyHostToDevice));
    
    // CPU reference
    cpuScan(h_scanInput, h_scanExpected, scanN);
    
    parallelScan<<<1, scanN>>>(d_scanInput, d_scanOutput, scanN);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_scanOutput, d_scanOutput, scanSize, cudaMemcpyDeviceToHost));
    
    verifyResults(h_scanOutput, h_scanExpected, scanN, 0.01f, "Parallel Scan");
    printf("  First 10 scan values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_scanOutput[i]);
    }
    printf("\n  (Expected: 1, 2, 3, 4, 5, ...)\n\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sum);
    cudaFree(d_lock);
    cudaFree(d_counter);
    cudaFree(d_blockResults);
    cudaFree(d_scanInput);
    cudaFree(d_scanOutput);
    free(h_input);
    free(h_output);
    free(h_expected);
    free(h_blockResults);
    free(h_scanInput);
    free(h_scanOutput);
    free(h_scanExpected);
    
    printf("=== Atomic Operations Summary ===\n");
    printf("| Operation     | Latency     | Use Case                    |\n");
    printf("|---------------|-------------|------------------------------|\n");
    printf("| atomicAdd     | ~500 cycles | Counters, reductions        |\n");
    printf("| atomicCAS     | ~500 cycles | Locks, complex atomics      |\n");
    printf("| atomicExch    | ~500 cycles | Lock release                |\n");
    printf("| atomicMin/Max | ~500 cycles | Min/max reduction           |\n");
    printf("| __shfl_sync   | ~10 cycles  | Intra-warp communication    |\n");
    printf("\n=== Synchronization Rules ===\n");
    printf("1. __syncthreads() must be reached by ALL threads in block\n");
    printf("2. Never put __syncthreads() inside conditionals\n");
    printf("3. Atomics work across blocks, shared memory does not\n");
    printf("4. Warp shuffles are fastest but only work within a warp\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Correct sync pattern:
//    if (condition) { data[idx] = x; }
//    __syncthreads();  // OUTSIDE conditional!
//    if (condition) { data[idx] += data[idx+1]; }
//
// 2. Atomic addition:
//    atomicAdd(&sum, value);
//
// 3. Spinlock pattern:
//    while (atomicCAS(&lock, 0, 1) != 0) { /* wait */ }
//    // critical section
//    atomicExch(&lock, 0);
//
// 4. Warp shuffle:
//    float val = __shfl_down_sync(0xFFFFFFFF, val, offset);
//    // 0xFFFFFFFF = mask for all active threads
//
// 5. Scan algorithm:
//    for (int stride = 1; stride < n; stride *= 2) {
//        if (tid >= stride) sdata[tid] += sdata[tid - stride];
//        __syncthreads();
//    }
// ============================================================================
