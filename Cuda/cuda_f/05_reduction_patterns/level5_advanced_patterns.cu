/*
 * Reduction Level 5: Advanced Reduction Patterns
 *
 * EXERCISE: Master advanced reduction variants used in real applications.
 *
 * CONCEPTS:
 * - Custom reduction operators
 * - ArgMax/ArgMin reduction
 * - Boolean reduction (any/all)
 * - Histogram as reduction
 * - Multi-value reduction
 *
 * SKILLS PRACTICED:
 * - Template metaprogramming
 * - Custom operators
 * - Compound reductions
 * - Index tracking
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#define N 1000000
#define BLOCK_SIZE 256

// ============================================================================
// KERNEL 1: ArgMax Reduction
 * Find the index of the maximum value
 * TODO: Track both value and index through reduction
// ============================================================================
__global__ void argMaxReduction(float *input, int *maxIdx, float *maxVal, int n) {
    __shared__ float sharedVals[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize with current thread's value and global index
    // Handle out-of-bounds with -FLT_MAX
    float localMax = /* YOUR CODE HERE */;
    int localMaxIndex = /* YOUR CODE HERE */;
    
    // Reduce: track both max value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        sharedVals[tid] = localMax;
        sharedIdx[tid] = localMaxIndex;
        __syncthreads();
        
        if (tid < stride) {
            // TODO: Compare with neighbor and keep max
            float otherVal = sharedVals[tid + stride];
            int otherIdx = sharedIdx[tid + stride];
            
            if (/* YOUR CODE HERE - Compare values */) {
                localMax = otherVal;
                localMaxIndex = otherIdx;
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // TODO: Use atomic operations to update global max
        // Need to atomically compare-and-swap
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 2: Min-Max Reduction (Simultaneous)
 * Find both minimum and maximum in single pass
 * TODO: Track both min and max through reduction
// ============================================================================
__global__ void minMaxReduction(float *input, float *minOut, float *maxOut, int n) {
    __shared__ float sharedMin[BLOCK_SIZE];
    __shared__ float sharedMax[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize local min and max
    float localMin = /* YOUR CODE HERE */;
    float localMax = /* YOUR CODE HERE */;
    
    // Reduce both min and max simultaneously
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        sharedMin[tid] = localMin;
        sharedMax[tid] = localMax;
        __syncthreads();
        
        if (tid < stride) {
            // TODO: Update both min and max
            localMin = fminf(localMin, sharedMin[tid + stride]);
            localMax = fmaxf(localMax, sharedMax[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // TODO: Atomically update global min and max
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 3: Boolean Reduction (Any/All)
 * Check if any or all elements satisfy a condition
 * TODO: Implement boolean reduction patterns
// ============================================================================
__global__ void booleanReduction(float *input, int *anyResult, int *allResult, 
                                  int n, float threshold) {
    __shared__ int sharedAny[BLOCK_SIZE];
    __shared__ int sharedAll[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check condition for this thread's element
    bool satisfies = (idx < n) && (input[idx] > threshold);
    
    // TODO: Initialize shared memory for boolean reduction
    // any: 1 if any thread satisfies, 0 otherwise
    // all: 1 if all threads satisfy, 0 otherwise
    sharedAny[tid] = /* YOUR CODE HERE */;
    sharedAll[tid] = /* YOUR CODE HERE */;
    __syncthreads();
    
    // TODO: Reduce using OR for 'any' and AND for 'all'
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedAny[tid] = sharedAny[tid] | sharedAny[tid + stride];
            sharedAll[tid] = sharedAll[tid] & sharedAll[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // TODO: Atomically update global results
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 4: Multi-Value Reduction (Sum + Sum of Squares)
 * Compute multiple statistics in one pass
 * TODO: Track multiple accumulators
// ============================================================================
__global__ void multiValueReduction(float *input, float *sumOut, float *sumSqOut, int n) {
    __shared__ float sharedSum[BLOCK_SIZE];
    __shared__ float sharedSumSq[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize accumulators with grid-stride
    float localSum = 0.0f;
    float localSumSq = 0.0f;
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        localSum += val;
        localSumSq += val * val;
    }
    
    sharedSum[tid] = localSum;
    sharedSumSq[tid] = localSumSq;
    __syncthreads();
    
    // TODO: Reduce both values simultaneously
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
            sharedSumSq[tid] += sharedSumSq[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // TODO: Atomically add to global accumulators
        /* YOUR CODE HERE */
    }
}

// ============================================================================
// KERNEL 5: Template-Based Custom Operator Reduction
 * Use templates for custom reduction operators
 * TODO: Complete the templated reduction framework
// ============================================================================

// Custom operator traits
template<typename T>
struct ReduceOp {
    __device__ static T identity();
    __device__ static T combine(T a, T b);
};

// Sum operator specialization
template<>
struct ReduceOp<float> {
    __device__ static float identity() { return 0.0f; }
    __device__ static float combine(float a, float b) { return a + b; }
};

// Max operator specialization
template<>
struct ReduceOp<float, true> {
    __device__ static float identity() { return -FLT_MAX; }
    __device__ static float combine(float a, float b) { return fmaxf(a, b); }
};

// TODO: Complete the templated reduction kernel
template<typename T, typename Op = ReduceOp<T>>
__global__ void templateReduction(T *input, T *output, int n) {
    __shared__ T sharedData[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize with identity and accumulate
    T localResult = Op::identity();
    
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        localResult = Op::combine(localResult, input[i]);
    }
    
    sharedData[tid] = localResult;
    __syncthreads();
    
    // TODO: Reduce using the operator
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = Op::combine(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// ============================================================================
// KERNEL 6: Histogram as Reduction
 * Build histogram using reduction pattern
 * TODO: Complete histogram reduction
// ============================================================================
__global__ void histogramReduction(float *input, unsigned int *histogram, 
                                    int n, int numBins) {
    __shared__ unsigned int localHist[256];  // Assume 256 bins
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize local histogram to zeros
    for (int i = tid; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // TODO: Each thread processes elements and increments bins
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int bin = (int)input[i];
        if (bin >= 0 && bin < numBins) {
            // TODO: Atomically increment local histogram
            /* YOUR CODE HERE */
        }
    }
    __syncthreads();
    
    // TODO: Add local histogram to global histogram
    for (int i = tid; i < numBins; i += blockDim.x) {
        if (localHist[i] > 0) {
            /* YOUR CODE HERE - atomicAdd to global histogram */
        }
    }
}

// Utility functions
void initArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

void initArrayRandom(float *arr, int n, float maxVal) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % (int)(maxVal * 100)) / 100.0f;
    }
}

int main() {
    printf("=== Reduction Level 5: Advanced Patterns ===\n\n");
    
    const int N = 100000;
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(100 * sizeof(float));
    int *h_idxOut = (int*)malloc(sizeof(int));
    float *h_valOut = (float*)malloc(sizeof(float));
    
    // Test data: values from 0 to N-1
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    float *d_input, *d_output;
    int *d_idxOut;
    float *d_valOut, *d_minOut, *d_maxOut;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, 100 * sizeof(float));
    cudaMalloc(&d_idxOut, sizeof(int));
    cudaMalloc(&d_valOut, sizeof(float));
    cudaMalloc(&d_minOut, sizeof(float));
    cudaMalloc(&d_maxOut, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = BLOCK_SIZE;
    int gridSize = 64;
    
    // Test 1: ArgMax reduction
    printf("Test 1: ArgMax reduction\n");
    printf("  Finding max value and its index in %d elements (0 to %d)...\n", N, N-1);
    
    // TODO: Launch argMaxReduction kernel
    // argMaxReduction<<<gridSize, blockSize>>>(d_input, d_idxOut, d_valOut, N);
    printf("  (Implementation exercise - see kernel comments)\n");
    
    // Test 2: Min-Max reduction
    printf("\nTest 2: Min-Max reduction\n");
    initArrayRandom(h_input, N, 1000.0f);
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    float h_min, h_max;
    // TODO: Launch minMaxReduction kernel
    // minMaxReduction<<<gridSize, blockSize>>>(d_input, d_minOut, d_maxOut, N);
    printf("  (Implementation exercise - see kernel comments)\n");
    
    // Test 3: Boolean reduction
    printf("\nTest 3: Boolean reduction (any/all above threshold)\n");
    const float THRESHOLD = 500.0f;
    int *d_anyResult, *d_allResult;
    cudaMalloc(&d_anyResult, sizeof(int));
    cudaMalloc(&d_allResult, sizeof(int));
    
    // TODO: Launch booleanReduction kernel
    // booleanReduction<<<gridSize, blockSize>>>(d_input, d_anyResult, d_allResult, N, THRESHOLD);
    printf("  (Implementation exercise - see kernel comments)\n");
    
    // Test 4: Multi-value reduction
    printf("\nTest 4: Multi-value reduction (sum + sum of squares)\n");
    initArray(h_input, N, 1.0f);  // All 1s for easy verification
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_sum, *d_sumSq;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sumSq, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemset(d_sumSq, 0, sizeof(float));
    
    // TODO: Launch multiValueReduction kernel
    // multiValueReduction<<<gridSize, blockSize>>>(d_input, d_sum, d_sumSq, N);
    printf("  Expected: sum=%d, sumSq=%d\n", N, N);
    printf("  (Implementation exercise - see kernel comments)\n");
    
    // Test 5: Template reduction
    printf("\nTest 5: Template-based reduction\n");
    // TODO: Launch templateReduction kernel
    // templateReduction<float><<<gridSize, blockSize>>>(d_input, d_output, N);
    printf("  (Implementation exercise - see kernel comments)\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_idxOut);
    free(h_valOut);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_idxOut);
    cudaFree(d_valOut);
    cudaFree(d_minOut);
    cudaFree(d_maxOut);
    cudaFree(d_anyResult);
    cudaFree(d_allResult);
    cudaFree(d_sum);
    cudaFree(d_sumSq);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- ArgMax: track both value and index through reduction\n");
    printf("- Min-Max: compute both in single pass for efficiency\n");
    printf("- Boolean: use OR for 'any', AND for 'all'\n");
    printf("- Multi-value: track multiple accumulators simultaneously\n");
    printf("- Templates: enable custom operators with same reduction logic\n");
    printf("- Histogram: reduction with atomic bin increments\n");
    printf("\n=== Reduction Patterns Module Complete ===\n");
    printf("Next: Explore matrix_multiplication for real-world applications\n");
    
    return 0;
}
