/*
 * Shared Memory Level 4: Advanced Patterns
 *
 * EXERCISE: Master complex shared memory algorithms used in real applications.
 *
 * CONCEPTS:
 * - Multi-stage algorithms
 * - Dynamic shared memory allocation
 * - Shared memory for histograms
 * - Shared memory for sorting (bitonic)
 * - Shared memory for convolution
 *
 * SKILLS PRACTICED:
 * - Complex synchronization patterns
 * - Dynamic shared memory sizing
 * - Cooperative algorithms
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define HISTOGRAM_BINS 256

// ============================================================================
// KERNEL 1: Shared Memory Histogram
 * Build a histogram using shared memory for efficient binning
 * TODO: Complete the shared memory histogram implementation
// ============================================================================
__global__ void sharedHistogram(float *input, int n, unsigned int *histogram) {
    // TODO: Declare shared memory for local histogram bins
    // __shared__ unsigned int localHist[HISTOGRAM_BINS];
    __shared__ unsigned int localHist[/* YOUR CODE HERE */];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Initialize shared memory bins to 0 (cooperatively)
    // Each thread initializes a subset of bins
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x) {
        /* YOUR CODE HERE */
    }
    __syncthreads();

    // TODO: Process input data in strided fashion
    // Each thread processes multiple elements
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        // TODO: Determine bin (0 to HISTOGRAM_BINS-1) from input value
        // Assume input values are 0.0 to 255.99
        int bin = /* YOUR CODE HERE */;

        // TODO: Atomically increment the local histogram bin
        /* YOUR CODE HERE */
    }

    __syncthreads();

    // TODO: Write local histogram to global memory
    // Each thread writes a subset of bins atomically to global histogram
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x) {
        if (localHist[i] > 0) {
            /* YOUR CODE HERE - atomicAdd to global histogram */
        }
    }
}

// ============================================================================
// KERNEL 2: Shared Memory Bitonic Sort (Incomplete)
 * Sort elements within a block using bitonic sort network
 * TODO: Complete the compare-and-swap stages
// ============================================================================
__global__ void sharedBitonicSort(float *input, float *output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    if (idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 999999.0f;  // Sentinel value
    }
    __syncthreads();

    // TODO: Implement bitonic sort stages
    // For a block of size BLOCK_SIZE, we need log2(BLOCK_SIZE) stages
    // Each stage has multiple compare-and-swap steps

    // Pseudocode:
    // for (int k = 2; k <= BLOCK_SIZE; k *= 2) {
    //     for (int j = k / 2; j > 0; j /= 2) {
    //         int ixj = tid ^ j;  // XOR to find compare partner
    //         if (ixj > tid) {
    //             if ((tid & k) == 0) {
    //                 // Ascending: swap if sharedData[tid] > sharedData[ixj]
    //             } else {
    //                 // Descending: swap if sharedData[tid] < sharedData[ixj]
    //             }
    //         }
    //         __syncthreads();
    //     }
    // }

    /* YOUR CODE HERE - Complete the bitonic sort */

    // Store sorted data
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}

// ============================================================================
// KERNEL 3: 1D Convolution with Shared Memory
 * Apply a 1D convolution filter using shared memory for stencil access
 * TODO: Complete the convolution with halo loading
// ============================================================================
__global__ void convolution1D(float *input, float *output, float *kernel, 
                              int n, int kernelSize) {
    // TODO: Declare shared memory with halo regions
    // Need BLOCK_SIZE + kernelSize - 1 elements
    const int haloSize = (kernelSize - 1) / 2;
    __shared__ float sharedData[/* YOUR CODE HERE */];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Load data with halo regions for boundary handling
    // Each thread loads one element, edge threads load halo
    if (idx < n) {
        sharedData[tid + haloSize] = input[idx];
    }

    // TODO: Load left halo (boundary threads)
    if (tid < haloSize) {
        int leftIdx = blockIdx.x * blockDim.x - haloSize + tid;
        sharedData[tid] = (leftIdx >= 0) ? input[leftIdx] : 0.0f;
    }

    // TODO: Load right halo (boundary threads)
    if (tid < haloSize) {
        int rightIdx = (blockIdx.x + 1) * blockDim.x + tid;
        sharedData[BLOCK_SIZE + tid] = (rightIdx < n) ? input[rightIdx] : 0.0f;
    }

    __syncthreads();

    // TODO: Apply convolution
    // output[idx] = sum(kernel[k] * sharedData[tid + k - haloSize]) for k = 0 to kernelSize-1
    if (idx < n) {
        float sum = 0.0f;
        /* YOUR CODE HERE - Apply convolution kernel */
        output[idx] = sum;
    }
}

// ============================================================================
// KERNEL 4: Dynamic Shared Memory Multi-Algorithm
 * Use dynamic shared memory to run different algorithms based on parameter
 * TODO: Complete the algorithm selection and implementation
// ============================================================================
__global__ void dynamicSharedMulti(float *input, float *output, int n, 
                                   int algorithm, int param) {
    // Dynamic shared memory - size determined at launch
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Implement different algorithms based on 'algorithm' parameter
    // algorithm = 0: Prefix sum (scan)
    // algorithm = 1: Reduction (sum)
    // algorithm = 2: Broadcast (first element to all)

    switch (algorithm) {
        case 0:  // Prefix sum
            // TODO: Implement shared memory prefix sum
            // Load, synchronize, then upsweep/downsweep phases
            break;

        case 1:  // Reduction
            // TODO: Implement shared memory reduction
            // Load, synchronize, then tree-based reduction
            break;

        case 2:  // Broadcast
            // TODO: Implement broadcast
            // Thread 0 loads, synchronize, all read from shared[0]
            break;
    }
}

// ============================================================================
// KERNEL 5: Matrix Multiplication with Shared Memory Caching
 * Advanced: Cache multiple tiles for better data reuse
 * TODO: Implement double-buffering with shared memory
// ============================================================================
__global__ void matMulDoubleBuffer(float *A, float *B, float *C, int width) {
    // TODO: Declare shared memory for double-buffering
    // Need space for 2 tiles of A and 2 tiles of B
    const int TILE_WIDTH = 16;
    __shared__ float As[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[2][TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float accumulator = 0.0f;
    int numTiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    // TODO: Implement double-buffered matrix multiplication
    // While computing with tile pair [t%2], prefetch tile pair [(t+1)%2]
    // This hides memory latency through overlapping

    /* YOUR CODE HERE */

    if (row < width && col < width) {
        C[row * width + col] = accumulator;
    }
}

// Utility functions
void initRandom(float *arr, int n, float maxVal) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % (int)(maxVal * 100)) / 100.0f;
    }
}

void initKernel(float *kernel, int size) {
    // Simple averaging kernel
    for (int i = 0; i < size; i++) {
        kernel[i] = 1.0f / size;
    }
}

bool verifyHistogram(unsigned int *result, float *input, int n, int bins) {
    unsigned int *expected = (unsigned int*)calloc(bins, sizeof(unsigned int));
    for (int i = 0; i < n; i++) {
        int bin = (int)input[i];
        if (bin >= 0 && bin < bins) {
            expected[bin]++;
        }
    }

    bool pass = true;
    for (int i = 0; i < bins; i++) {
        if (result[i] != expected[i]) {
            pass = false;
            break;
        }
    }
    free(expected);
    return pass;
}

bool verifyConvolution(float *result, float *input, float *kernel, 
                       int n, int kernelSize) {
    int haloSize = (kernelSize - 1) / 2;
    for (int i = 0; i < n; i++) {
        float expected = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            int idx = i + k - haloSize;
            float val = (idx >= 0 && idx < n) ? input[idx] : 0.0f;
            expected += kernel[k] * val;
        }
        if (fabsf(result[i] - expected) > 1e-4f) return false;
    }
    return true;
}

int main() {
    printf("=== Shared Memory Level 4: Advanced Patterns ===\n\n");

    // Test 1: Histogram
    printf("Test 1: Shared Memory Histogram\n");
    const int HIST_N = 100000;
    float *h_histIn = (float*)malloc(HIST_N * sizeof(float));
    unsigned int *h_hist = (unsigned int*)calloc(HISTOGRAM_BINS, sizeof(unsigned int));
    unsigned int *h_histOut = (unsigned int*)calloc(HISTOGRAM_BINS, sizeof(unsigned int));

    initRandom(h_histIn, HIST_N, HISTOGRAM_BINS);

    float *d_histIn;
    unsigned int *d_hist;
    cudaMalloc(&d_histIn, HIST_N * sizeof(float));
    cudaMalloc(&d_hist, HISTOGRAM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_histIn, h_histIn, HIST_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, HISTOGRAM_BINS * sizeof(unsigned int));

    int histBlocks = 64;
    int histThreads = 256;

    // TODO: Launch sharedHistogram kernel
    // sharedHistogram<<<histBlocks, histThreads>>>(d_histIn, HIST_N, d_hist);
    printf("  Launching histogram kernel...\n");
    /* YOUR CODE HERE */

    cudaMemcpy(h_histOut, d_hist, HISTOGRAM_BINS * sizeof(unsigned int), 
               cudaMemcpyDeviceToHost);

    if (verifyHistogram(h_histOut, h_histIn, HIST_N, HISTOGRAM_BINS)) {
        printf("  ✓ Histogram PASSED\n");
    } else {
        printf("  ✗ Histogram FAILED - Complete the implementation\n");
    }

    // Test 2: Convolution
    printf("\nTest 2: 1D Convolution\n");
    const int CONV_N = 10000;
    const int KERNEL_SIZE = 5;
    float *h_convIn = (float*)malloc(CONV_N * sizeof(float));
    float *h_convOut = (float*)malloc(CONV_N * sizeof(float));
    float *h_convKernel = (float*)malloc(KERNEL_SIZE * sizeof(float));
    float *h_convExpected = (float*)malloc(CONV_N * sizeof(float));

    initRandom(h_convIn, CONV_N, 10.0f);
    initKernel(h_convKernel, KERNEL_SIZE);

    // Compute expected on CPU
    int haloSize = (KERNEL_SIZE - 1) / 2;
    for (int i = 0; i < CONV_N; i++) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            int idx = i + k - haloSize;
            float val = (idx >= 0 && idx < CONV_N) ? h_convIn[idx] : 0.0f;
            sum += h_convKernel[k] * val;
        }
        h_convExpected[i] = sum;
    }

    float *d_convIn, *d_convOut, *d_convKernel;
    cudaMalloc(&d_convIn, CONV_N * sizeof(float));
    cudaMalloc(&d_convOut, CONV_N * sizeof(float));
    cudaMalloc(&d_convKernel, KERNEL_SIZE * sizeof(float));
    cudaMemcpy(d_convIn, h_convIn, CONV_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convKernel, h_convKernel, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int convBlocks = (CONV_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // TODO: Launch convolution1D kernel
    // convolution1D<<<convBlocks, BLOCK_SIZE>>>(d_convIn, d_convOut, d_convKernel, 
    //                                           CONV_N, KERNEL_SIZE);
    printf("  Launching convolution kernel...\n");
    /* YOUR CODE HERE */

    cudaMemcpy(h_convOut, d_convOut, CONV_N * sizeof(float), cudaMemcpyDeviceToHost);

    if (verifyConvolution(h_convOut, h_convIn, h_convKernel, CONV_N, KERNEL_SIZE)) {
        printf("  ✓ Convolution PASSED\n");
    } else {
        printf("  ✗ Convolution FAILED - Complete the halo loading and convolution\n");
    }

    // Test 3: Dynamic shared memory (reduction example)
    printf("\nTest 3: Dynamic Shared Memory Reduction\n");
    const int RED_N = 1000000;
    float *h_redIn = (float*)malloc(RED_N * sizeof(float));
    for (int i = 0; i < RED_N; i++) h_redIn[i] = 1.0f;  // All 1s for easy verification
    float *h_redOut = (float*)malloc(sizeof(float));

    float *d_redIn, *d_redOut;
    cudaMalloc(&d_redIn, RED_N * sizeof(float));
    cudaMalloc(&d_redOut, sizeof(float));
    cudaMemcpy(d_redIn, h_redIn, RED_N * sizeof(float), cudaMemcpyHostToDevice);

    int redBlocks = 64;
    int redThreads = 256;
    int sharedMemSize = redThreads * sizeof(float);

    printf("  (Implementation exercise - see kernel comments)\n");
    // TODO: Implement reduction in dynamicSharedMulti and launch:
    // dynamicSharedMulti<<<redBlocks, redThreads, sharedMemSize>>>(
    //     d_redIn, d_redOut, RED_N, 1, 0);  // algorithm 1 = reduction

    // Cleanup
    free(h_histIn);
    free(h_hist);
    free(h_histOut);
    free(h_convIn);
    free(h_convOut);
    free(h_convKernel);
    free(h_convExpected);
    free(h_redIn);
    free(h_redOut);
    cudaFree(d_histIn);
    cudaFree(d_hist);
    cudaFree(d_convIn);
    cudaFree(d_convOut);
    cudaFree(d_convKernel);
    cudaFree(d_redIn);
    cudaFree(d_redOut);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory enables efficient cooperative algorithms\n");
    printf("- Histograms use shared memory for binning, then atomic add to global\n");
    printf("- Convolution needs halo regions for boundary handling\n");
    printf("- Dynamic shared memory allows flexible sizing at runtime\n");
    printf("- Double-buffering overlaps computation with memory transfers\n");
    printf("\n=== Shared Memory Module Complete ===\n");
    printf("Next: Explore reduction_patterns for more optimization techniques\n");

    return 0;
}
