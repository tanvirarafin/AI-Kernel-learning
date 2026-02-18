/*
 * Atomic Operations Level 3: Histogram with Atomics
 *
 * EXERCISE: Build efficient histograms using atomic operations.
 *
 * CONCEPTS:
 * - Parallel histogram construction
 * - Bin contention management
 * - Private bins vs shared bins
 * - Atomic contention reduction
 *
 * SKILLS PRACTICED:
 * - atomicAdd for histogram bins
 * - Bin index calculation
 * - Contention-aware algorithms
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define NUM_BINS 256

// ============================================================================
// KERNEL 1: Basic Atomic Histogram
 * Each thread atomically increments the appropriate bin
 * TODO: Complete the histogram implementation
// ============================================================================
__global__ void atomicHistogram(float *input, unsigned int *histogram, 
                                 int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Calculate bin index from input value
        // Assume input values are in range [0, numBins)
        int bin = /* YOUR CODE HERE - (int)input[idx] */;
        
        // TODO: Bounds check and atomic increment
        if (bin >= 0 && bin < numBins) {
            // atomicAdd(&histogram[bin], 1);
            /* YOUR CODE HERE */
        }
    }
}

// ============================================================================
// KERNEL 2: Histogram with Private Bins (Per-Block)
 * Each block has private bins, then merge at the end
 * TODO: Complete the private bin histogram
// ============================================================================
__global__ void privateBinHistogram(float *input, unsigned int *privateHist,
                                     int n, int numBins, int numBlocks) {
    __shared__ unsigned int sharedHist[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Initialize shared memory bins to 0
    for (int i = tid; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int bin = (int)input[i];
        if (bin >= 0 && bin < numBins) {
            // TODO: Atomically increment shared bin
            /* YOUR CODE HERE */
        }
    }
    __syncthreads();
    
    // TODO: Copy shared histogram to private (global) histogram for this block
    // privateHist[blockIdx.x * numBins + tid] = sharedHist[tid];
    /* YOUR CODE HERE */
}

// Merge kernel: combine private histograms
__global__ void mergeHistograms(unsigned int *privateHist, 
                                 unsigned int *globalHist,
                                 int numBins, int numBlocks) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bin < numBins) {
        unsigned int sum = 0;
        
        // TODO: Sum across all block histograms
        for (int b = 0; b < numBlocks; b++) {
            sum += privateHist[b * numBins + bin];
        }
        
        globalHist[bin] = sum;
    }
}

// ============================================================================
// KERNEL 3: Atomic Histogram with Grid-Stride
 * Use grid-stride loop for better scalability
 * TODO: Complete the grid-stride histogram
// ============================================================================
__global__ void gridStrideHistogram(float *input, unsigned int *histogram,
                                     int n, int numBins) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned int sharedHist[256];
    
    // TODO: Initialize shared memory
    /* YOUR CODE HERE */
    __syncthreads();
    
    // TODO: Grid-stride loop over input
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int bin = (int)input[i];
        if (bin >= 0 && bin < numBins) {
            // TODO: Atomic increment
            /* YOUR CODE HERE */
        }
    }
    
    __syncthreads();
    
    // TODO: Write shared histogram to global
    /* YOUR CODE HERE */
}

// ============================================================================
// KERNEL 4: 2D Histogram (Bivariate)
 * Create a 2D histogram from pairs of values
 * TODO: Complete the 2D histogram
// ============================================================================
__global__ void histogram2D(float *inputX, float *inputY,
                            unsigned int *hist2D,
                            int n, int binsX, int binsY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // TODO: Calculate 2D bin indices
        int binX = (int)inputX[idx];
        int binY = (int)inputY[idx];
        
        // TODO: Bounds check
        if (binX >= 0 && binX < binsX && binY >= 0 && binY < binsY) {
            // TODO: Calculate 1D index for 2D histogram
            // int bin1D = binX * binsY + binY;
            // atomicAdd(&hist2D[bin1D], 1);
            /* YOUR CODE HERE */
        }
    }
}

// ============================================================================
// KERNEL 5: Weighted Histogram
 * Each element contributes a weight, not just 1
 * TODO: Complete the weighted histogram
// ============================================================================
__global__ void weightedHistogram(float *input, float *weights,
                                   unsigned int *histogram,
                                   int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = (int)input[idx];
        
        // TODO: Bounds check and atomic add with weight
        if (bin >= 0 && bin < numBins) {
            // atomicAdd(&histogram[bin], (unsigned int)weights[idx]);
            /* YOUR CODE HERE */
        }
    }
}

// Utility functions
void initInput(float *arr, int n, int numBins) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % numBins);
    }
}

bool verifyHistogram(unsigned int *result, float *input, int n, int numBins) {
    unsigned int *expected = (unsigned int*)calloc(numBins, sizeof(unsigned int));
    
    for (int i = 0; i < n; i++) {
        int bin = (int)input[i];
        if (bin >= 0 && bin < numBins) {
            expected[bin]++;
        }
    }
    
    bool pass = true;
    for (int i = 0; i < numBins; i++) {
        if (result[i] != expected[i]) {
            pass = false;
            break;
        }
    }
    free(expected);
    return pass;
}

int main() {
    printf("=== Atomic Operations Level 3: Histogram ===\n\n");
    
    const int N = 1000000;
    const int NUM_BINS = 256;
    
    float *h_input = (float*)malloc(N * sizeof(float));
    unsigned int *h_hist = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *h_histOut = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    
    initInput(h_input, N, NUM_BINS);
    
    float *d_input;
    unsigned int *d_hist;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = 64;
    
    // Test 1: Basic atomic histogram
    printf("Test 1: Basic atomic histogram\n");
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
    atomicHistogram<<<gridSize, blockSize>>>(d_input, d_hist, N, NUM_BINS);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_histOut, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    if (verifyHistogram(h_histOut, h_input, N, NUM_BINS)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete the bin calculation and atomic add\n");
    }
    
    // Test 2: Private bin histogram
    printf("\nTest 2: Private bin histogram with merge\n");
    unsigned int *d_privateHist;
    cudaMalloc(&d_privateHist, gridSize * NUM_BINS * sizeof(unsigned int));
    
    privateBinHistogram<<<gridSize, blockSize>>>(d_input, d_privateHist, N, NUM_BINS, gridSize);
    cudaDeviceSynchronize();
    
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
    mergeHistograms<<<(NUM_BINS + blockSize - 1) / blockSize, blockSize>>>(
        d_privateHist, d_hist, NUM_BINS, gridSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_histOut, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    if (verifyHistogram(h_histOut, h_input, N, NUM_BINS)) {
        printf("  ✓ PASSED\n");
    } else {
        printf("  ✗ FAILED - Complete private bins and merge\n");
    }
    
    // Test 3: 2D histogram
    printf("\nTest 3: 2D histogram\n");
    const int BINS_2D = 16;
    float *h_inputY = (float*)malloc(N * sizeof(float));
    initInput(h_inputY, N, BINS_2D);
    
    float *d_inputY;
    unsigned int *d_hist2D;
    cudaMalloc(&d_inputY, N * sizeof(float));
    cudaMalloc(&d_hist2D, BINS_2D * BINS_2D * sizeof(unsigned int));
    
    cudaMemcpy(d_inputY, h_inputY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_hist2D, 0, BINS_2D * BINS_2D * sizeof(unsigned int));
    
    histogram2D<<<gridSize, blockSize>>>(d_input, d_inputY, d_hist2D, N, BINS_2D, BINS_2D);
    cudaDeviceSynchronize();
    
    unsigned int *h_hist2DOut = (unsigned int*)malloc(BINS_2D * BINS_2D * sizeof(unsigned int));
    cudaMemcpy(h_hist2DOut, d_hist2D, BINS_2D * BINS_2D * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Verify total count
    unsigned int total = 0;
    for (int i = 0; i < BINS_2D * BINS_2D; i++) {
        total += h_hist2DOut[i];
    }
    printf("  Total counts: %u (Expected: %d)\n", total, N);
    
    // Cleanup
    free(h_input);
    free(h_inputY);
    free(h_hist);
    free(h_histOut);
    cudaFree(d_input);
    cudaFree(d_inputY);
    cudaFree(d_hist);
    cudaFree(d_privateHist);
    cudaFree(d_hist2D);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Basic histogram: atomicAdd per element\n");
    printf("- Private bins reduce contention, need merge step\n");
    printf("- Shared memory histogram reduces global atomics\n");
    printf("- 2D histogram: flatten to 1D index\n");
    printf("- Grid-stride improves scalability\n");
    printf("\nNext: Try level4_locks_mutex.cu for synchronization\n");
    
    return 0;
}
