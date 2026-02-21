/**
 * Shared Histogram - Kernel 1 from level4_advanced_patterns.cu
 * 
 * This kernel demonstrates building a histogram using shared memory.
 * Each block builds a local histogram, then merges to global.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 100000
#define HISTOGRAM_BINS 256
#define BLOCK_SIZE 256

__global__ void sharedHistogram(float *input, int n, unsigned int *histogram) {
    // Shared memory for local histogram bins
    __shared__ unsigned int localHist[HISTOGRAM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory bins to 0 (cooperatively)
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Process input data in strided fashion
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        // Determine bin from input value (0 to HISTOGRAM_BINS-1)
        int bin = (int)input[i];
        if (bin >= 0 && bin < HISTOGRAM_BINS) {
            // Atomically increment the local histogram bin
            atomicAdd(&localHist[bin], 1);
        }
    }

    __syncthreads();

    // Write local histogram to global memory
    for (int i = tid; i < HISTOGRAM_BINS; i += blockDim.x) {
        if (localHist[i] > 0) {
            atomicAdd(&histogram[i], localHist[i]);
        }
    }
}

void initRandom(float *arr, int n, float maxVal) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % (int)(maxVal * 100)) / 100.0f;
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

int main() {
    printf("=== Shared Memory Histogram ===\n\n");

    const int HIST_N = 100000;
    float *h_histIn = (float*)malloc(HIST_N * sizeof(float));
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

    printf("Launching histogram kernel...\n");
    sharedHistogram<<<histBlocks, histThreads>>>(d_histIn, HIST_N, d_hist);
    cudaDeviceSynchronize();

    cudaMemcpy(h_histOut, d_hist, HISTOGRAM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (verifyHistogram(h_histOut, h_histIn, HIST_N, HISTOGRAM_BINS)) {
        printf("Histogram PASSED\n");
    } else {
        printf("Histogram FAILED\n");
    }

    // Cleanup
    free(h_histIn);
    free(h_histOut);
    cudaFree(d_histIn);
    cudaFree(d_hist);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory reduces global atomic contention\n");
    printf("- Each block builds local histogram, then merges\n");
    printf("- atomicAdd for thread-safe bin increments\n");

    return 0;
}
