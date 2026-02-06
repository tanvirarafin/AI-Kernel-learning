#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// Atomic aggregation histogram
__global__ void atomicHistogram(int* input, int* histogram, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < n) {
        int value = input[idx];
        if(value >= 0 && value < numBins) {
            atomicAdd(&histogram[value], 1);
        }
    }
}

// Privatized histogram (per-block)
__global__ void privatizedHistogram(int* input, int* histogram, int n, int numBins) {
    // Shared memory for block-private histogram
    extern __shared__ int blockHist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory histogram
    for(int i = tid; i < numBins; i += blockDim.x) {
        blockHist[i] = 0;
    }
    __syncthreads();
    
    // Count in shared memory
    if(idx < n) {
        int value = input[idx];
        if(value >= 0 && value < numBins) {
            atomicAdd(&blockHist[value], 1);  // Safe since we're in shared memory
        }
    }
    __syncthreads();
    
    // Merge block histogram to global histogram
    for(int i = tid; i < numBins; i += blockDim.x) {
        if(blockHist[i] > 0) {
            atomicAdd(&histogram[i], blockHist[i]);
        }
    }
}

// Optimized privatized histogram with warps
__global__ void optimizedPrivatizedHistogram(int* input, int* histogram, int n, int numBins) {
    // Shared memory for block-private histogram
    extern __shared__ int blockHist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Initialize shared memory histogram
    for(int i = tid; i < numBins; i += blockDim.x) {
        blockHist[i] = 0;
    }
    __syncthreads();
    
    // Process elements with coalesced access pattern
    if(idx < n) {
        int value = input[idx];
        if(value >= 0 && value < numBins) {
            atomicAdd(&blockHist[value], 1);
        }
    }
    
    // Process additional elements if array is larger than grid
    idx += gridDim.x * blockDim.x;
    while(idx < n) {
        int value = input[idx];
        if(value >= 0 && value < numBins) {
            atomicAdd(&blockHist[value], 1);
        }
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    
    // Merge block histogram to global histogram
    for(int i = tid; i < numBins; i += blockDim.x) {
        if(blockHist[i] > 0) {
            atomicAdd(&histogram[i], blockHist[i]);
        }
    }
}

int main() {
    const int N = 1000000;  // 1M elements
    const int numBins = 256;  // Histogram with 256 bins
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // Host memory allocation
    std::vector<int> h_input(N);
    std::vector<int> h_histogram(numBins, 0);
    
    // Initialize input with random values in range [0, numBins)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, numBins - 1);
    
    for(int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }
    
    // Device memory allocation
    int *d_input, *d_histogram;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_histogram, numBins * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize histogram to zero on device
    cudaMemset(d_histogram, 0, numBins * sizeof(int));
    
    // Launch atomic histogram kernel
    atomicHistogram<<<gridSize, blockSize>>>(d_input, d_histogram, N, numBins);
    
    // Copy result back to host
    cudaMemcpy(h_histogram.data(), d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Atomic histogram - First 10 bins: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_histogram[i] << " ";
    }
    std::cout << std::endl;
    
    // Reset histogram
    cudaMemset(d_histogram, 0, numBins * sizeof(int));
    
    // Launch privatized histogram kernel
    int sharedMemSize = numBins * sizeof(int);
    privatizedHistogram<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_histogram, N, numBins);
    
    // Copy result back to host
    cudaMemcpy(h_histogram.data(), d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Privatized histogram - First 10 bins: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_histogram[i] << " ";
    }
    std::cout << std::endl;
    
    // Reset histogram
    cudaMemset(d_histogram, 0, numBins * sizeof(int));
    
    // Launch optimized privatized histogram kernel
    optimizedPrivatizedHistogram<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_histogram, N, numBins);
    
    // Copy result back to host
    cudaMemcpy(h_histogram.data(), d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Optimized privatized histogram - First 10 bins: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_histogram[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify the total count
    int totalCount = 0;
    for(int i = 0; i < numBins; i++) {
        totalCount += h_histogram[i];
    }
    std::cout << "Total count: " << totalCount << " (should be " << N << ")" << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_histogram);
    
    return 0;
}