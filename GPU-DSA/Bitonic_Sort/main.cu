#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Bitonic sort step kernel
__global__ void bitonicSortStep(int* arr, int n, int k, int j) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = idx ^ j;  // XOR operation
    
    if(ixj > idx && idx < n && ixj < n) {
        if((idx & k) == 0) {
            // Sort in ascending order
            if(arr[idx] > arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if(arr[idx] < arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void gpuBitonicSort(int* d_arr, int n) {
    // Ensure n is a power of 2
    int size = 1;
    while(size < n) size <<= 1;
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // Main sorting loop
    for(int k = 2; k <= size; k <<= 1) {
        for(int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<grid, block>>>(d_arr, n, k, j);
            
            // Synchronize after each step to ensure correctness
            cudaDeviceSynchronize();
        }
    }
}

// Optimized bitonic sort with shared memory
__global__ void optimizedBitonicSort(int* input, int* output, int n, int k, int j) {
    extern __shared__ int sharedData[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load data into shared memory
    if(idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = INT_MAX; // Fill with max value for out-of-bounds
    }
    __syncthreads();
    
    // Perform bitonic sort step
    int ixj = tid ^ j;  // XOR operation within block
    
    if(ixj > tid && tid < blockDim.x) {
        bool ascending = (tid & k) == 0;
        
        if(ascending) {
            // Sort in ascending order
            if(sharedData[tid] > sharedData[ixj]) {
                int temp = sharedData[tid];
                sharedData[tid] = sharedData[ixj];
                sharedData[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if(sharedData[tid] < sharedData[ixj]) {
                int temp = sharedData[tid];
                sharedData[tid] = sharedData[ixj];
                sharedData[ixj] = temp;
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory
    if(idx < n) {
        output[idx] = sharedData[tid];
    }
}

// Complete bitonic sort implementation
__global__ void bitonicCompare(int* data, int size, int k, int j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;
    
    if(ixj > idx && idx < size && ixj < size) {
        // Determine direction based on k
        bool ascending = (idx & k) == 0;
        
        // Compare and swap if needed
        if(ascending) {
            if(data[idx] > data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if(data[idx] < data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void completeBitonicSort(int* d_data, int n) {
    // Ensure n is a power of 2
    int size = 1;
    while(size < n) size <<= 1;
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // Main sorting loop
    for(int k = 2; k <= size; k <<= 1) {
        for(int j = k >> 1; j > 0; j >>= 1) {
            bitonicCompare<<<grid, block>>>(d_data, n, k, j);
            
            // Synchronize after each step to ensure correctness
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    // Test with a power of 2 size for bitonic sort
    const int N = 1024;  // Must be power of 2 for bitonic sort
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // Host memory allocation
    std::vector<int> h_input(N);
    std::vector<int> h_output(N);
    
    // Initialize input with random values
    for(int i = 0; i < N; i++) {
        h_input[i] = N - i;  // Reverse order for worst case
    }
    
    // Device memory allocation
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Copy to output for in-place sorting
    cudaMemcpy(d_output, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Perform bitonic sort
    completeBitonicSort(d_output, N);
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify the result is sorted
    bool isSorted = std::is_sorted(h_output.begin(), h_output.end());
    std::cout << "Bitonic sort result is " << (isSorted ? "correct" : "incorrect") << std::endl;
    
    // Print first 10 elements
    std::cout << "First 10 elements after bitonic sort: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Print last 10 elements
    std::cout << "Last 10 elements after bitonic sort: ";
    for(int i = N-10; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with a smaller array to show the process more clearly
    const int N_small = 16;
    std::vector<int> h_small(N_small);
    std::vector<int> h_result(N_small);
    
    // Initialize small array
    for(int i = 0; i < N_small; i++) {
        h_small[i] = N_small - i;  // [16, 15, 14, ..., 1]
    }
    
    int *d_small, *d_result;
    cudaMalloc(&d_small, N_small * sizeof(int));
    cudaMalloc(&d_result, N_small * sizeof(int));
    
    cudaMemcpy(d_small, h_small.data(), N_small * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_small.data(), N_small * sizeof(int), cudaMemcpyHostToDevice);
    
    // Perform bitonic sort on small array
    completeBitonicSort(d_result, N_small);
    
    cudaMemcpy(h_result.data(), d_result, N_small * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "\nSmall array test:" << std::endl;
    std::cout << "Before: ";
    for(int i = 0; i < N_small; i++) {
        std::cout << h_small[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "After:  ";
    for(int i = 0; i < N_small; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify small array is sorted
    bool isSmallSorted = std::is_sorted(h_result.begin(), h_result.end());
    std::cout << "Small array sort result is " << (isSmallSorted ? "correct" : "incorrect") << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_small);
    cudaFree(d_result);
    
    return 0;
}