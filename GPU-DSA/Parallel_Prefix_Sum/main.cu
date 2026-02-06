#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

// Kogge-Stone algorithm for parallel prefix sum
__global__ void koggeStoneScan(int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2*thid] = input[2*thid];
    temp[2*thid+1] = input[2*thid+1];
    
    // Up-sweep (reduce) phase
    for(int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if(thid == 0) temp[n-1] = 0;
    
    // Down-sweep phase
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results to device memory
    output[2*thid] = temp[2*thid];
    output[2*thid+1] = temp[2*thid+1];
}

// Work-efficient Blelloch algorithm
__global__ void blellochScan(int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    temp[2*thid] = input[2*thid];
    temp[2*thid+1] = input[2*thid+1];
    
    // Up-sweep (reduce) phase
    for(int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if(thid == 0) temp[n-1] = 0;
    
    // Down-sweep phase
    for(int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if(thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results to device memory
    output[2*thid] = temp[2*thid];
    output[2*thid+1] = temp[2*thid+1];
}

// Simple inclusive scan kernel
__global__ void simpleInclusiveScan(int* input, int* output, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Perform inclusive scan in shared memory
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if(tid >= stride) {
            temp = sdata[tid - stride];
        }
        __syncthreads();
        
        if(tid >= stride) {
            sdata[tid] += temp;
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if(i < n) {
        output[i] = sdata[tid];
    }
}

int main() {
    const int N = 16;  // Small example for demonstration
    const int blockSize = 16;
    const int gridSize = 1;  // For this example
    
    // Host memory allocation
    std::vector<int> h_input(N);
    std::vector<int> h_output(N);
    
    // Initialize input: [1, 1, 1, 1, ...]
    for(int i = 0; i < N; i++) {
        h_input[i] = 1;
    }
    
    // Device memory allocation
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch simple inclusive scan kernel
    simpleInclusiveScan<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Input: ";
    for(int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Inclusive Scan Result: ";
    for(int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Expected result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    // Test Kogge-Stone algorithm with a smaller example
    const int n_small = 8;
    std::vector<int> h_input_small(n_small);
    std::vector<int> h_output_small(n_small);
    
    for(int i = 0; i < n_small; i++) {
        h_input_small[i] = (i < 4) ? 1 : 2;  // [1, 1, 1, 1, 2, 2, 2, 2]
    }
    
    int *d_input_small, *d_output_small;
    cudaMalloc(&d_input_small, n_small * sizeof(int));
    cudaMalloc(&d_output_small, n_small * sizeof(int));
    
    cudaMemcpy(d_input_small, h_input_small.data(), n_small * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch Kogge-Stone kernel
    koggeStoneScan<<<1, n_small/2, n_small * sizeof(int)>>>(d_input_small, d_output_small, n_small);
    
    cudaMemcpy(h_output_small.data(), d_output_small, n_small * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "\nKogge-Stone Input: ";
    for(int i = 0; i < n_small; i++) {
        std::cout << h_input_small[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Kogge-Stone Result: ";
    for(int i = 0; i < n_small; i++) {
        std::cout << h_output_small[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_small);
    cudaFree(d_output_small);
    
    return 0;
}