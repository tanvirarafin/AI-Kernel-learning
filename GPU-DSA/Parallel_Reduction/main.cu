#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <algorithm>

// Parallel reduction using tree-based approach
__global__ void treeReduction(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if(tid % (2*s) == 0 && (tid + s) < blockDim.x) {
            temp[tid] += temp[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if(tid == 0) output[blockIdx.x] = temp[0];
}

// Optimized tree-based reduction
__global__ void optimizedReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Perform first level of reduction
    float mySum = (i < n) ? input[i] : 0.0f;
    if(i + blockDim.x < n) mySum += input[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduce in shared memory
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

// Warp shuffle reduction
__device__ float warpReduce(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void warpShuffleReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    float mySum = (i < n) ? input[i] : 0.0f;
    sdata[tid] = mySum;
    __syncthreads();
    
    // Perform reduction in shared memory until we have warps remaining
    for(int s = blockDim.x/2; s > warpSize; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Perform warp-level reduction
    if(tid < warpSize) {
        mySum = warpReduce(sdata[tid]);
    }
    
    // Write result for this block to global memory
    if(tid == 0) output[blockIdx.x] = mySum;
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // Host memory allocation
    std::vector<float> h_data(N);
    for(int i = 0; i < N; i++) {
        h_data[i] = 1.0f;  // Simple test: sum should be N
    }
    
    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    treeReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    
    // Copy result back to host
    std::vector<float> h_output(gridSize);
    cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Final reduction on CPU (since we have few blocks)
    float finalResult = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        finalResult += h_output[i];
    }
    
    std::cout << "Tree-based reduction result: " << finalResult << std::endl;
    std::cout << "Expected result: " << N << std::endl;
    
    // Test optimized version
    optimizedReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output.data(), d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    finalResult = 0.0f;
    for(int i = 0; i < gridSize; i++) {
        finalResult += h_output[i];
    }
    
    std::cout << "Optimized reduction result: " << finalResult << std::endl;
    
    // Test warp shuffle version
    const int gridSizeWarp = (N + blockSize - 1) / blockSize;
    warpShuffleReduction<<<gridSizeWarp, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output.data(), d_output, gridSizeWarp * sizeof(float), cudaMemcpyDeviceToHost);
    
    finalResult = 0.0f;
    for(int i = 0; i < gridSizeWarp; i++) {
        finalResult += h_output[i];
    }
    
    std::cout << "Warp shuffle reduction result: " << finalResult << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}