#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Helper function to find maximum in parallel
__global__ void findMax(float* input, float* max_val, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();
    
    // Perform reduction to find max
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if(tid == 0) {
        max_val[blockIdx.x] = sdata[0];
    }
}

// Warp-level reduction for max
__device__ float warpReduceMax(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warpReduceSum(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized online softmax with warp-level primitives
__global__ void optimizedOnlineSoftmax(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with -infinity for out-of-bounds elements
    float val = (i < n) ? input[i] : -INFINITY;
    
    // Find maximum using warp-level operations
    float max_val = warpReduceMax(val);
    
    // Share max value within block
    if(tid % warpSize == 0) {
        sdata[tid / warpSize] = max_val;
    }
    __syncthreads();
    
    // First warp finds block max
    if(tid < blockDim.x / warpSize) {
        max_val = sdata[tid];
    } else {
        max_val = -INFINITY;
    }
    
    if(tid % warpSize == 0) {
        max_val = warpReduceMax(max_val);
    }
    __syncthreads();
    
    // Broadcast block max
    max_val = sdata[0];
    __syncthreads();
    
    // Compute exponentials and sum
    if(i < n) {
        val = expf(input[i] - max_val);
    } else {
        val = 0.0f;
    }
    
    // Compute sum using warp-level operations
    float sum = warpReduceSum(val);
    
    // Share sum within block
    if(tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();
    
    // First warp sums partial sums
    if(tid < blockDim.x / warpSize) {
        sum = sdata[tid];
    } else {
        sum = 0.0f;
    }
    
    if(tid % warpSize == 0) {
        sum = warpReduceSum(sum);
    }
    __syncthreads();
    
    // Broadcast sum
    sum = sdata[0];
    __syncthreads();
    
    // Compute final softmax values
    if(i < n) {
        output[i] = val / sum;  // val already contains expf(input[i] - max_val)
    }
}

// Batch online softmax for neural networks
__global__ void batchOnlineSoftmax(float* input, float* output, 
                                 int batch_size, int seq_len) {
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if(batch_id >= batch_size || tid >= seq_len) return;
    
    extern __shared__ float sdata[];
    
    // Load sequence for this batch
    int offset = batch_id * seq_len;
    float val = input[offset + tid];
    
    // Find max in the sequence
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction to find max
    for(int s = seq_len / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < seq_len) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    __syncthreads();
    
    // Compute exponentials
    if(tid < seq_len) {
        sdata[tid] = expf(input[offset + tid] - max_val);
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reduction to find sum
    for(int s = seq_len / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < seq_len) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum = sdata[0];
    __syncthreads();
    
    // Compute final softmax values
    if(tid < seq_len) {
        output[offset + tid] = sdata[tid] / sum;
    }
}

// Standard softmax for comparison (unstable)
__global__ void standardSoftmax(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find maximum value across all elements
    float max_val = (i < n) ? input[i] : -INFINITY;
    sdata[tid] = max_val;
    __syncthreads();
    
    // Reduction to find max in shared memory
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Broadcast max value
    max_val = sdata[0];
    __syncthreads();
    
    // Compute sum of exponentials
    float exp_val = (i < n) ? expf(input[i] - max_val) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();
    
    // Reduction to compute sum
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Broadcast sum
    float sum = sdata[0];
    __syncthreads();
    
    // Compute final softmax values
    if(i < n) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

int main() {
    const int N = 1024;  // Sequence length
    const int BATCH_SIZE = 8;  // Number of batches
    const int TOTAL_SIZE = BATCH_SIZE * N;
    
    // Host memory allocation
    std::vector<float> h_input(TOTAL_SIZE);
    std::vector<float> h_output(TOTAL_SIZE);
    
    // Initialize input with values that could cause overflow in standard softmax
    for(int b = 0; b < BATCH_SIZE; b++) {
        for(int i = 0; i < N; i++) {
            // Use values that could cause overflow in standard softmax
            h_input[b * N + i] = 10.0f + static_cast<float>(rand()) / RAND_MAX * 20.0f;
        }
    }
    
    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_output, TOTAL_SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Launch batch online softmax kernel
    size_t sharedMemSize = N * sizeof(float);
    batchOnlineSoftmax<<<BATCH_SIZE, N, sharedMemSize>>>(d_input, d_output, BATCH_SIZE, N);
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Batch online softmax completed." << std::endl;
    
    // Verify that probabilities sum to 1 for each batch
    bool all_valid = true;
    for(int b = 0; b < BATCH_SIZE; b++) {
        float sum = 0.0f;
        for(int i = 0; i < N; i++) {
            sum += h_output[b * N + i];
        }
        
        if(std::abs(sum - 1.0f) > 1e-4) {
            std::cout << "Batch " << b << " sum: " << sum << " (should be 1.0)" << std::endl;
            all_valid = false;
        }
    }
    
    std::cout << "Softmax sums " << (all_valid ? "valid" : "invalid") << std::endl;
    
    // Print first batch results
    std::cout << "First batch softmax output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with problematic values that would cause standard softmax to fail
    const int PROBLEM_SIZE = 16;
    std::vector<float> h_problem(PROBLEM_SIZE);
    std::vector<float> h_result(PROBLEM_SIZE);
    
    // Fill with very large values that would cause overflow in standard softmax
    for(int i = 0; i < PROBLEM_SIZE; i++) {
        h_problem[i] = 100.0f + i;  // Large values that cause exp overflow
    }
    
    float *d_problem, *d_result;
    cudaMalloc(&d_problem, PROBLEM_SIZE * sizeof(float));
    cudaMalloc(&d_result, PROBLEM_SIZE * sizeof(float));
    
    cudaMemcpy(d_problem, h_problem.data(), PROBLEM_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use online softmax which handles large values correctly
    optimizedOnlineSoftmax<<<1, PROBLEM_SIZE, PROBLEM_SIZE * sizeof(float)>>>(
        d_problem, d_result, PROBLEM_SIZE);
    
    cudaMemcpy(h_result.data(), d_result, PROBLEM_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nTesting with large values (would cause overflow in standard softmax):" << std::endl;
    std::cout << "Input: ";
    for(int i = 0; i < PROBLEM_SIZE; i++) {
        std::cout << h_problem[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Online Softmax Output: ";
    float prob_sum = 0.0f;
    for(int i = 0; i < PROBLEM_SIZE; i++) {
        std::cout << h_result[i] << " ";
        prob_sum += h_result[i];
    }
    std::cout << std::endl;
    
    std::cout << "Sum of probabilities: " << prob_sum << " (should be 1.0)" << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_problem);
    cudaFree(d_result);
    
    return 0;
}