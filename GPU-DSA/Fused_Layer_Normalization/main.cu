#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Warp-level reduction for sum
__device__ float warpReduce(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__device__ float warpReduceMax(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp-level reduction for min
__device__ float warpReduceMin(float val) {
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Fused Layer Norm with Residual Connection
__global__ void fusedLayerNormResidual(
    const float* input,
    const float* residual,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon) {
    
    extern __shared__ float sdata[];
    
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if(batch_id >= batch_size) return;
    
    // Add residual connection first
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // Process with residual addition and compute statistics
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_id * hidden_size + i;
        float val = input[idx] + residual[idx];  // Add residual
        sdata[i % blockDim.x] = val;
        __syncthreads();
        
        // Compute partial sums
        float local_val = sdata[i % blockDim.x];
        sum += local_val;
        sq_sum += local_val * local_val;
    }
    
    // Store partial sums in shared memory
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sq_sum;
    __syncthreads();
    
    // Reduction to compute total sum
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    // Calculate mean and variance
    float mean = sdata[0] / hidden_size;
    float expected_sq = sdata[blockDim.x] / hidden_size;
    float variance = expected_sq - mean * mean;
    float inv_stddev = rsqrtf(fmaxf(variance + epsilon, 1e-6f));
    
    // Apply normalization and write output
    if(tid < hidden_size) {
        int idx = batch_id * hidden_size + tid;
        float val = input[idx] + residual[idx];  // Add residual
        float normalized = (val - mean) * inv_stddev;
        output[idx] = gamma[tid] * normalized + beta[tid];
    }
}

// Fused Layer Norm with Activation (GELU)
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fusedLayerNormActivation(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon,
    bool apply_activation = true) {
    
    extern __shared__ float sdata[];
    
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if(batch_id >= batch_size) return;
    
    // Compute statistics
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_id * hidden_size + i;
        float val = input[idx];
        sum += val;
        sq_sum += val * val;
    }
    
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sq_sum;
    __syncthreads();
    
    // Reduction
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    // Calculate statistics
    float mean = sdata[0] / hidden_size;
    float expected_sq = sdata[blockDim.x] / hidden_size;
    float variance = expected_sq - mean * mean;
    float inv_stddev = rsqrtf(fmaxf(variance + epsilon, 1e-6f));
    
    // Apply normalization and activation
    if(tid < hidden_size) {
        int idx = batch_id * hidden_size + tid;
        float normalized = (input[idx] - mean) * inv_stddev;
        float transformed = gamma[tid] * normalized + beta[tid];
        
        // Apply activation if needed
        if(apply_activation) {
            transformed = gelu(transformed);
        }
        
        output[idx] = transformed;
    }
}

// Warp-optimized Layer Norm
__global__ void warpOptimizedLayerNorm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon) {
    
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    
    if(batch_id >= batch_size) return;
    
    // Each thread computes partial sums for multiple elements
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // Process elements with stride
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_id * hidden_size + i;
        float val = input[idx];
        sum += val;
        sq_sum += val * val;
    }
    
    // Warp-level reduction
    sum = warpReduce(sum);
    sq_sum = warpReduce(sq_sum);
    
    // Only the first thread in each warp stores results
    if(lane_id == 0) {
        __shared__ float warp_sums[32];  // Assuming max 32 warps per block
        __shared__ float warp_sq_sums[32];
        
        warp_sums[warp_id] = sum;
        warp_sq_sums[warp_id] = sq_sum;
    }
    __syncthreads();
    
    // Process warp sums
    if(warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? 
              warp_sums[lane_id] : 0.0f;
        sq_sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? 
                 warp_sq_sums[lane_id] : 0.0f;
        
        sum = warpReduce(sum);
        sq_sum = warpReduce(sq_sum);
    }
    __syncthreads();
    
    // Calculate mean and variance
    float mean = sum / hidden_size;
    float expected_sq = sq_sum / hidden_size;
    float variance = expected_sq - mean * mean;
    float inv_stddev = rsqrtf(fmaxf(variance + epsilon, 1e-6f));
    
    // Apply normalization
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_id * hidden_size + i;
        float normalized = (input[idx] - mean) * inv_stddev;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}

// Standard Layer Norm for comparison
__global__ void standardLayerNorm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon) {
    
    extern __shared__ float sdata[];
    
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if(batch_id >= batch_size || tid >= hidden_size) return;
    
    // Compute mean
    float sum = 0.0f;
    for(int i = 0; i < hidden_size; i++) {
        int idx = batch_id * hidden_size + i;
        sum += input[idx];
    }
    float mean = sum / hidden_size;
    
    // Compute variance
    float var_sum = 0.0f;
    for(int i = 0; i < hidden_size; i++) {
        int idx = batch_id * hidden_size + i;
        float diff = input[idx] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / hidden_size;
    
    // Normalize and apply affine transformation
    int current_idx = batch_id * hidden_size + tid;
    float normalized = (input[current_idx] - mean) / sqrtf(variance + epsilon);
    output[current_idx] = gamma[tid] * normalized + beta[tid];
}

int main() {
    const int BATCH_SIZE = 4;
    const int HIDDEN_SIZE = 512;
    const int TOTAL_SIZE = BATCH_SIZE * HIDDEN_SIZE;
    const float EPSILON = 1e-5f;
    
    // Host memory allocation
    std::vector<float> h_input(TOTAL_SIZE);
    std::vector<float> h_residual(TOTAL_SIZE);
    std::vector<float> h_gamma(HIDDEN_SIZE);
    std::vector<float> h_beta(HIDDEN_SIZE);
    std::vector<float> h_output(TOTAL_SIZE);
    
    // Initialize input with random values
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;  // -2 to 2
        h_residual[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;  // Small residual
    }
    
    // Initialize gamma and beta
    for(int i = 0; i < HIDDEN_SIZE; i++) {
        h_gamma[i] = 1.0f + (static_cast<float>(rand()) / RAND_MAX) * 0.1f;  // ~1.0
        h_beta[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;  // ~0.0
    }
    
    // Device memory allocation
    float *d_input, *d_residual, *d_gamma, *d_beta, *d_output;
    cudaMalloc(&d_input, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_residual, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_gamma, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_beta, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, TOTAL_SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch fused layer norm with residual
    int blockSize = 256;
    int gridSize = BATCH_SIZE;
    
    fusedLayerNormResidual<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
        d_input, d_residual, d_output, d_gamma, d_beta, 
        BATCH_SIZE, HIDDEN_SIZE, EPSILON
    );
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Fused Layer Norm with Residual completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Launch fused layer norm with activation
    fusedLayerNormActivation<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
        d_input, d_output, d_gamma, d_beta, 
        BATCH_SIZE, HIDDEN_SIZE, EPSILON, true
    );
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nFused Layer Norm with Activation completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Launch warp-optimized layer norm
    warpOptimizedLayerNorm<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
        d_input, d_output, d_gamma, d_beta, 
        BATCH_SIZE, HIDDEN_SIZE, EPSILON
    );
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nWarp-optimized Layer Norm completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with different parameters
    const int SMALL_BATCH = 2;
    const int SMALL_HIDDEN = 128;
    const int SMALL_TOTAL = SMALL_BATCH * SMALL_HIDDEN;
    
    std::vector<float> h_input_small(SMALL_TOTAL);
    std::vector<float> h_output_small(SMALL_TOTAL);
    
    for(int i = 0; i < SMALL_TOTAL; i++) {
        h_input_small[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    float *d_input_small, *d_output_small;
    cudaMalloc(&d_input_small, SMALL_TOTAL * sizeof(float));
    cudaMalloc(&d_output_small, SMALL_TOTAL * sizeof(float));
    
    cudaMemcpy(d_input_small, h_input_small.data(), SMALL_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch with smaller dimensions
    int small_grid = SMALL_BATCH;
    int small_block = min(SMALL_HIDDEN, 256);
    
    warpOptimizedLayerNorm<<<small_grid, small_block, 2 * small_block * sizeof(float)>>>(
        d_input_small, d_output_small, d_gamma, d_beta, 
        SMALL_BATCH, SMALL_HIDDEN, EPSILON
    );
    
    cudaMemcpy(h_output_small.data(), d_output_small, SMALL_TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nSmall dimensions test completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output_small[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    cudaFree(d_input_small);
    cudaFree(d_output_small);
    
    return 0;
}