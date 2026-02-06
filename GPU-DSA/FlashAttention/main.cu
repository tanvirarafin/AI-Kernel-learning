#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define TILE_SIZE 64
#define HEAD_DIM 64

// Utility function for FlashAttention
__device__ void blockSoftmax(float* values, int n) {
    // Find max
    float max_val = values[0];
    for(int i = 1; i < n; i++) {
        if(values[i] > max_val) max_val = values[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for(int i = 0; i < n; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }
    
    // Normalize
    for(int i = 0; i < n; i++) {
        values[i] /= sum;
    }
}

// Optimized FlashAttention kernel
__global__ void optimizedFlashAttention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim, int batch_size) {
    
    const int tile_size = 64;
    const int threads_per_block = 256;
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if(token_id >= seq_len) return;
    
    // Process in tiles along the sequence dimension
    float l_sum = 0.0f;      // normalization factor
    float m_prev = -INFINITY; // maximum value
    float o_vals[HEAD_DIM];   // output accumulator
    
    // Initialize output accumulator
    for(int d = 0; d < HEAD_DIM; d++) {
        o_vals[d] = 0.0f;
    }
    
    // Iterate through K and V in tiles
    for(int k_start = 0; k_start < seq_len; k_start += tile_size) {
        int k_end = min(k_start + tile_size, seq_len);
        
        // Compute attention scores for current token against current K tile
        float scores[TILE_SIZE];
        int valid_scores = 0;
        
        for(int k_idx = k_start; k_idx < k_end; k_idx++) {
            float score = 0.0f;
            
            // Compute Q·K for current token and K position
            for(int d = 0; d < head_dim; d++) {
                int q_idx = batch_id * seq_len * head_dim + token_id * head_dim + d;
                int k_idx_full = batch_id * seq_len * head_dim + k_idx * head_dim + d;
                score += Q[q_idx] * K[k_idx_full];
            }
            
            score /= sqrtf(head_dim);
            scores[valid_scores++] = score;
        }
        
        // Apply online softmax update
        float m_curr = m_prev;
        float l_curr = l_sum * expf(m_prev - fmaxf(m_prev, scores[0]));
        float o_curr[HEAD_DIM];
        
        // Initialize current output
        for(int d = 0; d < HEAD_DIM; d++) {
            o_curr[d] = o_vals[d] * expf(m_prev - fmaxf(m_prev, scores[0]));
        }
        
        // Process each score in the tile
        for(int i = 0; i < valid_scores; i++) {
            float new_m = fmaxf(m_curr, scores[i]);
            float exp_m_curr = expf(m_curr - new_m);
            float exp_scores_i = expf(scores[i] - new_m);
            
            float new_l = l_curr * exp_m_curr + exp_scores_i;
            
            // Update output with V values
            for(int d = 0; d < head_dim; d++) {
                int v_idx = batch_id * seq_len * head_dim + (k_start + i) * head_dim + d;
                o_curr[d] = (o_curr[d] * l_curr * exp_m_curr + V[v_idx] * exp_scores_i) / new_l;
            }
            
            m_curr = new_m;
            l_curr = new_l;
        }
        
        m_prev = m_curr;
        l_sum = l_curr;
        
        for(int d = 0; d < HEAD_DIM; d++) {
            o_vals[d] = o_curr[d];
        }
    }
    
    // Write final output
    for(int d = 0; d < HEAD_DIM; d++) {
        int out_idx = batch_id * seq_len * head_dim + token_id * head_dim + d;
        output[out_idx] = o_vals[d];
    }
}

// Tiled attention computation
__global__ void tiledAttention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim) {
    
    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float K_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float V_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float O_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float l_vec[TILE_SIZE];  // normalization factors
    __shared__ float m_vec[TILE_SIZE];  // maximum values
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tx = threadIdx.x;  // within tile
    int ty = threadIdx.y;  // within tile
    
    int block_seq_start = blockIdx.z * TILE_SIZE;
    
    // Initialize output tile
    if(tx < TILE_SIZE && ty < HEAD_DIM) {
        O_tile[tx][ty] = 0.0f;
    }
    if(tx < TILE_SIZE) {
        l_vec[tx] = 0.0f;
        m_vec[tx] = -INFINITY;
    }
    __syncthreads();
    
    // Process in tiles along the sequence dimension
    for(int k_start = 0; k_start < seq_len; k_start += TILE_SIZE) {
        // Load tiles of K and V
        int k_idx = k_start + ty;
        if(tx < TILE_SIZE && k_idx < seq_len) {
            int k_global_idx = batch_id * seq_len * head_dim + k_idx * head_dim + tx;
            K_tile[ty][tx] = (k_global_idx < batch_id * seq_len * head_dim + seq_len * head_dim) ? 
                             K[k_global_idx] : 0.0f;
        }
        
        int v_idx = k_start + ty;
        if(tx < TILE_SIZE && v_idx < seq_len) {
            int v_global_idx = batch_id * seq_len * head_dim + v_idx * head_dim + tx;
            V_tile[ty][tx] = (v_global_idx < batch_id * seq_len * head_dim + seq_len * head_dim) ? 
                             V[v_global_idx] : 0.0f;
        }
        __syncthreads();
        
        // Compute attention scores for current Q tile and K tile
        if(tx < TILE_SIZE && ty < HEAD_DIM) {
            int q_idx = block_seq_start + tx;
            if(q_idx < seq_len) {
                int q_global_idx = batch_id * seq_len * head_dim + q_idx * head_dim + ty;
                float q_val = (q_global_idx < batch_id * seq_len * head_dim + seq_len * head_dim) ? 
                              Q[q_global_idx] : 0.0f;
                
                // Compute attention score with all K values in current tile
                float score = 0.0f;
                for(int d = 0; d < HEAD_DIM; d++) {
                    score += q_val * K_tile[tx][d];
                }
                score /= sqrtf(HEAD_DIM);
                
                // Apply online softmax update
                float old_max = m_vec[tx];
                float new_max = fmaxf(old_max, score);
                
                // Update normalization constants
                float exp_old = expf(old_max - new_max);
                float exp_new = expf(score - new_max);
                
                float old_l = l_vec[tx];
                l_vec[tx] = old_l * exp_old + exp_new;
                
                // Update output
                for(int d = 0; d < HEAD_DIM; d++) {
                    O_tile[tx][d] = (O_tile[tx][d] * old_l * exp_old + V_tile[tx][ty] * exp_new) / l_vec[tx];
                }
                
                m_vec[tx] = new_max;
            }
        }
        __syncthreads();
    }
    
    // Write results back to global memory
    int out_seq_idx = block_seq_start + tx;
    if(out_seq_idx < seq_len && ty < HEAD_DIM) {
        int out_global_idx = batch_id * seq_len * head_dim + out_seq_idx * head_dim + ty;
        output[out_global_idx] = O_tile[tx][ty];
    }
}

// Simple attention for comparison
__global__ void simpleAttention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim,
    float scale) {
    
    int batch_id = blockIdx.x;
    int token_id = blockIdx.y * blockDim.y + threadIdx.y;
    int head_id = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(token_id >= seq_len || head_id >= head_dim) return;
    
    // Compute attention scores for this token
    float scores[64];  // Assuming max sequence length of 64 for this example
    int actual_seq_len = seq_len < 64 ? seq_len : 64;
    
    for(int k = 0; k < actual_seq_len; k++) {
        float score = 0.0f;
        for(int d = 0; d < head_dim; d++) {
            int q_idx = batch_id * seq_len * head_dim + token_id * head_dim + d;
            int k_idx = batch_id * seq_len * head_dim + k * head_dim + d;
            score += Q[q_idx] * K[k_idx];
        }
        scores[k] = score * scale;  // Apply scaling
    }
    
    // Apply softmax to scores
    float max_score = scores[0];
    for(int k = 1; k < actual_seq_len; k++) {
        if(scores[k] > max_score) max_score = scores[k];
    }
    
    float sum_exp = 0.0f;
    for(int k = 0; k < actual_seq_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum_exp += scores[k];
    }
    
    for(int k = 0; k < actual_seq_len; k++) {
        scores[k] /= sum_exp;
    }
    
    // Compute output as weighted sum of V values
    float result = 0.0f;
    for(int k = 0; k < actual_seq_len; k++) {
        int v_idx = batch_id * seq_len * head_dim + k * head_dim + head_id;
        result += scores[k] * V[v_idx];
    }
    
    int out_idx = batch_id * seq_len * head_dim + token_id * head_dim + head_id;
    output[out_idx] = result;
}

int main() {
    const int BATCH_SIZE = 2;
    const int SEQ_LEN = 128;
    const int HEAD_DIM = 64;
    const int TOTAL_SIZE = BATCH_SIZE * SEQ_LEN * HEAD_DIM;
    
    // Host memory allocation
    std::vector<float> h_Q(TOTAL_SIZE);
    std::vector<float> h_K(TOTAL_SIZE);
    std::vector<float> h_V(TOTAL_SIZE);
    std::vector<float> h_output(TOTAL_SIZE);
    
    // Initialize with random values
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_Q[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;  // -1 to 1
        h_K[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_V[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Device memory allocation
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_K, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_V, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_output, TOTAL_SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_Q, h_Q.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch FlashAttention kernel
    dim3 grid(BATCH_SIZE, 1, (SEQ_LEN + 255) / 256);  // 256 threads per block
    dim3 block(256);
    
    optimizedFlashAttention<<<grid, block>>>(d_Q, d_K, d_V, d_output, SEQ_LEN, HEAD_DIM, BATCH_SIZE);
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "FlashAttention completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with smaller dimensions for tiled attention
    const int SMALL_SEQ = 64;
    const int SMALL_HEAD = 64;
    const int SMALL_TOTAL = BATCH_SIZE * SMALL_SEQ * SMALL_HEAD;
    
    std::vector<float> h_Q_small(SMALL_TOTAL);
    std::vector<float> h_K_small(SMALL_TOTAL);
    std::vector<float> h_V_small(SMALL_TOTAL);
    std::vector<float> h_output_small(SMALL_TOTAL);
    
    // Initialize small tensors
    for(int i = 0; i < SMALL_TOTAL; i++) {
        h_Q_small[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_K_small[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        h_V_small[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    float *d_Q_small, *d_K_small, *d_V_small, *d_output_small;
    cudaMalloc(&d_Q_small, SMALL_TOTAL * sizeof(float));
    cudaMalloc(&d_K_small, SMALL_TOTAL * sizeof(float));
    cudaMalloc(&d_V_small, SMALL_TOTAL * sizeof(float));
    cudaMalloc(&d_output_small, SMALL_TOTAL * sizeof(float));
    
    cudaMemcpy(d_Q_small, h_Q_small.data(), SMALL_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_small, h_K_small.data(), SMALL_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_small, h_V_small.data(), SMALL_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch tiled attention kernel
    dim3 grid_tiled(BATCH_SIZE, 1, (SMALL_SEQ + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block_tiled(TILE_SIZE, HEAD_DIM);
    
    tiledAttention<<<grid_tiled, block_tiled>>>(d_Q_small, d_K_small, d_V_small, d_output_small, 
                                               SMALL_SEQ, SMALL_HEAD);
    
    cudaMemcpy(h_output_small.data(), d_output_small, SMALL_TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nTiled attention completed." << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output_small[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_Q_small);
    cudaFree(d_K_small);
    cudaFree(d_V_small);
    cudaFree(d_output_small);
    
    return 0;
}