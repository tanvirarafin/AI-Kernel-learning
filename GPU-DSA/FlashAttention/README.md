# FlashAttention: Fused Attention with Recomputation

## Overview

FlashAttention is an optimized attention mechanism that significantly reduces memory usage and improves performance compared to standard attention implementations. It achieves this through tiling, recomputation, and fusion of operations, making it possible to scale transformer models to much longer sequences.

## Why FlashAttention?

Standard attention has quadratic memory and computational complexity with respect to sequence length:
- **Memory**: O(N²) for storing attention scores
- **Computation**: O(N²) for attention score calculations
- **Bandwidth**: High memory bandwidth requirements

FlashAttention addresses these issues by:
- Reducing memory usage from O(N²) to O(N)
- Improving memory bandwidth efficiency
- Maintaining numerical accuracy
- Enabling longer sequence lengths

## Key Concepts

### Standard Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### FlashAttention Approach
- Process attention in tiles
- Recompute intermediate values instead of storing them
- Fuse operations to reduce memory traffic
- Use HBM (High Bandwidth Memory) efficiently

### Tiled Computation
Divide Q, K, V matrices into smaller tiles and process them iteratively.

### Recomputation Strategy
Trade computation for memory by recomputing values instead of storing them.

## Memory vs Computation Trade-off

### Standard Attention
- Stores full attention matrix: O(N²) memory
- Fewer computations: O(N²) total operations

### FlashAttention
- Stores only tiles: O(N) memory
- More computations due to recomputation: O(N²) operations but better memory access

## Step-by-Step Implementation Guide

### Step 1: Standard Attention (for comparison)
```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__global__ void standardAttention(
    const float* Q, const float* K, const float* V,
    float* output, float* attention_scores,
    int seq_len, int head_dim) {
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = threadIdx.x + blockIdx.z * blockDim.x;
    
    if(token_id >= seq_len) return;
    
    // Compute attention scores for this token
    for(int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for(int d = 0; d < head_dim; d++) {
            int q_idx = batch_id * seq_len * head_dim + token_id * head_dim + d;
            int k_idx = batch_id * seq_len * head_dim + k * head_dim + d;
            score += Q[q_idx] * K[k_idx];
        }
        score /= sqrtf(head_dim);
        attention_scores[token_id * seq_len + k] = score;
    }
    
    // Apply softmax to attention scores
    float max_score = attention_scores[token_id * seq_len];
    for(int k = 1; k < seq_len; k++) {
        if(attention_scores[token_id * seq_len + k] > max_score) {
            max_score = attention_scores[token_id * seq_len + k];
        }
    }
    
    float sum_exp = 0.0f;
    for(int k = 0; k < seq_len; k++) {
        float exp_score = expf(attention_scores[token_id * seq_len + k] - max_score);
        attention_scores[token_id * seq_len + k] = exp_score;
        sum_exp += exp_score;
    }
    
    // Normalize and compute output
    for(int d = 0; d < head_dim; d++) {
        float result = 0.0f;
        for(int k = 0; k < seq_len; k++) {
            int v_idx = batch_id * seq_len * head_dim + k * head_dim + d;
            result += attention_scores[token_id * seq_len + k] * V[v_idx] / sum_exp;
        }
        int out_idx = batch_id * seq_len * head_dim + token_id * head_dim + d;
        output[out_idx] = result;
    }
}
```

### Step 2: Tiled Attention Computation
```cpp
#define TILE_SIZE 64
#define HEAD_DIM 64

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
```

### Step 3: FlashAttention with Online Softmax
```cpp
__global__ void flashAttention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim, int batch_size) {
    
    // Parameters
    const int tile_size = 64;
    const int head_dim_eff = HEAD_DIM;
    
    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float* Q_tile = shared_mem;
    float* K_tile = &shared_mem[tile_size * head_dim_eff];
    float* V_tile = &shared_mem[2 * tile_size * head_dim_eff];
    float* O_tile = &shared_mem[3 * tile_size * head_dim_eff];
    float* l_vec = &shared_mem[4 * tile_size * head_dim_eff];
    float* m_vec = &shared_mem[4 * tile_size * head_dim_eff + tile_size];
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tx = threadIdx.x;  // thread's position in tile
    int block_seq_start = blockIdx.z * tile_size;
    
    // Initialize output tile and normalization factors
    for(int i = tx; i < tile_size * head_dim_eff; i += blockDim.x) {
        if(i < tile_size * head_dim_eff) {
            int row = i / head_dim_eff;
            int col = i % head_dim_eff;
            O_tile[row * head_dim_eff + col] = 0.0f;
        }
    }
    
    for(int i = tx; i < tile_size * 2; i += blockDim.x) {
        if(i < tile_size) {
            l_vec[i] = 0.0f;
            m_vec[i] = -INFINITY;
        }
    }
    __syncthreads();
    
    // Iterate through K and V tiles
    for(int k_start = 0; k_start < seq_len; k_start += tile_size) {
        // Load K tile
        for(int i = tx; i < tile_size * head_dim_eff; i += blockDim.x) {
            int k_row = i / head_dim_eff;
            int k_col = i % head_dim_eff;
            int k_idx = k_start + k_row;
            
            if(k_idx < seq_len && k_col < head_dim_eff) {
                int k_global_idx = batch_id * seq_len * head_dim_eff + k_idx * head_dim_eff + k_col;
                K_tile[i] = K[k_global_idx];
            } else {
                K_tile[i] = 0.0f;
            }
        }
        
        // Load V tile
        for(int i = tx; i < tile_size * head_dim_eff; i += blockDim.x) {
            int v_row = i / head_dim_eff;
            int v_col = i % head_dim_eff;
            int v_idx = k_start + v_row;
            
            if(v_idx < seq_len && v_col < head_dim_eff) {
                int v_global_idx = batch_id * seq_len * head_dim_eff + v_idx * head_dim_eff + v_col;
                V_tile[i] = V[v_global_idx];
            } else {
                V_tile[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Process Q tile against current K,V tiles
        int q_idx = block_seq_start + tx;
        if(q_idx < seq_len) {
            // Load Q values for this thread
            float q_vals[HEAD_DIM];
            for(int d = 0; d < head_dim_eff; d++) {
                int q_global_idx = batch_id * seq_len * head_dim_eff + q_idx * head_dim_eff + d;
                q_vals[d] = Q[q_global_idx];
            }
            
            // Compute attention scores with current K tile
            for(int k_local = 0; k_local < tile_size && k_start + k_local < seq_len; k_local++) {
                float score = 0.0f;
                for(int d = 0; d < head_dim_eff; d++) {
                    int k_offset = k_local * head_dim_eff + d;
                    score += q_vals[d] * K_tile[k_offset];
                }
                score /= sqrtf(head_dim_eff);
                
                // Online softmax update
                float old_max = m_vec[tx];
                float new_max = fmaxf(old_max, score);
                float exp_old = expf(old_max - new_max);
                float exp_new = expf(score - new_max);
                
                float old_l = l_vec[tx];
                float new_l = old_l * exp_old + exp_new;
                
                // Update output with V values
                for(int d = 0; d < head_dim_eff; d++) {
                    int v_offset = k_local * head_dim_eff + d;
                    O_tile[tx * head_dim_eff + d] = 
                        (O_tile[tx * head_dim_eff + d] * old_l * exp_old + 
                         V_tile[v_offset] * exp_new) / new_l;
                }
                
                l_vec[tx] = new_l;
                m_vec[tx] = new_max;
            }
        }
        __syncthreads();
    }
    
    // Write final output
    int out_seq_idx = block_seq_start + tx;
    if(out_seq_idx < seq_len) {
        for(int d = 0; d < head_dim_eff; d++) {
            int out_global_idx = batch_id * seq_len * head_dim_eff + out_seq_idx * head_dim_eff + d;
            output[out_global_idx] = O_tile[tx * head_dim_eff + d];
        }
    }
}
```

### Step 4: Optimized FlashAttention with Fused Operations
```cpp
// Utility functions for FlashAttention
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
```

## Common Pitfalls and Solutions

### 1. Memory Layout Issues
- **Problem**: Inefficient memory access patterns
- **Solution**: Optimize for coalesced access and proper tiling

### 2. Numerical Stability
- **Problem**: Overflow/underflow in exponential computations
- **Solution**: Use online softmax with maximum subtraction

### 3. Shared Memory Limitations
- **Problem**: Insufficient shared memory for large tiles
- **Solution**: Adjust tile sizes based on available memory

### 4. Synchronization Errors
- **Problem**: Incorrect synchronization causing race conditions
- **Solution**: Proper use of `__syncthreads()` and memory fences

## Performance Considerations

### Memory Bandwidth
- FlashAttention optimizes memory bandwidth usage
- Focus on coalesced access patterns

### Arithmetic Intensity
- Balance between computation and memory access
- Optimize for the specific GPU architecture

### Tile Size Selection
- Trade-off between memory usage and computation efficiency
- Consider the GPU's shared memory capacity

### Sequence Length Scaling
- Performance characteristics change with sequence length
- Optimize for expected sequence length ranges

## Real-World Applications

- **Large Language Models**: Transformers with long contexts
- **Computer Vision**: Vision transformers with high-resolution inputs
- **Speech Recognition**: Processing long audio sequences
- **Genomics**: Analyzing long DNA/RNA sequences
- **Time Series**: Long-term forecasting and analysis

## Advanced Techniques

### Variable Sequence Lengths
Handle sequences of different lengths efficiently.

### Multi-GPU FlashAttention
Scale FlashAttention across multiple GPUs.

### Quantized FlashAttention
Use quantized operations to reduce memory and computation.

## Summary

FlashAttention revolutionizes attention computation by dramatically reducing memory usage from O(N²) to O(N) while maintaining numerical accuracy. Through tiling, recomputation, and fusion of operations, it enables scaling of transformer models to much longer sequences. Understanding FlashAttention is crucial for implementing efficient large-scale neural networks.