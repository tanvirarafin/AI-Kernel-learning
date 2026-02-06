#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define PAGE_SIZE 16
#define TILE_SIZE 64

// Structure to represent the paged KV-cache
struct PagedKVCache {
    float* keys;           // Physical key storage
    float* values;         // Physical value storage
    int* page_table;       // Mapping from virtual to physical pages
    int num_heads;
    int head_dim;
    int max_pages;
    int page_size;         // Number of tokens per page
    
    PagedKVCache(int max_pages, int page_size, int num_heads, int head_dim) 
        : max_pages(max_pages), page_size(page_size), 
          num_heads(num_heads), head_dim(head_dim) {
        
        // Allocate physical storage for all possible pages
        int total_tokens = max_pages * page_size;
        cudaMalloc(&keys, total_tokens * num_heads * head_dim * sizeof(float));
        cudaMalloc(&values, total_tokens * num_heads * head_dim * sizeof(float));
        cudaMalloc(&page_table, max_pages * sizeof(int));
    }
    
    ~PagedKVCache() {
        cudaFree(keys);
        cudaFree(values);
        cudaFree(page_table);
    }
};

// Device function to get physical address from virtual address
__device__ int getPhysicalAddr(int virtual_page_id, int offset_in_page, 
                              int* page_table, int page_size) {
    int physical_page_id = page_table[virtual_page_id];
    return physical_page_id * page_size + offset_in_page;
}

// Function to access keys in paged cache
__device__ float* getPagedKeyPtr(float* physical_keys, int virtual_pos, 
                                int head_id, int head_dim, 
                                int* page_table, int page_size) {
    int page_id = virtual_pos / page_size;
    int offset_in_page = virtual_pos % page_size;
    int physical_addr = getPhysicalAddr(page_id, offset_in_page, page_table, page_size);
    
    return &physical_keys[(physical_addr * head_dim + head_id * head_dim)];
}

// Function to access values in paged cache
__device__ float* getPagedValuePtr(float* physical_values, int virtual_pos, 
                                 int head_id, int head_dim, 
                                 int* page_table, int page_size) {
    int page_id = virtual_pos / page_size;
    int offset_in_page = virtual_pos % page_size;
    int physical_addr = getPhysicalAddr(page_id, offset_in_page, page_table, page_size);
    
    return &physical_values[(physical_addr * head_dim + head_id * head_dim)];
}

// Optimized PagedAttention kernel
__global__ void optimizedPagedAttention(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* page_table,
    float* output,
    int* seq_lengths,
    int batch_size,
    int max_seq_len,
    int num_heads,
    int head_dim,
    float scale) {
    
    extern __shared__ float shared_mem[];
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if(token_id >= seq_lengths[batch_id]) return;  // Skip padding tokens
    
    // Load query into shared memory
    for(int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int q_idx = batch_id * max_seq_len * num_heads * head_dim + 
                    token_id * num_heads * head_dim + 
                    head_id * head_dim + d;
        shared_mem[d] = query[q_idx];
    }
    __syncthreads();
    
    // Process keys and values in tiles
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_vals[128];  // Assuming head_dim <= 128
    
    for(int d = 0; d < head_dim; d++) {
        output_vals[d] = 0.0f;
    }
    
    // Process in tiles of keys/values
    for(int tile_start = 0; tile_start <= token_id; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, token_id + 1);
        
        // Compute attention scores for this tile
        for(int k_token = tile_start; k_token < tile_end; k_token++) {
            float score = 0.0f;
            
            // Compute attention score
            for(int d = 0; d < head_dim; d++) {
                float* k_ptr = getPagedKeyPtr(
                    (float*)key_cache, 
                    k_token, 
                    head_id, 
                    head_dim, 
                    (int*)page_table, 
                    PAGE_SIZE
                );
                score += shared_mem[d] * k_ptr[d];
            }
            score *= scale;
            
            // Online softmax
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_score = expf(score - max_score);
            sum_exp = sum_exp * expf(old_max - max_score) + exp_score;
            
            // Accumulate values
            float* v_ptr = getPagedValuePtr(
                (float*)value_cache, 
                k_token, 
                head_id, 
                head_dim, 
                (int*)page_table, 
                PAGE_SIZE
            );
            
            for(int d = 0; d < head_dim; d++) {
                output_vals[d] = output_vals[d] * expf(old_max - max_score) + v_ptr[d] * exp_score;
            }
        }
    }
    
    // Write normalized output
    for(int d = 0; d < head_dim; d++) {
        int out_idx = batch_id * max_seq_len * num_heads * head_dim + 
                      token_id * num_heads * head_dim + 
                      head_id * head_dim + d;
        output[out_idx] = output_vals[d] / sum_exp;
    }
}

// PagedAttention kernel without optimization
__global__ void pagedAttention(
    const float* query,           // [batch_size, seq_len, num_heads, head_dim]
    const float* key_cache,       // Physical key cache
    const float* value_cache,     // Physical value cache
    const int* page_table,        // Page table mapping
    float* output,                // [batch_size, seq_len, num_heads, head_dim]
    int* seq_lengths,             // Actual sequence length for each batch
    int batch_size, 
    int max_seq_len,
    int num_heads,
    int head_dim,
    int page_size,
    float scale) {
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if(token_id >= seq_lengths[batch_id]) return;  // Skip padding tokens
    
    // Get query for this token and head
    int q_idx = batch_id * max_seq_len * num_heads * head_dim + 
                token_id * num_heads * head_dim + 
                head_id * head_dim;
    
    // Compute attention scores with all previous tokens in the sequence
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_vals[128];  // Assuming head_dim <= 128
    
    // Initialize output
    for(int d = 0; d < head_dim; d++) {
        output_vals[d] = 0.0f;
    }
    
    // Iterate through all previous tokens in the sequence
    for(int k_token = 0; k_token <= token_id; k_token++) {
        // Compute attention score with key at k_token
        float score = 0.0f;
        for(int d = 0; d < head_dim; d++) {
            float q_val = query[q_idx + d];
            
            // Access key from paged cache
            float* k_ptr = getPagedKeyPtr(
                (float*)key_cache, 
                k_token, 
                head_id, 
                head_dim, 
                (int*)page_table, 
                page_size
            );
            score += q_val * k_ptr[d];
        }
        score *= scale;  // Scale by 1/sqrt(head_dim)
        
        // Online softmax computation
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_score = expf(score - max_score);
        sum_exp = sum_exp * expf(old_max - max_score) + exp_score;
        
        // Accumulate weighted values
        float* v_ptr = getPagedValuePtr(
            (float*)value_cache, 
            k_token, 
            head_id, 
            head_dim, 
            (int*)page_table, 
            page_size
        );
        
        for(int d = 0; d < head_dim; d++) {
            output_vals[d] = output_vals[d] * expf(old_max - max_score) + v_ptr[d] * exp_score;
        }
    }
    
    // Normalize and write output
    for(int d = 0; d < head_dim; d++) {
        int out_idx = batch_id * max_seq_len * num_heads * head_dim + 
                      token_id * num_heads * head_dim + 
                      head_id * head_dim + d;
        output[out_idx] = output_vals[d] / sum_exp;
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
    const int MAX_SEQ_LEN = 128;
    const int NUM_HEADS = 8;
    const int HEAD_DIM = 64;
    const int MAX_PAGES = 32;
    const int PAGE_SIZE_VAL = 16;
    const int TOTAL_SIZE = BATCH_SIZE * MAX_SEQ_LEN * NUM_HEADS * HEAD_DIM;
    
    // Host memory allocation
    std::vector<float> h_query(TOTAL_SIZE);
    std::vector<float> h_output(TOTAL_SIZE);
    std::vector<int> h_seq_lengths(BATCH_SIZE);
    
    // Initialize query with random values
    for(int i = 0; i < TOTAL_SIZE; i++) {
        h_query[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;  // -1 to 1
    }
    
    // Set sequence lengths (variable length sequences)
    h_seq_lengths[0] = 100;  // First sequence has 100 tokens
    h_seq_lengths[1] = 80;   // Second sequence has 80 tokens
    
    // Create paged KV cache
    PagedKVCache* paged_cache = new PagedKVCache(MAX_PAGES, PAGE_SIZE_VAL, NUM_HEADS, HEAD_DIM);
    
    // Initialize page table (identity mapping for simplicity)
    std::vector<int> h_page_table(MAX_PAGES);
    for(int i = 0; i < MAX_PAGES; i++) {
        h_page_table[i] = i;  // Initially identity mapping
    }
    
    // Device memory allocation
    float *d_query, *d_output;
    int *d_seq_lengths, *d_page_table;
    cudaMalloc(&d_query, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_output, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_seq_lengths, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_page_table, MAX_PAGES * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_query, h_query.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_lengths, h_seq_lengths.data(), BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_page_table, h_page_table.data(), MAX_PAGES * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize KV cache with random values
    int kv_cache_size = MAX_PAGES * PAGE_SIZE_VAL * NUM_HEADS * HEAD_DIM;
    std::vector<float> h_kv_cache(kv_cache_size);
    for(int i = 0; i < kv_cache_size; i++) {
        h_kv_cache[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
    
    cudaMemcpy(paged_cache->keys, h_kv_cache.data(), kv_cache_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(paged_cache->values, h_kv_cache.data(), kv_cache_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch PagedAttention kernel
    dim3 grid(BATCH_SIZE, NUM_HEADS, (MAX_SEQ_LEN + 255) / 256);
    dim3 block(256);
    
    float scale = 1.0f / sqrtf(HEAD_DIM);
    
    optimizedPagedAttention<<<grid, block, HEAD_DIM * sizeof(float)>>>(
        d_query,
        paged_cache->keys,
        paged_cache->values,
        d_page_table,
        d_output,
        d_seq_lengths,
        BATCH_SIZE,
        MAX_SEQ_LEN,
        NUM_HEADS,
        HEAD_DIM,
        scale
    );
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "PagedAttention completed." << std::endl;
    std::cout << "Sequence lengths: [" << h_seq_lengths[0] << ", " << h_seq_lengths[1] << "]" << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Test with different sequence lengths
    std::vector<int> h_seq_lengths_test = {50, 75};
    cudaMemcpy(d_seq_lengths, h_seq_lengths_test.data(), BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    optimizedPagedAttention<<<grid, block, HEAD_DIM * sizeof(float)>>>(
        d_query,
        paged_cache->keys,
        paged_cache->values,
        d_page_table,
        d_output,
        d_seq_lengths,
        BATCH_SIZE,
        MAX_SEQ_LEN,
        NUM_HEADS,
        HEAD_DIM,
        scale
    );
    
    cudaMemcpy(h_output.data(), d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nPagedAttention with different sequence lengths completed." << std::endl;
    std::cout << "New sequence lengths: [" << h_seq_lengths_test[0] << ", " << h_seq_lengths_test[1] << "]" << std::endl;
    std::cout << "Output (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_query);
    cudaFree(d_output);
    cudaFree(d_seq_lengths);
    cudaFree(d_page_table);
    delete paged_cache;
    
    return 0;
}