# PagedAttention: Virtual Memory Mapping for KV-Cache

## Overview

PagedAttention is an innovative attention mechanism that applies virtual memory concepts to transformer KV-cache management. It allows for efficient handling of variable-length sequences and significantly reduces memory fragmentation compared to traditional attention mechanisms. This technique enables serving longer sequences and larger batch sizes with the same memory footprint.

## Why PagedAttention?

Traditional transformer attention faces several memory challenges:
- **Fixed KV-cache allocation**: Must allocate for maximum possible sequence length
- **Memory fragmentation**: Inefficient memory usage with variable-length sequences
- **Limited context length**: Memory constraints limit maximum sequence length
- **Poor batching efficiency**: Waste memory when sequences have different lengths

PagedAttention addresses these issues by:
- Using virtual memory concepts for KV-cache management
- Enabling variable-length sequences without wasting memory
- Reducing memory fragmentation
- Supporting longer context lengths

## Key Concepts

### Virtual vs Physical Memory
- **Virtual Memory**: Logical addresses used by the model
- **Physical Memory**: Actual memory locations on the GPU
- **Page Table**: Maps virtual pages to physical pages

### KV-Cache Pages
- Divide the KV-cache into fixed-size pages
- Each page stores a segment of keys and values
- Pages can be allocated dynamically as needed

### Page Table
- Maintains mapping between virtual and physical pages
- Enables non-contiguous memory allocation
- Allows for efficient memory management

## Memory Management Benefits

### Reduced Fragmentation
- Pages can be allocated anywhere in memory
- No need for contiguous large allocations

### Flexible Allocation
- Allocate pages as sequences grow
- Deallocate pages when no longer needed

### Efficient Batching
- Different sequences can have different lengths
- No memory wasted on unused portions

## Step-by-Step Implementation Guide

### Step 1: Basic Page Table Structure
```cpp
#include <cuda_runtime.h>
#include <vector>

struct PageTable {
    int* page_indices;      // Maps virtual page ID to physical page ID
    int num_pages;          // Total number of virtual pages
    int page_size;          // Size of each page (in tokens)
    
    // Constructor
    PageTable(int max_pages, int page_size) : page_size(page_size) {
        cudaMalloc(&page_indices, max_pages * sizeof(int));
        num_pages = max_pages;
    }
    
    ~PageTable() {
        cudaFree(page_indices);
    }
};

// Structure to represent the paged KV-cache
struct PagedKVCache {
    float* keys;           // Physical key storage
    float* values;         // Physical value storage
    PageTable* page_table; // Mapping from virtual to physical pages
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
        
        page_table = new PageTable(max_pages, page_size);
    }
    
    ~PagedKVCache() {
        cudaFree(keys);
        cudaFree(values);
        delete page_table;
    }
};
```

### Step 2: Page Access Functions
```cpp
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
```

### Step 3: PagedAttention Kernel
```cpp
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
```

### Step 4: Optimized PagedAttention with Tiling
```cpp
#define PAGE_SIZE 16
#define TILE_SIZE 64

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
    
    if(token_id >= seq_lengths[batch_id]) return;
    
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
```

### Step 5: Complete PagedAttention System
```cpp
class PagedAttentionManager {
private:
    PagedKVCache* kv_cache;
    int* d_page_table;      // Device page table
    int* d_seq_lengths;     // Device sequence lengths
    int max_pages;
    int page_size;
    int num_heads;
    int head_dim;
    
public:
    PagedAttentionManager(int max_pages, int page_size, int num_heads, int head_dim)
        : max_pages(max_pages), page_size(page_size), 
          num_heads(num_heads), head_dim(head_dim) {
        
        kv_cache = new PagedKVCache(max_pages, page_size, num_heads, head_dim);
        cudaMalloc(&d_page_table, max_pages * sizeof(int));
        cudaMalloc(&d_seq_lengths, max_pages * sizeof(int));  // Assuming max batch size
        
        // Initialize page table (identity mapping initially)
        std::vector<int> h_page_table(max_pages);
        for(int i = 0; i < max_pages; i++) {
            h_page_table[i] = i;  // Initially identity mapping
        }
        cudaMemcpy(d_page_table, h_page_table.data(), 
                   max_pages * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    ~PagedAttentionManager() {
        delete kv_cache;
        cudaFree(d_page_table);
        cudaFree(d_seq_lengths);
    }
    
    // Function to allocate a new page
    int allocatePage() {
        // In a real implementation, this would manage free pages
        // For simplicity, we'll just return the next available page
        static int next_page = 0;
        return next_page++;
    }
    
    // Function to update page table
    void updatePageTable(int virtual_page_id, int physical_page_id) {
        int h_page_table[] = {physical_page_id};
        cudaMemcpy(&d_page_table[virtual_page_id], h_page_table, 
                   sizeof(int), cudaMemcpyHostToDevice);
    }
    
    // Main attention computation function
    void computeAttention(
        const float* query,
        float* output,
        const std::vector<int>& seq_lengths,
        int batch_size,
        int max_seq_len) {
        
        // Copy sequence lengths to device
        cudaMemcpy(d_seq_lengths, seq_lengths.data(), 
                   batch_size * sizeof(int), cudaMemcpyHostToDevice);
        
        // Set up kernel launch parameters
        dim3 grid(batch_size, num_heads, (max_seq_len + 255) / 256);
        dim3 block(256);
        
        // Calculate shared memory size needed
        size_t shared_mem_size = head_dim * sizeof(float);
        
        // Launch kernel
        optimizedPagedAttention<<<grid, block, shared_mem_size>>>(
            query,
            kv_cache->keys,
            kv_cache->values,
            d_page_table,
            output,
            d_seq_lengths,
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            1.0f / sqrtf(head_dim)
        );
        
        cudaDeviceSynchronize();
    }
};
```

## Common Pitfalls and Solutions

### 1. Page Table Management
- **Problem**: Complex page allocation and deallocation
- **Solution**: Implement efficient page management algorithms

### 2. Memory Access Patterns
- **Problem**: Indirect memory access causing performance issues
- **Solution**: Optimize for cache efficiency and coalescing

### 3. Synchronization Issues
- **Problem**: Race conditions in page allocation
- **Solution**: Proper synchronization mechanisms

### 4. Page Size Selection
- **Problem**: Suboptimal page sizes affecting performance
- **Solution**: Tune based on workload characteristics

## Performance Considerations

### Memory Bandwidth
- Paged access may increase memory bandwidth requirements
- Optimize page sizes for the target architecture

### Cache Efficiency
- Consider cache line alignment for page boundaries
- Minimize cache misses through proper design

### Page Size Trade-offs
- Larger pages: Less metadata overhead, more internal fragmentation
- Smaller pages: More metadata overhead, less internal fragmentation

### Parallelism
- Ensure efficient parallel access to different pages
- Minimize contention for page table access

## Real-World Applications

- **Large Language Models**: Serving models with long contexts
- **Variable-Length Sequences**: Efficient batching of different lengths
- **Memory-Constrained Environments**: Maximizing sequence length within memory limits
- **Continuous Generation**: Supporting ongoing text generation
- **Multi-Modal Models**: Handling variable-length inputs across modalities

## Advanced Techniques

### Adaptive Page Management
- Dynamically adjust page sizes based on access patterns
- Implement predictive page allocation

### Multi-GPU PagedAttention
- Distribute pages across multiple GPUs
- Manage cross-device page table coordination

### Compression Techniques
- Compress key-value pairs to increase effective page capacity
- Use quantization for memory efficiency

## Summary

PagedAttention revolutionizes transformer memory management by applying virtual memory concepts to KV-cache handling. This enables efficient processing of variable-length sequences, reduces memory fragmentation, and allows for longer context lengths. Understanding PagedAttention is crucial for implementing scalable transformer models with efficient memory utilization.