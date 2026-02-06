#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Simple processing kernel
__global__ void processingKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        // Example processing: multiply by 2 and add 1
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// Class to demonstrate double buffering
class DoubleBufferPipeline {
private:
    float *d_buffer1, *d_buffer2;
    float *h_pinned1, *h_pinned2;
    cudaStream_t stream1, stream2;
    int buffer_size;
    bool current_buffer; // true for buffer1, false for buffer2

public:
    DoubleBufferPipeline(int size) : buffer_size(size), current_buffer(true) {
        // Allocate device memory for both buffers
        cudaMalloc(&d_buffer1, buffer_size * sizeof(float));
        cudaMalloc(&d_buffer2, buffer_size * sizeof(float));
        
        // Allocate pinned host memory for both buffers
        cudaMallocHost(&h_pinned1, buffer_size * sizeof(float));
        cudaMallocHost(&h_pinned2, buffer_size * sizeof(float));
        
        // Create streams
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
    }
    
    ~DoubleBufferPipeline() {
        cudaFree(d_buffer1);
        cudaFree(d_buffer2);
        cudaFreeHost(h_pinned1);
        cudaFreeHost(h_pinned2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
    
    void processChunk(float* input_chunk, float* output_chunk, int chunk_size) {
        // Select current and next buffers
        float *current_d_buffer = current_buffer ? d_buffer1 : d_buffer2;
        float *current_h_pinned = current_buffer ? h_pinned1 : h_pinned2;
        float *next_d_buffer = current_buffer ? d_buffer2 : d_buffer1;
        float *next_h_pinned = current_buffer ? h_pinned2 : h_pinned1;
        
        cudaStream_t current_stream = current_buffer ? stream1 : stream2;
        cudaStream_t next_stream = current_buffer ? stream2 : stream1;
        
        // Copy input to current buffer asynchronously
        cudaMemcpyAsync(current_d_buffer, input_chunk, 
                        chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, current_stream);
        
        // Process current buffer
        processingKernel<<<(chunk_size + 255) / 256, 256, 0, current_stream>>>(
            current_d_buffer, current_d_buffer, chunk_size);
        
        // Copy result back to host asynchronously
        cudaMemcpyAsync(output_chunk, current_d_buffer,
                        chunk_size * sizeof(float),
                        cudaMemcpyDeviceToHost, current_stream);
        
        // Update buffer selection for next iteration
        current_buffer = !current_buffer;
    }
    
    void synchronize() {
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    }
};

// Advanced pipelined processing function
void advancedPipelinedProcessing(float* h_input, float* h_output, 
                               int total_elements, int chunk_size) {
    // Calculate number of chunks
    int num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    
    // Allocate device memory for two chunks (double buffering)
    float *d_input1, *d_input2, *d_output1, *d_output2;
    cudaMalloc(&d_input1, chunk_size * sizeof(float));
    cudaMalloc(&d_input2, chunk_size * sizeof(float));
    cudaMalloc(&d_output1, chunk_size * sizeof(float));
    cudaMalloc(&d_output2, chunk_size * sizeof(float));
    
    // Allocate pinned host memory for faster transfers
    float *h_pinned_input1, *h_pinned_input2;
    float *h_pinned_output1, *h_pinned_output2;
    cudaMallocHost(&h_pinned_input1, chunk_size * sizeof(float));
    cudaMallocHost(&h_pinned_input2, chunk_size * sizeof(float));
    cudaMallocHost(&h_pinned_output1, chunk_size * sizeof(float));
    cudaMallocHost(&h_pinned_output2, chunk_size * sizeof(float));
    
    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Initialize first chunk transfer
    if(num_chunks > 0) {
        // Copy first chunk to pinned memory
        int first_chunk_elements = std::min(chunk_size, 
                                          total_elements - 0);
        memcpy(h_pinned_input1, h_input, 
               first_chunk_elements * sizeof(float));
        
        // Start first async transfer
        cudaMemcpyAsync(d_input1, h_pinned_input1,
                        first_chunk_elements * sizeof(float),
                        cudaMemcpyHostToDevice, stream1);
    }
    
    // Pipeline processing
    for(int i = 0; i < num_chunks; i++) {
        // Determine current chunk
        int current_start = i * chunk_size;
        int current_elements = std::min(chunk_size, 
                                      total_elements - current_start);
        
        // Select buffers based on iteration
        float *d_in = (i % 2 == 0) ? d_input1 : d_input2;
        float *d_out = (i % 2 == 0) ? d_output1 : d_output2;
        float *h_pin_in = (i % 2 == 0) ? h_pinned_input1 : h_pinned_input2;
        float *h_pin_out = (i % 2 == 0) ? h_pinned_output1 : h_pinned_output2;
        cudaStream_t stream = (i % 2 == 0) ? stream1 : stream2;
        
        // Launch kernel on current data
        processingKernel<<<(current_elements + 255) / 256, 256, 0, stream>>>(
            d_in, d_out, current_elements);
        
        // Copy result back to host
        cudaMemcpyAsync(h_pin_out, d_out,
                        current_elements * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        
        // Prepare next chunk for transfer (prefetching)
        if(i + 1 < num_chunks) {
            int next_start = (i + 1) * chunk_size;
            int next_elements = std::min(chunk_size, 
                                       total_elements - next_start);
            
            float *next_h_pin_in = (i % 2 == 0) ? h_pinned_input2 : h_pinned_input1;
            cudaStream_t next_stream = (i % 2 == 0) ? stream2 : stream1;
            
            // Copy next chunk to pinned memory
            memcpy(next_h_pin_in, h_input + next_start,
                   next_elements * sizeof(float));
            
            // Start async transfer for next chunk
            cudaMemcpyAsync((i % 2 == 0) ? d_input2 : d_input1,
                            next_h_pin_in,
                            next_elements * sizeof(float),
                            cudaMemcpyHostToDevice, next_stream);
        }
        
        // Copy results back to original output array
        if(i > 0) { // Wait for previous iteration's result
            float *prev_h_pin_out = (i % 2 == 0) ? h_pinned_output2 : h_pinned_output1;
            int prev_start = (i - 1) * chunk_size;
            int prev_elements = std::min(chunk_size, 
                                       total_elements - prev_start);
            
            cudaMemcpyAsync(h_output + prev_start, prev_h_pin_out,
                            prev_elements * sizeof(float),
                            cudaMemcpyHostToDevice, 0); // Use default stream
        }
    }
    
    // Handle the last chunk's result
    if(num_chunks > 0) {
        int last_start = (num_chunks - 1) * chunk_size;
        int last_elements = std::min(chunk_size, 
                                   total_elements - last_start);
        float *last_h_pin_out = ((num_chunks - 1) % 2 == 0) ? 
                                 h_pinned_output1 : h_pinned_output2;
        
        cudaMemcpyAsync(h_output + last_start, last_h_pin_out,
                        last_elements * sizeof(float),
                        cudaMemcpyHostToDevice, 0);
    }
    
    // Wait for all operations to complete
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_input1); cudaFree(d_input2);
    cudaFree(d_output1); cudaFree(d_output2);
    cudaFreeHost(h_pinned_input1); cudaFreeHost(h_pinned_input2);
    cudaFreeHost(h_pinned_output1); cudaFreeHost(h_pinned_output2);
    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
}

int main() {
    const int TOTAL_ELEMENTS = 1024 * 1024;  // 1M elements
    const int CHUNK_SIZE = 64 * 1024;        // 64K elements per chunk
    const int NUM_CHUNKS = (TOTAL_ELEMENTS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Allocate host memory
    std::vector<float> h_input(TOTAL_ELEMENTS);
    std::vector<float> h_output(TOTAL_ELEMENTS);
    
    // Initialize input
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        h_input[i] = static_cast<float>(i % 1000);  // Some pattern
    }
    
    std::cout << "Starting double buffering and async copy pipelining demo..." << std::endl;
    std::cout << "Processing " << TOTAL_ELEMENTS << " elements in " << NUM_CHUNKS << " chunks" << std::endl;
    
    // Create pipeline
    DoubleBufferPipeline pipeline(CHUNK_SIZE);
    
    // Process chunks using double buffering
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < NUM_CHUNKS; i++) {
        int start_idx = i * CHUNK_SIZE;
        int current_size = std::min(CHUNK_SIZE, TOTAL_ELEMENTS - start_idx);
        
        pipeline.processChunk(&h_input[start_idx], &h_output[start_idx], current_size);
    }
    
    pipeline.synchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Double buffering completed in " << duration.count() << " ms" << std::endl;
    
    // Verify results (first few elements)
    std::cout << "Verification (first 10 elements):" << std::endl;
    std::cout << "Input:  ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify that the processing was correct (input * 2 + 1)
    bool is_correct = true;
    for(int i = 0; i < 10; i++) {
        if(abs(h_output[i] - (h_input[i] * 2.0f + 1.0f)) > 1e-5) {
            is_correct = false;
            break;
        }
    }
    
    std::cout << "Processing " << (is_correct ? "correct" : "incorrect") << std::endl;
    
    // Test advanced pipelined processing
    std::vector<float> h_input_adv(TOTAL_ELEMENTS);
    std::vector<float> h_output_adv(TOTAL_ELEMENTS);
    
    // Reinitialize input
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        h_input_adv[i] = static_cast<float>(i % 1000);
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    advancedPipelinedProcessing(h_input_adv.data(), h_output_adv.data(), 
                              TOTAL_ELEMENTS, CHUNK_SIZE);
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nAdvanced pipelined processing completed in " << duration.count() << " ms" << std::endl;
    
    // Verify advanced results
    std::cout << "Advanced processing verification (first 10 elements):" << std::endl;
    std::cout << "Input:  ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_input_adv[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output_adv[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify that the processing was correct
    is_correct = true;
    for(int i = 0; i < 10; i++) {
        if(abs(h_output_adv[i] - (h_input_adv[i] * 2.0f + 1.0f)) > 1e-5) {
            is_correct = false;
            break;
        }
    }
    
    std::cout << "Advanced processing " << (is_correct ? "correct" : "incorrect") << std::endl;
    
    return 0;
}