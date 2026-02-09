#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

int main() {
    // Host arrays
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input with values that will show activation function effect
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)(i - N/2) / 10.0f;  // Range from -50 to 50
    }
    
    // Device arrays
    float *d_input, *d_output;
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load PTX and get function
    CUmodule module;
    CUfunction function;
    
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    
    cuModuleLoad(&module, "activation_kernel.ptx");
    cuModuleGetFunction(&function, module, "custom_activation");
    
    // Set up kernel parameters
    void* args[] = {&d_input, &d_output, &N};
    
    // Launch kernel with enough blocks to cover all elements
    int num_blocks = (N + 255) / 256;
    cuLaunchKernel(function, num_blocks, 1, 1, 256, 1, 1, 0, 0, args, 0);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("Activation function results (first 10 elements):\n");
    printf("Input\tOutput\tFunction\n");
    for(int i = 0; i < 10; i++) {
        printf("%.2f\t%.2f\tSwish\n", h_input[i], h_output[i]);
    }
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}