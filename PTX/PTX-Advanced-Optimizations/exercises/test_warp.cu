#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

int main() {
    // Host arrays
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)i;
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
    
    cuModuleLoad(&module, "warp_coop.ptx");
    cuModuleGetFunction(&function, module, "warp_coop");
    
    // Set up kernel parameters
    void* args[] = {&d_input, &d_output, &N};
    
    // Launch kernel
    cuLaunchKernel(function, (N + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("Warp cooperation results (first 10 elements):\n");
    for(int i = 0; i < 10; i++) {
        printf("%.1f -> %.1f\n", h_input[i], h_output[i]);
    }
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}