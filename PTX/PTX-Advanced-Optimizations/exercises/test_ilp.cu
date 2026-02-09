#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

int main() {
    // Host arrays
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output_naive = (float*)malloc(N * sizeof(float));
    float *h_output_opt = (float*)malloc(N * sizeof(float));
    
    // Initialize input
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    // Device arrays
    float *d_input, *d_output_naive, *d_output_opt;
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output_naive, N * sizeof(float));
    cudaMalloc((void**)&d_output_opt, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load PTX and get functions
    CUmodule module;
    CUfunction func_naive, func_opt;
    
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    
    cuModuleLoad(&module, "compute_bound.ptx");
    cuModuleGetFunction(&func_naive, module, "compute_bound");
    
    // Reload module for optimized kernel
    cuModuleUnload(module);
    cuModuleLoad(&module, "ilp_optimized.ptx");
    cuModuleGetFunction(&func_opt, module, "ilp_optimized");
    
    // Set up kernel parameters
    void* args_naive[] = {&d_input, &d_output_naive, &N};
    void* args_opt[] = {&d_input, &d_output_opt, &N};
    
    // Launch naive kernel
    cuLaunchKernel(func_naive, N, 1, 1, 1, 1, 1, 0, 0, args_naive, 0);
    
    // Launch optimized kernel
    cuLaunchKernel(func_opt, N, 1, 1, 1, 1, 1, 0, 0, args_opt, 0);
    
    // Copy results back to host
    cudaMemcpy(h_output_naive, d_output_naive, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_opt, d_output_opt, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("ILP optimization results (first 5 elements):\n");
    printf("Input\tNaive\tOptimized\n");
    for(int i = 0; i < 5; i++) {
        printf("%.1f\t%.1f\t%.1f\n", h_input[i], h_output_naive[i], h_output_opt[i]);
    }
    
    // Cleanup
    free(h_input); free(h_output_naive); free(h_output_opt);
    cudaFree(d_input); cudaFree(d_output_naive); cudaFree(d_output_opt);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}