#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

int main() {
    // Host arrays
    float h_a[N], h_b[N], h_c[N];
    
    // Initialize host arrays
    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Device arrays
    float *d_a, *d_b, *d_c;
    
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load PTX and get function
    CUmodule module;
    CUfunction function;
    
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    
    cuModuleLoad(&module, "vector_add.ptx");
    cuModuleGetFunction(&function, module, "vector_add");
    
    // Set up kernel parameters
    void* args[] = {&d_a, &d_b, &d_c, &N};
    
    // Launch kernel with 1 block of N threads
    cuLaunchKernel(function, 1, 1, 1, N, 1, 1, 0, 0, args, 0);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("First 10 results:\n");
    for(int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}