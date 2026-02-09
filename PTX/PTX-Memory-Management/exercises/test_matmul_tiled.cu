#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define N 1024

int main() {
    // Host matrices
    float *h_a = (float*)malloc(N * N * sizeof(float));
    float *h_b = (float*)malloc(N * N * sizeof(float));
    float *h_c = (float*)malloc(N * N * sizeof(float));
    
    // Initialize matrices
    for(int i = 0; i < N * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device matrices
    float *d_a, *d_b, *d_c;
    
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    cudaMalloc((void**)&d_c, N * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load PTX and get function
    CUmodule module;
    CUfunction function;
    
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    
    cuModuleLoad(&module, "matmul_tiled.ptx");
    cuModuleGetFunction(&function, module, "matmul_tiled");
    
    // Calculate grid and block dimensions
    int dim = N / TILE_SIZE;
    if (N % TILE_SIZE) dim++;
    
    // Set up kernel parameters
    void* args[] = {&d_a, &d_b, &d_c, &N};
    
    // Launch kernel
    cuLaunchKernel(function, dim, dim, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, 0);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("First few results of matrix multiplication:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%.1f ", h_c[i*N+j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}