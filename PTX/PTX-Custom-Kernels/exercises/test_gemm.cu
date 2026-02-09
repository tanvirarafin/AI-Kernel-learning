#include <cuda_runtime.h>
#include <stdio.h>

#define M 256
#define N 256
#define K 256

int main() {
    // Host matrices
    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    float *h_c = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    for(int i = 0; i < M * K; i++) {
        h_a[i] = 1.0f;
    }
    for(int i = 0; i < K * N; i++) {
        h_b[i] = 2.0f;
    }
    
    // Device matrices
    float *d_a, *d_b, *d_c;
    
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Load PTX and get function
    CUmodule module;
    CUfunction function;
    
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    
    cuModuleLoad(&module, "gemm_kernel.ptx");
    cuModuleGetFunction(&function, module, "custom_gemm");
    
    // Calculate grid and block dimensions
    int dim_x = (N + 15) / 16;  // TILE_N = 16
    int dim_y = (M + 15) / 16;  // TILE_M = 16
    
    // Set up kernel parameters
    void* args[] = {&d_a, &d_b, &d_c, &M, &N, &K};
    
    // Launch kernel
    cuLaunchKernel(function, dim_x, dim_y, 1, 16, 16, 1, 0, 0, args, 0);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few results
    printf("GEMM results (first 5x5 submatrix):\n");
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