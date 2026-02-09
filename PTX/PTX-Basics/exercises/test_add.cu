#include <cuda_runtime.h>
#include <stdio.h>

// Host function to call PTX kernel
extern "C" void launch_add_kernel(unsigned int a, unsigned int b, unsigned int* result);

int main() {
    // Allocate device memory
    unsigned int *d_result;
    unsigned int h_result;
    
    cudaMalloc((void**)&d_result, sizeof(unsigned int));
    
    // Launch kernel
    launch_add_kernel(5, 7, d_result);
    
    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("Result: 5 + 7 = %u\n", h_result);
    
    // Cleanup
    cudaFree(d_result);
    
    return 0;
}