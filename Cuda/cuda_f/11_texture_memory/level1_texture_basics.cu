/*
 * Texture Memory - Complete Exercise File
 * 
 * This file contains exercises for texture memory usage.
 * Complete the TODO sections to learn texture memory programming.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 512
#define HEIGHT 512

// Texture object
texture<float, 2, cudaReadModeElementType> texRef;

// Kernel using texture memory
__global__ void textureKernel(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // TODO: Fetch from texture memory
        // float val = tex2D(texRef, x, y);
        
        /* YOUR CODE HERE */
        
        output[y * width + x] = val;
    }
}

// Kernel with normalized coordinates
__global__ void normalizedTextureKernel(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // TODO: Use normalized coordinates (0.0 to 1.0)
        // float u = (x + 0.5f) / width;
        // float v = (y + 0.5f) / height;
        // float val = tex2D(texRef, u, v);
        
        /* YOUR CODE HERE */
        
        output[y * width + x] = val;
    }
}

// Kernel with linear interpolation
__global__ void interpolationKernel(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // TODO: Use fractional coordinates for interpolation
        // float u = (x + 0.5f) / width;
        // float v = (y + 0.5f) / height;
        // With linear filtering enabled, this interpolates
        
        /* YOUR CODE HERE */
        
        output[y * width + x] = val;
    }
}

void initImage(float *image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[y * width + x] = sinf(x * 0.1f) * cosf(y * 0.1f);
        }
    }
}

int main() {
    printf("=== Texture Memory Exercises ===\n\n");
    
    const int WIDTH = 512;
    const int HEIGHT = 512;
    size_t size = WIDTH * HEIGHT * sizeof(float);
    
    float *h_image = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    initImage(h_image, WIDTH, HEIGHT);
    
    // Create CUDA array for texture
    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT);
    
    // Copy data to array
    cudaMemcpy2DToArray(cuArray, 0, 0, h_image, WIDTH * sizeof(float),
                        WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);
    
    // TODO: Create texture object
    // cudaTextureDesc texDesc = {};
    // texDesc.addressMode[0] = cudaAddressModeClamp;
    // texDesc.addressMode[1] = cudaAddressModeClamp;
    // texDesc.filterMode = cudaFilterModePoint;  // or cudaFilterModeLinear
    // texDesc.readMode = cudaReadModeElementType;
    // texDesc.normalizedCoords = 0;
    
    // cudaResourceDesc resDesc = {};
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = cuArray;
    
    // cudaTextureObject_t texObj = 0;
    // cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    /* YOUR CODE HERE */
    
    float *d_output;
    cudaMalloc(&d_output, size);
    
    dim3 blockDim(32, 32);
    dim3 gridDim((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
    
    // Test: Texture kernel
    printf("Test: Texture memory access\n");
    textureKernel<<<gridDim, blockDim>>>(d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    printf("  ✓ Completed\n");
    
    // Cleanup
    // cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    free(h_image);
    free(h_output);
    
    printf("\n=== Key Takeaways ===\n");
    printf("- Texture memory provides cached reads with spatial locality\n");
    printf("- cudaArray is optimized for texture access\n");
    printf("- Normalized coordinates: 0.0 to 1.0 range\n");
    printf("- Linear filtering provides hardware interpolation\n");
    printf("- Address modes: Clamp, Wrap, Mirror\n");
    
    return 0;
}
