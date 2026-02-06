#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Fast Morton encoding using bit manipulation
__device__ __host__ unsigned int mortonEncodeFast(unsigned int x, unsigned int y) {
    x &= 0x0000ffff;  // Mask to lower 16 bits
    y &= 0x0000ffff;  // Mask to lower 16 bits
    
    // Spread x bits
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    
    // Spread y bits
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    
    // Interleave x and y
    return x | (y << 1);
}

// Fast Morton decoding using bit manipulation
__device__ __host__ void mortonDecodeFast(unsigned int morton, unsigned int& x, unsigned int& y) {
    // Extract x bits (even positions)
    x = morton & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    
    // Extract y bits (odd positions)
    y = (morton >> 1) & 0x55555555;
    y = (y | (y >> 1)) & 0x33333333;
    y = (y | (y >> 2)) & 0x33333333;
    y = (y | (y >> 4)) & 0x0F0F0F0F;
    y = (y | (y >> 8)) & 0x00FF00FF;
    y = (y | (y >> 8)) & 0x0000FFFF;
}

// Class to encapsulate Z-Curve utilities
class ZCurveUtils {
public:
    // Host function to reorder array to Z-curve order
    static void reorderToZCurve(float* input, float* output, unsigned int width, unsigned int height) {
        // Create temporary array to hold reordered data
        std::vector<float> temp(width * height);
        
        // Process each position in Z-curve order
        for(unsigned int mortonIdx = 0; mortonIdx < width * height; mortonIdx++) {
            unsigned int x, y;
            mortonDecodeFast(mortonIdx, x, y);
            
            if(x < width && y < height) {
                unsigned int originalIdx = y * width + x;
                temp[mortonIdx] = input[originalIdx];
            }
        }
        
        // Copy to output
        for(unsigned int i = 0; i < width * height; i++) {
            output[i] = temp[i];
        }
    }
    
    // Host function to reorder from Z-curve back to original order
    static void reorderFromZCurve(float* input, float* output, unsigned int width, unsigned int height) {
        // Process each position in Z-curve order
        for(unsigned int mortonIdx = 0; mortonIdx < width * height; mortonIdx++) {
            unsigned int x, y;
            mortonDecodeFast(mortonIdx, x, y);
            
            if(x < width && y < height) {
                unsigned int originalIdx = y * width + x;
                output[originalIdx] = input[mortonIdx];
            }
        }
    }
};

// GPU kernel that processes data in Z-curve order
__global__ void zCurveProcessing(float* input, float* output, unsigned int width, unsigned int height) {
    // Calculate thread's position in Z-curve order
    unsigned int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Decode to 2D coordinates
    unsigned int x, y;
    mortonDecodeFast(linearIdx, x, y);
    
    // Check bounds
    if(x < width && y < height) {
        // Process element at (x, y)
        unsigned int originalIdx = y * width + x;
        output[linearIdx] = input[originalIdx] * 2.0f;  // Example processing
    }
}

// GPU kernel that benefits from Z-curve memory layout
__global__ void optimizedZCurveKernel(float* zOrderedData, float* result, 
                                     unsigned int width, unsigned int height) {
    // Each thread processes multiple elements in Z-curve order
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    
    // Process elements in Z-curve order with stride
    for(unsigned int mortonIdx = tid; mortonIdx < width * height; mortonIdx += stride) {
        unsigned int x, y;
        mortonDecodeFast(mortonIdx, x, y);
        
        if(x < width && y < height) {
            // Access neighbors in Z-curve order for better cache performance
            float center = zOrderedData[mortonIdx];
            float sum = center;
            
            // Access neighbors (if they exist in bounds)
            if(x > 0) {
                unsigned int leftMorton = mortonEncodeFast(x-1, y);
                if(leftMorton < width * height) {
                    sum += zOrderedData[leftMorton];
                }
            }
            
            if(x < width-1) {
                unsigned int rightMorton = mortonEncodeFast(x+1, y);
                if(rightMorton < width * height) {
                    sum += zOrderedData[rightMorton];
                }
            }
            
            if(y > 0) {
                unsigned int upMorton = mortonEncodeFast(x, y-1);
                if(upMorton < width * height) {
                    sum += zOrderedData[upMorton];
                }
            }
            
            if(y < height-1) {
                unsigned int downMorton = mortonEncodeFast(x, y+1);
                if(downMorton < width * height) {
                    sum += zOrderedData[downMorton];
                }
            }
            
            result[mortonIdx] = sum / 5.0f;  // Average with neighbors
        }
    }
}

// GPU kernel for Z-curve convolution
__global__ void zCurveConvolution(float* zOrderedData, float* result, 
                                 unsigned int width, unsigned int height) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < width * height) {
        unsigned int x, y;
        mortonDecodeFast(tid, x, y);
        
        if(x > 0 && x < width-1 && y > 0 && y < height-1) {
            // 3x3 convolution kernel
            float sum = 0.0f;
            
            // Process 3x3 neighborhood
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    unsigned int nx = x + dx;
                    unsigned int ny = y + dy;
                    
                    unsigned int neighborMorton = mortonEncodeFast(nx, ny);
                    if(neighborMorton < width * height) {
                        sum += zOrderedData[neighborMorton];
                    }
                }
            }
            
            result[tid] = sum / 9.0f;  // Normalize
        } else {
            // Boundary handling
            result[tid] = zOrderedData[tid];
        }
    }
}

int main() {
    const unsigned int WIDTH = 64;
    const unsigned int HEIGHT = 64;
    const unsigned int TOTAL_SIZE = WIDTH * HEIGHT;
    
    // Host memory allocation
    std::vector<float> h_original(TOTAL_SIZE);
    std::vector<float> h_z_ordered(TOTAL_SIZE);
    std::vector<float> h_result(TOTAL_SIZE);
    
    // Initialize original data with a pattern
    for(unsigned int i = 0; i < HEIGHT; i++) {
        for(unsigned int j = 0; j < WIDTH; j++) {
            h_original[i * WIDTH + j] = i * WIDTH + j;
        }
    }
    
    // Reorder to Z-curve
    ZCurveUtils::reorderToZCurve(h_original.data(), h_z_ordered.data(), WIDTH, HEIGHT);
    
    // Device memory allocation
    float *d_original, *d_z_ordered, *d_result;
    cudaMalloc(&d_original, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_z_ordered, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_result, TOTAL_SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_original, h_original.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z_ordered, h_z_ordered.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch Z-curve processing kernel
    int blockSize = 256;
    int gridSize = (TOTAL_SIZE + blockSize - 1) / blockSize;
    
    zCurveProcessing<<<gridSize, blockSize>>>(d_original, d_result, WIDTH, HEIGHT);
    
    // Copy result back
    cudaMemcpy(h_result.data(), d_result, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Z-Curve processing completed." << std::endl;
    std::cout << "First 10 elements of result: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Test optimized Z-curve kernel
    optimizedZCurveKernel<<<gridSize, blockSize>>>(d_z_ordered, d_result, WIDTH, HEIGHT);
    
    cudaMemcpy(h_result.data(), d_result, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Optimized Z-curve kernel completed." << std::endl;
    std::cout << "First 10 elements after neighborhood processing: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Test Z-curve convolution
    zCurveConvolution<<<gridSize, blockSize>>>(d_z_ordered, d_result, WIDTH, HEIGHT);
    
    cudaMemcpy(h_result.data(), d_result, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Z-curve convolution completed." << std::endl;
    std::cout << "First 10 elements after convolution: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify Morton encoding/decoding
    std::cout << "\nVerifying Morton encoding/decoding:" << std::endl;
    bool encoding_correct = true;
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            unsigned int encoded = mortonEncodeFast(i, j);
            unsigned int decoded_x, decoded_y;
            mortonDecodeFast(encoded, decoded_x, decoded_y);
            
            if(decoded_x != i || decoded_y != j) {
                encoding_correct = false;
                std::cout << "Mismatch at (" << i << "," << j << "): encoded=" << encoded 
                          << ", decoded=(" << decoded_x << "," << decoded_y << ")" << std::endl;
            }
        }
    }
    
    std::cout << "Morton encoding/decoding " << (encoding_correct ? "correct" : "incorrect") << std::endl;
    
    // Cleanup
    cudaFree(d_original);
    cudaFree(d_z_ordered);
    cudaFree(d_result);
    
    return 0;
}