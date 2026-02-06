# Z-Curve / Morton Order: Space-Filling Curves for GPU Memory Optimization

## Overview

Z-Curve (also known as Morton Order) is a space-filling curve that maps multi-dimensional data to one dimension while preserving spatial locality. In GPU computing, Morton ordering is used to improve memory access patterns by ensuring that spatially close elements in multi-dimensional space are also close in memory, leading to better cache performance and memory coalescing.

## Why Z-Curve Ordering?

Traditional row-major or column-major storage of multi-dimensional arrays can lead to poor memory access patterns when traversing data in certain ways. Z-Curve ordering addresses this by:
- Preserving spatial locality in memory layout
- Improving cache hit rates
- Enhancing memory coalescing on GPUs
- Reducing memory bandwidth requirements

## Key Concepts

### Space-Filling Curves
Space-filling curves are continuous curves that pass through every point in a multi-dimensional space. The Z-Curve is a discrete approximation that visits all points in a grid in a specific order.

### Morton Number
A Morton number is created by interleaving the bits of coordinate values. For 2D coordinates (x, y), the Morton number is formed by alternating bits from x and y.

### Spatial Locality
Elements that are close in multi-dimensional space tend to be stored close in memory with Z-Curve ordering, improving cache performance.

## Z-Curve Properties

### Bit Interleaving
For 2D coordinates (x, y):
```
x = x₃x₂x₁x₀
y = y₃y₂y₁y₀
Morton = y₃x₃y₂x₂y₁x₁y₀x₀
```

### Recursive Structure
The Z-Curve has a recursive structure, dividing space into quadrants and recursively applying the same pattern.

### Cache Performance
Adjacent elements in Z-order are more likely to be in the same cache line than in row-major order.

## Step-by-Step Implementation Guide

### Step 1: Basic Morton Encoding
```cpp
// Encode 2D coordinates to Morton number
__device__ __host__ unsigned int mortonEncode(unsigned int x, unsigned int y) {
    unsigned int answer = 0;
    
    for(unsigned int i = 0; i < sizeof(unsigned int) * 4; i++) {
        answer |= ((x & (1U << i)) << i) | ((y & (1U << i)) << (i + 1));
    }
    
    return answer;
}

// More efficient bit interleaving using magic masks
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
```

### Step 2: Morton Decoding
```cpp
// Decode Morton number back to 2D coordinates
__device__ __host__ void mortonDecode(unsigned int morton, unsigned int& x, unsigned int& y) {
    x = 0;
    y = 0;
    
    for(unsigned int i = 0; i < sizeof(unsigned int) * 4; i++) {
        x |= (morton & (1U << (2 * i))) >> i;
        y |= (morton & (1U << (2 * i + 1))) >> (i + 1);
    }
}

// Fast decoding using magic masks
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
```

### Step 3: Z-Curve Array Layout
```cpp
// Function to convert 2D index to Z-curve linear index
__device__ __host__ unsigned int zCurveIndex(unsigned int x, unsigned int y, unsigned int width) {
    // For a 2D array of size width x height, encode coordinates
    return mortonEncodeFast(x, y);
}

// Function to convert Z-curve linear index back to 2D coordinates
__device__ __host__ void zCurveCoords(unsigned int mortonIndex, unsigned int& x, unsigned int& y) {
    mortonDecodeFast(mortonIndex, x, y);
}
```

### Step 4: GPU Kernel Using Z-Curve Ordering
```cpp
// Kernel that processes data in Z-curve order
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
```

### Step 5: Optimized Z-Curve Memory Access
```cpp
// Kernel that benefits from Z-curve memory layout
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
```

### Step 6: Complete Z-Curve Implementation with Utilities
```cpp
#include <cuda_runtime.h>
#include <iostream>

class ZCurveUtils {
public:
    // Host function to reorder array to Z-curve order
    static void reorderToZCurve(float* input, float* output, unsigned int width, unsigned int height) {
        // Create temporary array to hold reordered data
        float* temp = new float[width * height];
        
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
        
        delete[] temp;
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
    
private:
    // Fast Morton encoding (repeated from above)
    static unsigned int mortonEncodeFast(unsigned int x, unsigned int y) {
        x &= 0x0000ffff;
        y &= 0x0000ffff;
        
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;
        
        y = (y | (y << 8)) & 0x00FF00FF;
        y = (y | (y << 4)) & 0x0F0F0F0F;
        y = (y | (y << 2)) & 0x33333333;
        y = (y | (y << 1)) & 0x55555555;
        
        return x | (y << 1);
    }
    
    // Fast Morton decoding (repeated from above)
    static void mortonDecodeFast(unsigned int morton, unsigned int& x, unsigned int& y) {
        x = morton & 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        
        y = (y | (y >> 1)) & 0x55555555;
        y = (y | (y >> 2)) & 0x33333333;
        y = (y | (y >> 4)) & 0x0F0F0F0F;
        y = (y | (y >> 8)) & 0x00FF00FF;
        y = (y | (y >> 8)) & 0x0000FFFF;
    }
};

// GPU kernel that demonstrates Z-curve benefits
__global__ void zCurveConvolution(float* zOrderedData, float* result, 
                                 unsigned int width, unsigned int height) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < width * height) {
        unsigned int x, y;
        ZCurveUtils::mortonDecodeFast(tid, x, y);
        
        if(x > 0 && x < width-1 && y > 0 && y < height-1) {
            // 3x3 convolution kernel
            float sum = 0.0f;
            
            // Process 3x3 neighborhood
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    unsigned int nx = x + dx;
                    unsigned int ny = y + dy;
                    
                    unsigned int neighborMorton = ZCurveUtils::mortonEncodeFast(nx, ny);
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
```

## Common Pitfalls and Solutions

### 1. Bit Width Limitations
- **Problem**: Morton encoding limited by integer size
- **Solution**: Use larger integer types or handle overflow

### 2. Memory Layout Conversion Cost
- **Problem**: Converting to/from Z-curve order is expensive
- **Solution**: Only convert if processing benefits outweigh conversion cost

### 3. Bounds Checking
- **Problem**: Morton indices may exceed array bounds
- **Solution**: Proper bounds checking in kernels

### 4. Cache Line Alignment
- **Problem**: Z-curve elements may not align with cache lines
- **Solution**: Consider cache line size in optimization

## Performance Considerations

### Cache Performance
- Z-curve ordering improves spatial locality
- Better cache hit rates for neighborhood operations

### Memory Bandwidth
- Reduced memory bandwidth for certain access patterns
- Better coalescing for specific algorithms

### Computational Overhead
- Encoding/decoding has computational cost
- Benefit must outweigh encoding overhead

### Data Size
- More beneficial for larger datasets
- Overhead may not justify benefit for small arrays

## Real-World Applications

- **Image Processing**: Convolution, filtering operations
- **Scientific Computing**: N-body simulations, spatial queries
- **Databases**: Spatial indexing and range queries
- **Graphics**: Texture mapping, spatial data structures
- **Machine Learning**: Spatial feature processing

## Advanced Techniques

### 3D Morton Ordering
Extend to three dimensions by interleaving bits from x, y, and z coordinates.

### Adaptive Ordering
Switch between different ordering schemes based on access patterns.

### Hierarchical Z-Curves
Use multiple levels of Z-curve ordering for very large datasets.

## Summary

Z-Curve ordering is a powerful technique for improving memory access patterns in GPU computing by preserving spatial locality. By mapping multi-dimensional data to one dimension in a way that maintains proximity, we can achieve better cache performance and memory coalescing. Understanding when and how to apply Z-curve ordering is essential for optimizing GPU kernels that operate on multi-dimensional data with spatial relationships.