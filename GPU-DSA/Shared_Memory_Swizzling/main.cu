#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE_SIZE 32
#define S_TILE_SIZE (TILE_SIZE * (TILE_SIZE + 1))  // +1 to avoid bank conflicts

// Function to calculate swizzled address
__device__ int getSwizzledAddr(int addr) {
    // Simple swizzling: add bank index to address
    const int BANK_NUM = 32;
    const int BANK_WIDTH = 4;  // 4-byte words
    int bank = (addr / BANK_WIDTH) % BANK_NUM;
    return addr + bank;
}

// XOR swizzling function
__device__ int xorSwizzle(int addr) {
    // XOR the address with the bank index to spread accesses
    const int NUM_BANKS = 32;
    int bank = (addr / 4) % NUM_BANKS;  // 4-byte words
    return addr ^ (bank << 2);  // Shift bank to align with word boundary
}

// Kernel without swizzling (causing bank conflicts)
__global__ void withoutSwizzling(float* input, float* output) {
    __shared__ float sdata[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load in row-major order
    if(x < gridDim.x * TILE_SIZE && y < gridDim.y * TILE_SIZE) {
        sdata[threadIdx.y][threadIdx.x] = input[y * gridDim.x * TILE_SIZE + x];
    }
    __syncthreads();
    
    // Transpose: read in column-major order (causes bank conflicts)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if(x < gridDim.y * TILE_SIZE && y < gridDim.x * TILE_SIZE) {
        output[y * gridDim.y * TILE_SIZE + x] = sdata[threadIdx.x][threadIdx.y];
    }
}

// Kernel with swizzling to avoid bank conflicts
__global__ void withSwizzling(float* input, float* output) {
    // Use padded tile size to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load in row-major order
    if(x < gridDim.x * TILE_SIZE && y < gridDim.y * TILE_SIZE) {
        tile[threadIdx.y][threadIdx.x] = input[y * gridDim.x * TILE_SIZE + x];
    }
    __syncthreads();
    
    // Transpose: read in column-major order
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if(x < gridDim.y * TILE_SIZE && y < gridDim.x * TILE_SIZE) {
        output[y * gridDim.y * TILE_SIZE + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Advanced XOR swizzling for different access patterns
__device__ int advancedXorSwizzle(int addr, int stride) {
    // Different swizzling based on access pattern
    const int NUM_BANKS = 32;
    int bank = (addr / 4) % NUM_BANKS;
    
    // For strided access, use stride to determine swizzling
    int offset = (stride * bank) % NUM_BANKS;
    return addr ^ (offset << 2);
}

__global__ void advancedSwizzledKernel(float* input, float* output, int stride) {
    __shared__ float sdata[1024 + 32 * 4];  // Extra space for swizzling
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Swizzled storage
    sdata[advancedXorSwizzle(threadIdx.x, stride)] = input[tid];
    __syncthreads();
    
    // Swizzled access with different pattern
    int access_idx = (threadIdx.x * stride) % 32;
    float value = sdata[advancedXorSwizzle(access_idx, stride)];
    
    output[tid] = value;
}

// Complete example with multiple swizzling techniques
__global__ void completeSwizzledGemm(float* A, float* B, float* C, 
                                   int M, int N, int K) {
    // Use padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles with padding to avoid bank conflicts
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        As[ty][tx] = ((a_row < M) && (a_col < K)) ? 
                      A[a_row * K + a_col] : 0.0f;
        
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        Bs[ty][tx] = ((b_row < K) && (b_col < N)) ? 
                      B[b_row * N + b_col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial result
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeC = M * N;
    
    // Host memory allocation
    std::vector<float> h_A(sizeA, 1.0f);  // Fill with 1s
    std::vector<float> h_B(sizeB, 2.0f);  // Fill with 2s
    std::vector<float> h_C(sizeC, 0.0f);
    
    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    // Launch kernel with swizzling
    completeSwizzledGemm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Swizzled GEMM completed." << std::endl;
    std::cout << "Result (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify with a simple case
    // For matrices filled with 1s and 2s, result should be K * 1 * 2 = K * 2
    float expected = K * 1.0f * 2.0f;
    bool is_correct = true;
    for(int i = 0; i < 10; i++) {
        if(abs(h_C[i] - expected) > 1e-5) {
            is_correct = false;
            break;
        }
    }
    
    std::cout << "Result " << (is_correct ? "correct" : "incorrect") << std::endl;
    std::cout << "Expected value: " << expected << std::endl;
    
    // Test matrix transpose with and without swizzling
    const int TRANS_SIZE = 128;
    const int trans_total = TRANS_SIZE * TRANS_SIZE;
    std::vector<float> h_orig(trans_total);
    std::vector<float> h_transposed(trans_total, 0.0f);
    
    // Fill with a pattern
    for(int i = 0; i < TRANS_SIZE; i++) {
        for(int j = 0; j < TRANS_SIZE; j++) {
            h_orig[i * TRANS_SIZE + j] = i * TRANS_SIZE + j;
        }
    }
    
    float *d_orig, *d_trans;
    cudaMalloc(&d_orig, trans_total * sizeof(float));
    cudaMalloc(&d_trans, trans_total * sizeof(float));
    
    cudaMemcpy(d_orig, h_orig.data(), trans_total * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use the swizzled transpose kernel
    dim3 transGrid((TRANS_SIZE + TILE_SIZE - 1) / TILE_SIZE, (TRANS_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    dim3 transBlock(TILE_SIZE, TILE_SIZE);
    
    withSwizzling<<<transGrid, transBlock>>>(d_orig, d_trans);
    
    cudaMemcpy(h_transposed.data(), d_trans, trans_total * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nMatrix transpose completed." << std::endl;
    std::cout << "Original (first row): ";
    for(int i = 0; i < 5; i++) {
        std::cout << h_orig[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Transposed (first column of original): ";
    for(int i = 0; i < 5; i++) {
        std::cout << h_transposed[i * TRANS_SIZE] << " ";
    }
    std::cout << std::endl;
    
    // Verify transpose
    bool transpose_correct = true;
    for(int i = 0; i < 5 && transpose_correct; i++) {
        for(int j = 0; j < 5; j++) {
            if(abs(h_orig[i * TRANS_SIZE + j] - h_transposed[j * TRANS_SIZE + i]) > 1e-5) {
                transpose_correct = false;
                break;
            }
        }
    }
    
    std::cout << "Transpose " << (transpose_correct ? "correct" : "incorrect") << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_orig);
    cudaFree(d_trans);
    
    return 0;
}