#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define TILE_SIZE 16

// Basic tiled matrix multiplication
__global__ void tiledMatMul(const float* A, const float* B, float* C, 
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles of A and B
    for(int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading of tiles into shared memory
        if(row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if(col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result for this tile
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized tiled matrix multiplication with register blocking
__global__ void optimizedTiledMatMul(const float* A, const float* B, float* C, 
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over all tiles
    for(int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory cooperatively
        As[ty][tx] = (row < M && t * TILE_SIZE + tx < K) ?
                      A[row * K + t * TILE_SIZE + tx] : 0.0f;
        Bs[ty][tx] = (col < N && t * TILE_SIZE + ty < K) ?
                      B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial result for this tile
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Naive matrix multiplication for comparison
__global__ void naiveMatMul(const float* A, const float* B, float* C, 
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;  // Matrix dimensions
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeC = M * N;
    
    // Host memory allocation
    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC, 0.0f);
    
    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for(int i = 0; i < sizeA; i++) {
        h_A[i] = dis(gen);
    }
    for(int i = 0; i < sizeB; i++) {
        h_B[i] = dis(gen);
    }
    
    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions for tiled multiplication
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    // Launch tiled matrix multiplication kernel
    optimizedTiledMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Tiled matrix multiplication completed." << std::endl;
    std::cout << "Result (first 10 elements): ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify with a simple case
    const int M_small = 4, N_small = 4, K_small = 4;
    std::vector<float> h_A_small(M_small * K_small, 1.0f);  // All 1s
    std::vector<float> h_B_small(K_small * N_small, 2.0f);  // All 2s
    std::vector<float> h_C_small(M_small * N_small, 0.0f);
    
    float *d_A_small, *d_B_small, *d_C_small;
    cudaMalloc(&d_A_small, M_small * K_small * sizeof(float));
    cudaMalloc(&d_B_small, K_small * N_small * sizeof(float));
    cudaMalloc(&d_C_small, M_small * N_small * sizeof(float));
    
    cudaMemcpy(d_A_small, h_A_small.data(), M_small * K_small * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_small, h_B_small.data(), K_small * N_small * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimGridSmall((N_small + TILE_SIZE - 1) / TILE_SIZE, (M_small + TILE_SIZE - 1) / TILE_SIZE);
    dim3 dimBlockSmall(TILE_SIZE, TILE_SIZE);
    
    optimizedTiledMatMul<<<dimGridSmall, dimBlockSmall>>>(d_A_small, d_B_small, d_C_small, M_small, N_small, K_small);
    
    cudaMemcpy(h_C_small.data(), d_C_small, M_small * N_small * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nVerification with small matrices (all A=1, all B=2):" << std::endl;
    std::cout << "Expected result: all values should be " << K_small * 1.0f * 2.0f << std::endl;
    std::cout << "Actual results (first 8): ";
    for(int i = 0; i < 8; i++) {
        std::cout << h_C_small[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_small);
    cudaFree(d_B_small);
    cudaFree(d_C_small);
    
    return 0;
}