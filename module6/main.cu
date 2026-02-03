#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"

// CuTe headers
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

using namespace cute;

/**
 * MODULE 6: Fused Epilogues - Functional Avoiding VRAM Roundtrips
 * 
 * FIRST PRINCIPLES EXPLANATION:
 * 
 * Fused epilogues are critical for performance in deep learning workloads. Rather than storing
 * intermediate results to global memory and then loading them again for post-processing operations
 * like bias-add and activation functions, we fuse these operations directly into the GEMM kernel.
 * This avoids expensive roundtrips to VRAM, significantly improving performance.
 * 
 * KEY CONCEPTS:
 * 1. Epilogue Fusion: Combining post-multiplication operations within the kernel
 * 2. Bias Addition: Adding per-channel bias values to the output
 * 3. Activation Functions: Applying functions like ReLU in-place
 * 4. Memory Efficiency: Eliminating intermediate memory accesses
 * 
 * The fused approach performs: C = activation(alpha * A * B + bias + beta * C)
 * All computations happen in registers/shared memory without VRAM roundtrips.
 */

template <
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementAccumulator = float,
    class ElementCompute = float
>
__global__ void fused_gemm_epilogue_kernel(
    ElementA const* A_ptr,      // Input matrix A (M x K)
    ElementB const* B_ptr,      // Input matrix B (K x N) 
    ElementC* C_ptr,            // Output matrix C (M x N)
    ElementC const* bias_ptr,   // Bias vector (N)
    int M, int N, int K,
    int lda, int ldb, int ldc,
    ElementCompute alpha,
    ElementCompute beta
) {
    // Get thread information
    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;
    int const bidy = blockIdx.y;
    
    // Define tile sizes (simplified for this example)
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    // Calculate which tile this thread block is responsible for
    int start_m = bidy * TILE_M;
    int start_n = bidx * TILE_N;
    
    // Shared memory for double buffering
    extern __shared__ char smem_char[];
    ElementA* smem_A = reinterpret_cast<ElementA*>(smem_char);
    ElementB* smem_B = reinterpret_cast<ElementB*>(smem_char + 2 * TILE_M * TILE_K * sizeof(ElementA));
    
    // Pointers to the two buffers for A and B
    ElementA* buffer_A0 = smem_A;
    ElementA* buffer_A1 = smem_A + TILE_M * TILE_K;
    ElementB* buffer_B0 = smem_B;
    ElementB* buffer_B1 = smem_B + TILE_K * TILE_N;
    
    // Accumulator register (simplified)
    ElementAccumulator accumulator[TILE_M][TILE_N];
    
    // Initialize accumulator to zero
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            accumulator[i][j] = ElementAccumulator(0);
        }
    }
    
    // Main pipelined loop with double buffering
    int buffer_idx = 0;
    
    // Pre-load first tile into buffer 0
    if (start_m < M && start_n < N) {
        // Load A tile into buffer 0
        for (int k = 0; k < TILE_K && start_m + k < M; ++k) {
            for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
                int global_idx = (start_m + m) * lda + k;
                if (k < K) {
                    buffer_A0[m * TILE_K + k] = A_ptr[global_idx];
                } else {
                    buffer_A0[m * TILE_K + k] = ElementA(0);
                }
            }
        }
        
        // Load B tile into buffer 0
        for (int k = 0; k < TILE_K && k < K; ++k) {
            for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
                int global_idx = k * ldb + (start_n + n);
                if (k < K) {
                    buffer_B0[k * TILE_N + n] = B_ptr[global_idx];
                } else {
                    buffer_B0[k * TILE_N + n] = ElementB(0);
                }
            }
        }
    }
    
    __syncthreads(); // Ensure first tile is loaded
    
    // Iterate through K dimension in chunks
    for (int k_step = 0; k_step < K; k_step += TILE_K) {
        // If not the first iteration, compute the previous buffer while loading current
        if (k_step > 0) {
            // Perform computation using the previous buffer
            ElementA* prev_buffer_A = (buffer_idx == 0) ? buffer_A1 : buffer_A0;
            ElementB* prev_buffer_B = (buffer_idx == 0) ? buffer_B1 : buffer_B0;
            
            // Perform matrix multiplication for the previous buffer
            for (int k = 0; k < TILE_K; ++k) {
                for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
                    for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
                        accumulator[m][n] += 
                            static_cast<ElementAccumulator>(prev_buffer_A[m * TILE_K + k]) * 
                            static_cast<ElementAccumulator>(prev_buffer_B[k * TILE_N + n]);
                    }
                }
            }
        }
        
        // Load next tile into the other buffer (if there is one)
        if (k_step + TILE_K < K) {
            ElementA* next_buffer_A = (buffer_idx == 0) ? buffer_A1 : buffer_A0;
            ElementB* next_buffer_B = (buffer_idx == 0) ? buffer_B1 : buffer_B0;
            
            // Calculate the next K position
            int next_k = k_step + TILE_K;
            
            // Load A tile into next buffer
            for (int k = 0; k < TILE_K && next_k + k < K; ++k) {
                for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
                    int global_idx = (start_m + m) * lda + (next_k + k);
                    if (next_k + k < K) {
                        next_buffer_A[m * TILE_K + k] = A_ptr[global_idx];
                    } else {
                        next_buffer_A[m * TILE_K + k] = ElementA(0);
                    }
                }
            }
            
            // Load B tile into next buffer
            for (int k = 0; k < TILE_K && next_k + k < K; ++k) {
                for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
                    int global_idx = (next_k + k) * ldb + (start_n + n);
                    if (next_k + k < K) {
                        next_buffer_B[k * TILE_N + n] = B_ptr[global_idx];
                    } else {
                        next_buffer_B[k * TILE_N + n] = ElementB(0);
                    }
                }
            }
        }
        
        // Toggle buffer index for next iteration
        buffer_idx = 1 - buffer_idx;
        
        __syncthreads(); // Synchronize before next iteration
    }
    
    // Process the final buffer after the loop (the last computation)
    ElementA* final_buffer_A = (buffer_idx == 0) ? buffer_A1 : buffer_A0;
    ElementB* final_buffer_B = (buffer_idx == 0) ? buffer_B1 : buffer_B0;
    
    // Perform matrix multiplication for the final buffer
    for (int k = 0; k < TILE_K; ++k) {
        for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
            for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
                accumulator[m][n] += 
                    static_cast<ElementAccumulator>(final_buffer_A[m * TILE_K + k]) * 
                    static_cast<ElementAccumulator>(final_buffer_B[k * TILE_N + n]);
            }
        }
    }
    
    // Apply fused epilogue: bias addition and ReLU activation
    for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
        for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
            // Linear combination: alpha * A * B + beta * C
            ElementAccumulator result = alpha * accumulator[m][n];
            
            // Add bias (if bias pointer is provided)
            if (bias_ptr != nullptr) {
                result += static_cast<ElementAccumulator>(bias_ptr[start_n + n]);
            }
            
            // Add previous C value scaled by beta (if beta != 0)
            if (beta != ElementCompute(0)) {
                int global_idx = (start_m + m) * ldc + (start_n + n);
                result += beta * static_cast<ElementAccumulator>(C_ptr[global_idx]);
            }
            
            // Apply ReLU activation: max(0, result)
            result = (result > ElementAccumulator(0)) ? result : ElementAccumulator(0);
            
            // Store final result
            int global_idx = (start_m + m) * ldc + (start_n + n);
            C_ptr[global_idx] = static_cast<ElementC>(result);
        }
    }
}

/**
 * Host wrapper function to launch the fused GEMM kernel with epilogue
 */
template<typename ElementA, typename ElementB, typename ElementC, typename ElementCompute = float>
cudaError_t launch_fused_gemm_epilogue(
    ElementA const* A,
    ElementB const* B,
    ElementC* C,
    ElementC const* bias,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0)
) {
    // Define thread block and grid dimensions
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int TILE_K = 32;
    
    dim3 block(256, 1, 1);  // 256 threads per block
    dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, 
              (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1);
    
    // Calculate shared memory size (two buffers for A and B)
    size_t smem_size = 2 * (BLOCK_SIZE_M * TILE_K * sizeof(ElementA) + 
                             TILE_K * BLOCK_SIZE_N * sizeof(ElementB));
    
    // Launch kernel
    fused_gemm_epilogue_kernel<<<grid, block, smem_size>>>(A, B, C, bias, M, N, K, lda, ldb, ldc, alpha, beta);
    
    return cudaGetLastError();
}

/**
 * Helper function to initialize matrices with sample data
 */
void initialize_matrices(std::vector<half_t>& A, std::vector<half_t>& B, std::vector<float>& bias, int M, int N, int K) {
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<half_t>(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<half_t>(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < N; ++i) {
        bias[i] = static_cast<float>(static_cast<float>(rand()) / RAND_MAX);
    }
}

int main() {
    // Problem dimensions (small for testing)
    const int M = 1024, N = 1024, K = 1024;
    const int lda = M, ldb = K, ldc = M;
    
    // Allocate host matrices
    std::vector<half_t> h_A(M * K);
    std::vector<half_t> h_B(K * N);
    std::vector<float> h_bias(N);  // Bias vector - same type as output
    std::vector<float> h_C(M * N, 0.0f);
    
    // Initialize matrices
    initialize_matrices(h_A, h_B, h_bias, M, N, K);
    
    // Allocate device matrices
    half_t* d_A;
    half_t* d_B;
    float* d_bias;  // Same type as output
    float* d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half_t));
    cudaMalloc(&d_B, K * N * sizeof(half_t));
    cudaMalloc(&d_bias, N * sizeof(float));  // Bias vector - same type as output
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice);  // Same type
    cudaMemset(d_C, 0, M * N * sizeof(float));
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up run
    launch_fused_gemm_epilogue(d_A, d_B, d_C, d_bias, M, N, K, lda, ldb, ldc);
    cudaDeviceSynchronize();
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch the fused GEMM kernel with epilogue
    cudaError_t err = launch_fused_gemm_epilogue(d_A, d_B, d_C, d_bias, M, N, K, lda, ldb, ldc);
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Fused GEMM with Bias-Add and ReLU completed successfully!" << std::endl;
    std::cout << "Problem size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // Calculate GFLOPs
    float gflops = (2.0f * M * N * K) / (milliseconds / 1000.0f) / 1e9f;
    std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}