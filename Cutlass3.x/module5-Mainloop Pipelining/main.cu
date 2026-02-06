#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

// CuTe headers
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

using namespace cute;

/**
 * MODULE 5: Mainloop Pipelining - Temporal Overlap & Throughput
 * 
 * FIRST PRINCIPLES EXPLANATION:
 * 
 * Mainloop pipelining is the heart of high-performance GEMM kernels. The concept involves
 * overlapping memory loads with computation to hide memory latency. In a "double-buffered"
 * approach, while the current tile is being computed, the next tile is being loaded into
 * shared memory. This creates a pipeline where memory operations and compute operations
 * happen concurrently.
 * 
 * KEY CONCEPTS:
 * 1. Double Buffering: Two buffers (A and B) alternate between loading and computing
 * 2. Pipeline Stages: Load -> Compute -> Store (overlapping stages)
 * 3. Temporal Overlap: Current computation overlaps with next tile loading
 * 4. Throughput Optimization: Maximizes utilization of compute and memory bandwidth
 * 
 * The pipeline typically follows this pattern:
 * - Stage 0: Load tiles for iteration 0 into buffer 0
 * - Stage 1: Load tiles for iteration 1 into buffer 1, compute tiles from buffer 0
 * - Stage 2+: Alternate loading into one buffer while computing from the other
 */

template <
    class ElementA,
    class ElementB,
    class ElementC
>
__global__ void double_buffered_gemm_kernel(
    ElementA const* A_ptr,  // Input matrix A (M x K)
    ElementB const* B_ptr,  // Input matrix B (K x N) 
    ElementC* C_ptr,        // Output matrix C (M x N)
    int M, int N, int K,
    int lda, int ldb, int ldc,
    ElementC alpha,
    ElementC beta
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
    ElementC accumulator[TILE_M][TILE_N];
    
    // Initialize accumulator to zero
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            accumulator[i][j] = ElementC(0);
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
                            static_cast<ElementC>(prev_buffer_A[m * TILE_K + k]) * 
                            static_cast<ElementC>(prev_buffer_B[k * TILE_N + n]);
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
                    static_cast<ElementC>(final_buffer_A[m * TILE_K + k]) * 
                    static_cast<ElementC>(final_buffer_B[k * TILE_N + n]);
            }
        }
    }
    
    // Store results back to global memory
    for (int m = 0; m < TILE_M && start_m + m < M; ++m) {
        for (int n = 0; n < TILE_N && start_n + n < N; ++n) {
            int global_idx = (start_m + m) * ldc + (start_n + n);
            C_ptr[global_idx] = alpha * accumulator[m][n] + beta * C_ptr[global_idx];
        }
    }
}

/**
 * Host wrapper function to launch the kernel
 */
template<typename ElementA, typename ElementB, typename ElementC>
cudaError_t launch_double_buffered_gemm(
    ElementA const* A,
    ElementB const* B, 
    ElementC* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    ElementC alpha = ElementC(1),
    ElementC beta = ElementC(0)
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
    double_buffered_gemm_kernel<<<grid, block, smem_size>>>(A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    
    return cudaGetLastError();
}

/**
 * Helper function to initialize matrices with sample data
 */
void initialize_matrices(std::vector<half_t>& A, std::vector<half_t>& B, int M, int N, int K) {
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<half_t>(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<half_t>(static_cast<float>(rand()) / RAND_MAX);
    }
}

int main() {
    // Problem dimensions (small for testing)
    const int M = 1024, N = 1024, K = 1024;
    const int lda = M, ldb = K, ldc = M;
    
    // Allocate host matrices
    std::vector<half_t> h_A(M * K);
    std::vector<half_t> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    // Initialize matrices
    initialize_matrices(h_A, h_B, M, N, K);
    
    // Allocate device matrices
    half_t* d_A;
    half_t* d_B;
    float* d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half_t));
    cudaMalloc(&d_B, K * N * sizeof(half_t));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up run
    launch_double_buffered_gemm(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
    cudaDeviceSynchronize();
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch the double-buffered GEMM kernel
    cudaError_t err = launch_double_buffered_gemm(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
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
    
    std::cout << "Double-buffered GEMM completed successfully!" << std::endl;
    std::cout << "Problem size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // Calculate GFLOPs
    float gflops = (2.0f * M * N * K) / (milliseconds / 1000.0f) / 1e9f;
    std::cout << "Performance: " << gflops << " GFLOPs" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}