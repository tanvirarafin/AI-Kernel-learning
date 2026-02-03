#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

// CuTe includes
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_traits.hpp"

using namespace cute;

/*
 * Module 4: MMA Atoms - Spatial Tensor Core Mastery
 *
 * First Principles Explanation:
 * Tensor Cores are specialized processing units that perform matrix multiply-accumulate (MMA) operations
 * extremely efficiently. In CUTLASS 3.x, CuTe provides Mma_Atom abstractions that encapsulate the
 * hardware-specific MMA operations. These atoms define how data flows between registers and how
 * the tensor core operations are performed.
 *
 * Key Concepts:
 * - Mma_Atom: Represents a single MMA operation unit that maps to hardware tensor cores
 * - Shape: Defines the dimensions of the MMA operation (M, N, K)
 * - Layout: Defines how elements are arranged in memory/register space
 * - Thread Mapping: How threads participate in the MMA operation
 *
 * For sm_89 (RTX 4060), we use 16x8x16 MMA operations which map to the tensor core capabilities:
 * - A operand: 16x16 elements
 * - B operand: 16x8 elements  
 * - C/D operand: 16x8 elements (result)
 */

template<class Element>
__global__ void mma_atom_spatial_kernel(
    Element const* __restrict__ ptr_A,
    Element const* __restrict__ ptr_B,
    Element*       __restrict__ ptr_D,
    int M, int N, int K) {
    
    // Define the MMA atom for 16x8x16 operations (for sm_89)
    // This represents a single tensor core operation
    auto mma_atom = Mma_Atom<SM80_16x8x16_F32F32F32F32_TN>{};
    
    // Calculate which tile this thread block is responsible for
    int block_start_m = blockIdx.x * 16;  // Each block handles 16 rows
    int block_start_n = blockIdx.y * 8;   // Each block handles 8 columns
    
    // Early exit if this block is out of bounds
    if (block_start_m >= M || block_start_n >= N) return;
    
    // Create tensors for A, B, and D operands using CuTe
    auto A_layout = make_layout(make_shape(M, K), make_stride(Int<K>{}, Int<1>{})); // Row-major: stride for rows is K, cols is 1
    auto B_layout = make_layout(make_shape(K, N), make_stride(Int<1>{}, Int<K>{})); // Column-major: stride for rows is 1, cols is K (TN: A is NT)
    auto D_layout = make_layout(make_shape(M, N), make_stride(Int<N>{}, Int<1>{})); // Row-major
    
    auto tensor_a = make_tensor(ptr_A, A_layout);
    auto tensor_b = make_tensor(ptr_B, B_layout);
    auto tensor_d = make_tensor(ptr_D, D_layout);
    
    // Define register fragments for A, B operands and accumulator
    auto tCs_A = make_tensor<Element>(repeat(shape(mma_atom), cute::A_tag{}));
    auto tCs_B = make_tensor<Element>(repeat(shape(mma_atom), cute::B_tag{}));
    auto tCr_D = make_tensor<Element>(repeat(shape(mma_atom), cute::C_tag{}));
    
    // Initialize the accumulator fragment
    fill(tCr_D, Element(0));
    
    // Iterate through the K dimension in chunks of 16 (the K dimension of the MMA operation)
    for (int k_block = 0; k_block < K; k_block += 16) {
        // Extract the A and B tiles for this iteration
        auto a_tile = tensor_a.slice(make_coord(range(block_start_m, min(M, block_start_m + 16)), 
                                                range(k_block, min(K, k_block + 16))));
        auto b_tile = tensor_b.slice(make_coord(range(k_block, min(K, k_block + 16)), 
                                                range(block_start_n, min(N, block_start_n + 8))));
        
        // Load A and B fragments from global memory to registers
        copy(a_tile, tCs_A);
        copy(b_tile, tCs_B);
        
        // Execute the MMA operation using the atom
        // This performs: tCr_D = tCs_A * tCs_B + tCr_D
        gemm(mma_atom, tCs_A, tCs_B, tCr_D, tCr_D);
    }
    
    // Store the result back to the output tensor
    auto d_tile = tensor_d.slice(make_coord(range(block_start_m, min(M, block_start_m + 16)), 
                                            range(block_start_n, min(N, block_start_n + 8))));
    copy(tCr_D, d_tile);
}

// Host function to launch the kernel
template<class Element>
void run_mma_atom_spatial(int M, int N, int K) {
    // Allocate host memory
    std::vector<Element> h_A(M * K);
    std::vector<Element> h_B(K * N);  // For TN operation: B is KxN in column-major layout
    std::vector<Element> h_D(M * N);
    
    // Initialize matrices with sample values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<Element>(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<Element>(2.0f);
    }
    
    // Allocate device memory
    Element *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, M * K * sizeof(Element));
    cudaMalloc(&d_B, K * N * sizeof(Element));
    cudaMalloc(&d_D, M * N * sizeof(Element));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(Element), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 grid((M + 15) / 16, (N + 7) / 8);  // Grid size based on 16x8 tile dimensions
    dim3 block(32);  // 32 threads per block (each block handles 16x8 tile)
    
    // Launch kernel
    mma_atom_spatial_kernel<<<grid, block>>>(d_A, d_B, d_D, M, N, K);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_D.data(), d_D, M * N * sizeof(Element), cudaMemcpyDeviceToHost);
    
    // Print a sample of the result
    std::cout << "Sample results (first 10 elements): ";
    for (int i = 0; i < std::min(10, static_cast<int>(h_D.size())); ++i) {
        std::cout << h_D[i] << " ";
    }
    std::cout << std::endl;
    
    // Calculate expected result for verification
    // For A filled with 1.0 and B filled with 2.0, each element of D should be K * 1.0 * 2.0 = 2*K
    std::cout << "Expected value for each element: " << 2 * K << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    
    std::cout << "MMA Atom Spatial kernel completed successfully!" << std::endl;
}

int main() {
    std::cout << "Module 4: MMA Atoms - Spatial Tensor Core Mastery" << std::endl;
    std::cout << "Running 16x8x16 Matrix Multiply using mma.sync instructions via cute::Mma_Atom" << std::endl;
    
    // Run with a small problem size for demonstration
    const int M = 32;
    const int N = 16;
    const int K = 32;
    
    run_mma_atom_spatial<float>(M, N, K);
    
    return 0;
}