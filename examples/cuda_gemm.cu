/*
examples/cuda_gemm.cu

Simple CUDA GEMM example file with:
- CPU reference implementation (naive)
- CUDA naive kernel (global memory reads only)
- CUDA tiled shared-memory kernel (optimized)
- Validation and simple timing using cudaEvents
- GFLOPS calculation and basic correctness checks

Build:
  nvcc -O3 -arch=sm_70 examples/cuda_gemm.cu -o examples/cuda_gemm

Run:
  ./examples/cuda_gemm [M] [N] [K] [repeat]
  default: M=N=K=1024, repeat=10

Notes:
- This is a didactic example. For production use prefer libraries like cuBLAS or
CUTLASS.
- TILE is chosen as 16 for portability; for modern GPUs consider 32 or using
WMMA/CUTLASS for Tensor Cores.
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int TILE = 16; // tile dimension for shared memory kernel

// CPU reference (naive) : C = A * B
void cpu_gemm(int M, int N, int K, const float *A, const float *B, float *C) {
  // Zero C
  for (int i = 0; i < M * N; ++i)
    C[i] = 0.0f;

  // Naive i-k-j loop (cache-friendly for B if B is in column-major; here we
  // keep row-major)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a = A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

// Naive CUDA kernel: each thread computes one element C[row, col]
// All loads from global memory (no shared memory)
__global__ void gemm_naive_kernel(int M, int N, int K,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

// Tiled shared-memory kernel: each block loads a TILExTILE tile of A and B into
// shared memory
__global__ void gemm_tiled_kernel(int M, int N, int K,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C) {
  // Shared memory for a tile of A and B
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  // Loop over tiles
  int numTiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t) {
    // Load A tile: row, (t * TILE + tx)
    int aCol = t * TILE + threadIdx.x;
    if (row < M && aCol < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    // Load B tile: (t * TILE + ty), col
    int bRow = t * TILE + threadIdx.y;
    if (bRow < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

// Compute partial product for this tile
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// Utility: initialize matrix with random floats
void init_matrix(int rows, int cols, float *mat, float min = -1.0f,
                 float max = 1.0f) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(min, max);
  for (int i = 0; i < rows * cols; ++i)
    mat[i] = dist(rng);
}

// Compute maximum absolute and relative error vs reference
void compute_error(int M, int N, const float *ref, const float *out,
                   float &max_abs, float &max_rel) {
  max_abs = 0.0f;
  max_rel = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    float a = ref[i];
    float b = out[i];
    float abs_err = std::fabs(a - b);
    if (abs_err > max_abs)
      max_abs = abs_err;
    float denom = std::fabs(a);
    float rel = denom > 1e-6f ? abs_err / denom : abs_err;
    if (rel > max_rel)
      max_rel = rel;
  }
}

int main(int argc, char **argv) {
  int M = 1024, N = 1024, K = 1024;
  int repeat = 10;

  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }
  if (argc >= 5)
    repeat = std::atoi(argv[4]);
  if (M <= 0 || N <= 0 || K <= 0) {
    std::cerr << "Invalid matrix size\n";
    return 1;
  }

  std::cout << "GEMM sizes M=" << M << " N=" << N << " K=" << K
            << " repeat=" << repeat << "\n";

  size_t sizeA = size_t(M) * size_t(K);
  size_t sizeB = size_t(K) * size_t(N);
  size_t sizeC = size_t(M) * size_t(N);

  // Host allocations
  std::vector<float> h_A(sizeA);
  std::vector<float> h_B(sizeB);
  std::vector<float> h_C_ref(sizeC);
  std::vector<float> h_C_gpu(sizeC);

  init_matrix(M, K, h_A.data());
  init_matrix(K, N, h_B.data());

  // Compute CPU reference (single run)
  {
    std::cout << "Computing CPU reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_gemm(M, N, K, h_A.data(), h_B.data(), h_C_ref.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double gflops = 2.0 * double(M) * double(N) * double(K) / (1e9 * sec);
    std::cout << "CPU time: " << sec << " s, GFLOPS: " << gflops << "\n";
  }

  // Device allocations
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

  // Copy inputs to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Kernel launch parameters
  dim3 block(TILE, TILE);
  dim3 grid_naive((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  dim3 grid_tiled((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  // Warm up kernels
  CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(float)));
  gemm_naive_kernel<<<grid_naive, block>>>(M, N, K, d_A, d_B, d_C);
  CUDA_CHECK(cudaPeekAtLastError());

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(float)));
  gemm_tiled_kernel<<<grid_tiled, block>>>(M, N, K, d_A, d_B, d_C);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing structures
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 1) Time naive kernel
  CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    gemm_naive_kernel<<<grid_naive, block>>>(M, N, K, d_A, d_B, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_naive = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
  double sec_naive = ms_naive / 1000.0;
  double avg_sec_naive = sec_naive / repeat;
  double gflops_naive =
      2.0 * double(M) * double(N) * double(K) / (1e9 * avg_sec_naive);
  std::cout << "\nNaive kernel average time: " << avg_sec_naive
            << " s, GFLOPS: " << gflops_naive << "\n";

  // Copy back and validate (naive)
  CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, sizeC * sizeof(float),
                        cudaMemcpyDeviceToHost));
  {
    float max_abs, max_rel;
    compute_error(M, N, h_C_ref.data(), h_C_gpu.data(), max_abs, max_rel);
    std::cout << "Naive kernel: max abs error = " << max_abs
              << ", max rel error = " << max_rel << "\n";
  }

  // 2) Time tiled kernel
  CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    gemm_tiled_kernel<<<grid_tiled, block>>>(M, N, K, d_A, d_B, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_tiled = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));
  double sec_tiled = ms_tiled / 1000.0;
  double avg_sec_tiled = sec_tiled / repeat;
  double gflops_tiled =
      2.0 * double(M) * double(N) * double(K) / (1e9 * avg_sec_tiled);
  std::cout << "\nTiled kernel average time: " << avg_sec_tiled
            << " s, GFLOPS: " << gflops_tiled << "\n";

  // Copy back and validate (tiled)
  CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, sizeC * sizeof(float),
                        cudaMemcpyDeviceToHost));
  {
    float max_abs, max_rel;
    compute_error(M, N, h_C_ref.data(), h_C_gpu.data(), max_abs, max_rel);
    std::cout << "Tiled kernel: max abs error = " << max_abs
              << ", max rel error = " << max_rel << "\n";
  }

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  std::cout << "\nDone.\n";
  return 0;
}
