#!/usr/bin/env python3
# examples/triton_gemm.py
#
# Minimal Triton GEMM example:
# - Implements a blocked matrix multiply kernel in Triton (shared-memory style
#   at Python level via tiling in Triton).
# - Demonstrates masking for boundaries and a simple driver using PyTorch tensors.
# - Provides basic validation vs. PyTorch matmul and timing.
#
# Requirements:
#   pip install triton torch
#
# Run:
#   python3 examples/triton_gemm.py
#
# Notes:
# - This is a didactic example. For highly-optimized production kernels prefer
#   CUTLASS/cuBLAS or carefully autotuned Triton kernels.
# - The kernel below is compatible with Triton APIs that expose `triton.jit` and
#   `triton.language (tl)`. If your Triton version supports `triton.autotune`,
#   consider adding autotuning configs around the kernel as shown in the comment
#   below.

import math
import time

import torch
import triton
import triton.language as tl

# Block sizes (constexpr for Triton)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


# Example autotune hint (commented).
# If your Triton installation supports `triton.autotune`, you can wrap the
# kernel with autotune configs. Example (uncomment and adjust if supported):
#
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
#     ],
#     key=['M', 'N', 'K']
# )
#
# Note: Autotuning searches configurations at kernel launch time. Use with
# representative shapes and keep the search space small to avoid long tuning.


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,  # pointers to matrices (device addresses)
    M,
    N,
    K,  # matrix sizes
    stride_am,
    stride_ak,  # A row stride, A col stride (in elements)
    stride_bk,
    stride_bn,  # B row stride, B col stride (in elements)
    stride_cm,
    stride_cn,  # C row stride, C col stride (in elements)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Blocked matrix multiply: computes a BLOCK_M x BLOCK_N tile of C.

    A shape: (M, K)
    B shape: (K, N)
    C shape: (M, N)
    """

    pid_m = tl.program_id(0)  # which M tile
    pid_n = tl.program_id(1)  # which N tile

    # Row and column offsets for this tile
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks for rows/cols inside the matrix (for boundary handling)
    row_mask = row_offsets < M
    col_mask = col_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks of BLOCK_K
    k = 0
    while k < K:
        # Compute the K-range for this micro-tile
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Pointers for loading A and B:
        # A[row, k + kk]  -> A_ptr + row*stride_am + (k+kk)*stride_ak
        a_ptrs = (
            A_ptr
            + (row_offsets[:, None] * stride_am)
            + (k_offsets[None, :] * stride_ak)
        )
        # B[k + kk, col] -> B_ptr + (k+kk)*stride_bk + col*stride_bn
        b_ptrs = (
            B_ptr
            + (k_offsets[:, None] * stride_bk)
            + (col_offsets[None, :] * stride_bn)
        )

        # Load tiles with masking for bounds. Triton will return zeros for out-of-bounds
        a_tile = tl.load(a_ptrs, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & col_mask[None, :]), other=0.0)

        # a_tile: (BLOCK_M, BLOCK_K)
        # b_tile: (BLOCK_K, BLOCK_N)
        # Compute partial product. `tl.dot` reduces over the last axis of both operands.
        # For 2D x 2D, it performs a batched dot producing shape (BLOCK_M, BLOCK_N).
        acc += tl.dot(a_tile, b_tile)

        k += BLOCK_K

    # Store the result back to C with mask
    c_ptrs = (
        C_ptr + (row_offsets[:, None] * stride_cm) + (col_offsets[None, :] * stride_cn)
    )
    tl.store(c_ptrs, acc, mask=(row_mask[:, None] & col_mask[None, :]))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    High-level helper to call the Triton kernel.

    A: (M, K), float32, device='cuda'
    B: (K, N), float32, device='cuda'
    returns: C (M, N) on GPU
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Triton expects strides in number of elements (not bytes)
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    # Launch the kernel
    matmul_kernel[grid](
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def validate_and_benchmark(M=1024, N=1024, K=1024, repeat=10):
    device = "cuda"
    print(f"Running Triton GEMM: M={M}, N={N}, K={K}, repeat={repeat}")
    torch.manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)

    # Warm-up
    C_triton = triton_matmul(A, B)
    torch.cuda.synchronize()

    # Validate against PyTorch
    C_ref = torch.matmul(A, B)
    max_abs_err = (C_ref - C_triton).abs().max().item()
    max_rel_err = (C_ref - C_triton).abs().max().item() / (
        C_ref.abs().max().item() + 1e-12
    )
    print(f"Validation: max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}")

    # Benchmark Triton kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        C_triton = triton_matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_sec = (end - start) / repeat
    gflops = 2.0 * M * N * K / (avg_sec * 1e9)
    print(f"Triton kernel average time: {avg_sec:.6f} s, GFLOPS: {gflops:.2f}")

    # Benchmark PyTorch (cuBLAS) for reference
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_sec_ref = (end - start) / repeat
    gflops_ref = 2.0 * M * N * K / (avg_sec_ref * 1e9)
    print(f"PyTorch cuBLAS average time: {avg_sec_ref:.6f} s, GFLOPS: {gflops_ref:.2f}")


if __name__ == "__main__":
    # Small demo run. Adjust sizes if you want faster runs on CI or small GPUs.
    try:
        validate_and_benchmark(M=1024, N=1024, K=1024, repeat=10)
    except Exception as e:
        print("Error running Triton GEMM example:", e)
        print(
            "Make sure Triton and PyTorch are installed and you have a CUDA GPU available."
        )
