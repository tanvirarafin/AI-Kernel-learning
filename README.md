# CUTLASS 3.x & CuTe Learning Repository

This repository serves as a structured learning path for mastering CUTLASS 3.x and CuTe, focusing on composable abstractions for high-performance GPU programming on NVIDIA hardware.

## Target Hardware
- NVIDIA RTX 4060 (Compute Capability 8.9 / Ada Lovelace)

## Repository Structure

### Module 1: Layouts and Tensors (CuTe basics, nested layouts)
- Introduction to `cute::Layout` and `cute::Tensor`
- Understanding Shape and Stride algebra
- Composable tensor partitioning for thread mapping

### Module 2: Tiled Copy (Vectorized global-to-shared memory movement)
- Efficient memory access patterns
- Vectorized loads and stores
- Shared memory tiling strategies

### Module 3: Tiled MMA (Using Tensor Cores via CuTe atoms)
- Tensor Core operations with CuTe
- MMA atom composition
- Performance optimization techniques

### Module 4: The Epilogue (Fused Bias-Add and ReLU implementations)
- Epilogue fusion techniques
- Memory-efficient activation functions
- Pipeline optimization

### Module 5: Mainloop Pipelining - Temporal Overlap & Throughput
- Double-buffered approach for hiding memory latency
- Temporal overlap of load and compute operations
- Throughput optimization techniques
- High-performance kernel design principles

### Module 6: Fused Epilogues - Functional Avoiding VRAM Roundtrips
- Fusing bias-add and activation functions within GEMM kernels
- Eliminating intermediate memory accesses
- Memory efficiency through in-register operations
- Performance optimization for neural network inference

## Prerequisites

- CUDA Toolkit compatible with Compute Capability 8.9
- CMake 3.18 or later
- Git for submodule management

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Initialize submodules:
```bash
git submodule update --init --recursive
```

3. Build the project:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Compilation Notes

Each module contains a standalone `main.cu` file that can be compiled individually using the provided NVCC command in the module's directory.

## Learning Approach

This repository emphasizes "composable abstractions" over manual indexing. Instead of traditional nested loops, we focus on:
- Partitioning tensors for threads using CuTe layouts
- Mathematical representation of memory mappings
- Functional composition of tensor operations