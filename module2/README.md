# Module 2: Tiled Copy (Vectorized global-to-shared memory movement)

## Overview
This module focuses on efficient memory access patterns using CuTe's tiled copy mechanisms. We'll explore how to move data between global and shared memory in a vectorized, coalesced manner.

## Key Concepts
- Tiled memory access patterns
- Vectorized load/store operations
- Memory coalescing with CuTe
- Shared memory tiling strategies
- Copy atom definitions

## Learning Objectives
By the end of this module, you will understand:
1. How to define tiled copy operations for optimal memory throughput
2. Techniques for vectorizing memory accesses
3. How to align data for efficient transfer between memory hierarchies
4. Practical examples of global-to-shared memory movement

## Files
- `main.cu` - Demonstrates tiled copy operations
- `README.md` - This file