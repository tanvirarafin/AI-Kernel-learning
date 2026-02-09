# PTX Basics Exercises

This directory contains hands-on exercises to reinforce your understanding of PTX basics.

## Exercise 1: Simple Addition Kernel

### Objective
Write a PTX kernel that adds two arrays element-wise.

### Files
- `add_arrays.ptx` - PTX implementation
- `test_add_arrays.cu` - CUDA test harness

## Exercise 2: Vector Operations

### Objective
Implement basic vector operations (dot product, vector addition) in PTX.

### Files
- `vector_ops.ptx` - PTX implementation
- `test_vectors.cu` - CUDA test harness

## Exercise 3: Control Flow

### Objective
Write a PTX kernel that implements conditional operations.

### Files
- `conditional.ptx` - PTX implementation with branches
- `test_conditional.cu` - CUDA test harness

## Exercise 4: Memory Access Patterns

### Objective
Explore different memory access patterns in PTX (coalesced vs non-coalesced).

### Files
- `memory_patterns.ptx` - PTX implementations
- `test_memory.cu` - CUDA test harness

## How to Use These Exercises

1. Study the provided PTX examples
2. Modify them to complete the exercises
3. Compile with `nvcc -ptx filename.ptx -o filename.ptx` (to validate syntax)
4. Test with the provided CUDA harness
5. Examine the generated assembly with `cuobjdump`