# PTX Debugging and Profiling Exercises

This directory contains hands-on exercises to reinforce your understanding of PTX debugging and profiling techniques.

## Exercise 1: Register Spill Detection

### Objective
Identify and fix register spill issues in a PTX kernel.

### Files
- `spilling_kernel.ptx` - PTX kernel with register spilling issues
- `optimized_kernel.ptx` - Fixed version with reduced register usage
- `test_spill.cu` - CUDA test harness
- `profile_spill.sh` - Script to compare register usage

## Exercise 2: Memory Access Pattern Analysis

### Objective
Analyze and optimize memory access patterns in PTX code.

### Files
- `poor_pattern.ptx` - PTX with poor memory access patterns
- `optimized_pattern.ptx` - PTX with optimized access patterns
- `test_memory.cu` - CUDA test harness
- `profile_memory.sh` - Script to compare memory efficiency

## Exercise 3: Branch Divergence Debugging

### Objective
Identify and fix branch divergence issues in PTX code.

### Files
- `divergent_kernel.ptx` - PTX with branch divergence
- `convergent_kernel.ptx` - Fixed version with reduced divergence
- `test_divergence.cu` - CUDA test harness
- `profile_divergence.sh` - Script to compare warp execution efficiency

## Exercise 4: Complete Kernel Debugging Session

### Objective
Perform a complete debugging and optimization cycle on a complex kernel.

### Files
- `complex_kernel.ptx` - PTX kernel with multiple issues
- `debugged_kernel.ptx` - Fully optimized version
- `analysis.md` - Documentation of debugging process
- `test_complex.cu` - CUDA test harness
- `full_profile.sh` - Script for comprehensive profiling

## How to Use These Exercises

1. Use `cuobjdump` to examine the PTX and SASS code
2. Profile the kernels using `nvprof` or `Nsight Compute`
3. Identify performance bottlenecks and correctness issues
4. Modify the PTX code to fix the issues
5. Re-profile to validate improvements
6. Document your findings and the fixes applied