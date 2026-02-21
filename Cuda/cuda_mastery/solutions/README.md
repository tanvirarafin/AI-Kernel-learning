# CUDA Mastery Exercises - Solutions Guide

## 📖 How to Use This Directory

This directory contains **exercise templates** and **solutions** for mastering CUDA programming.

### Workflow

1. **Try the exercise first!** Go to the appropriate lesson directory (e.g., `01_basics/`)
2. Complete the TODO sections in the exercise files (e.g., `04_exercises_vector_ops.cu`)
3. Compile and test your solution
4. **Stuck?** Check the corresponding solution in `solutions/` directory

## 📁 Directory Structure

```
cuda_mastery/
├── 01_basics/
│   ├── 01_hello_cuda.cu          # Worked example
│   ├── 02_vector_add.cu          # Worked example
│   ├── 03_thread_indexing.cu     # Worked example
│   ├── 04_exercises_vector_ops.cu    # YOUR TURN! ✏️
│   └── 05_exercises_thread_indexing.cu  # YOUR TURN! ✏️
├── 02_memory_model/
│   ├── 01_memory_types.cu        # Worked example
│   ├── 02_memory_coalescing.cu   # Worked example
│   └── 01_exercises_memory_coalescing.cu  # YOUR TURN! ✏️
├── 03_shared_memory/
│   ├── 01_shared_memory_basics.cu    # Worked example
│   ├── 02_tiled_matrix_multiply.cu   # Worked example
│   └── 01_exercises_shared_memory_basics.cu  # YOUR TURN! ✏️
├── 04_synchronization/
│   └── 01_exercises_sync_atomics.cu  # YOUR TURN! ✏️
├── 05_optimization/
│   └── 01_exercises_occupancy_tuning.cu  # YOUR TURN! ✏️
├── 06_advanced/
│   ├── 01_cuda_streams.cu        # Worked example
│   ├── 02_unified_memory.cu      # Worked example
│   ├── 01_exercises_cuda_streams.cu    # YOUR TURN! ✏️
│   └── 02_exercises_unified_memory.cu  # YOUR TURN! ✏️
└── solutions/
    ├── 01_vector_ops_solution.cu
    ├── 02_thread_indexing_solution.cu
    └── ... (more solutions)
```

## ✏️ Exercise Files

Exercise files have:
- **TODO comments** indicating where to write code
- **FIXME hints** showing what needs to be fixed
- **Verification code** to check your answers
- **Hints section** at the bottom

### Example Exercise Structure

```cpp
// ============================================================================
// EXERCISE 1: Vector Subtraction
// Implement: C[i] = A[i] - B[i]
// ============================================================================
__global__ void vectorSubtract(const float *A, const float *B, float *C, int n) {
    // TODO: Calculate global thread index
    int i = 0;  // FIXME: Replace with correct formula
    
    // TODO: Add bounds check
    
    // TODO: Implement subtraction: C[i] = A[i] - B[i]
}
```

## 🔧 Compilation

### Exercises
```bash
cd cuda_mastery/01_basics
nvcc -o ex1.4 04_exercises_vector_ops.cu
./ex1.4
```

### Solutions
```bash
cd cuda_mastery
nvcc -o sol1.4 solutions/01_vector_ops_solution.cu
./sol1.4
```

## 📋 Exercise Checklist

### Level 1: Basics ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 1.4 | Vector Operations | `01_basics/04_exercises_vector_ops.cu` | Subtraction, Multiply, Scale, SAXPY |
| 1.5 | Thread Indexing | `01_basics/05_exercises_thread_indexing.cu` | 2D, 3D, Grid-stride, Diagonal |

### Level 2: Memory Model ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 2.1 | Memory Coalescing | `02_memory_model/01_exercises_memory_coalescing.cu` | Coalesced access, AoS vs SoA, Transpose |

### Level 3: Shared Memory ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 3.1 | Shared Memory Basics | `03_shared_memory/01_exercises_shared_memory_basics.cu` | Neighbor access, Reduction, Histogram |

### Level 4: Synchronization ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 4.1 | Sync and Atomics | `04_synchronization/01_exercises_sync_atomics.cu` | Barriers, Atomic ops, Spinlock, Warp shuffle |

### Level 5: Optimization ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 5.1 | Occupancy Tuning | `05_optimization/01_exercises_occupancy_tuning.cu` | Register usage, Block size, Occupancy API |

### Level 6: Advanced ✓

| # | Exercise | File | Topics |
|---|----------|------|--------|
| 6.1 | CUDA Streams | `06_advanced/01_exercises_cuda_streams.cu` | Concurrent execution, Async transfers |
| 6.2 | Unified Memory | `06_advanced/02_exercises_unified_memory.cu` | cudaMallocManaged, Prefetching, Advice |

## 🎯 Tips for Success

1. **Read the comments** - Each exercise explains what to implement
2. **Start simple** - Complete basic exercises before challenges
3. **Test incrementally** - Compile after each exercise
4. **Use the hints** - Check the HINTS section if stuck
5. **Compare with solutions** - Learn from the reference implementations
6. **Experiment** - Try different block sizes, configurations

## 🐛 Debugging Tips

### Common Errors

```
CUDA error: an illegal memory access was encountered
```
→ Check your bounds! Make sure `if (idx < n)` is present.

```
Wrong results / Verification failed
```
→ Check your indexing formula. Print intermediate values.

```
Kernel doesn't terminate
```
→ Make sure `__syncthreads()` is NOT inside conditionals.

### Debugging Pattern

```cpp
__global__ void debugKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Thread %d: idx=%d, n=%d\n", threadIdx.x, idx, n);
    
    if (idx < n) {
        printf("Processing: data[%d] = %f\n", idx, data[idx]);
        data[idx] = data[idx] * 2.0f;
    }
}
```

## 📚 Additional Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/145)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - Profiling tool
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - Kernel profiler

## 🚀 Next Steps

After completing all exercises:

1. **Try the challenge problems** in `exercises.md`
2. **Profile your code** with Nsight Systems/Compute
3. **Optimize further** - Can you beat the solution performance?
4. **Explore advanced topics** - Tensor cores, cooperative groups, etc.
5. **Build a project** - Apply your CUDA skills to real problems!

---

Good luck on your CUDA mastery journey! 🎉
