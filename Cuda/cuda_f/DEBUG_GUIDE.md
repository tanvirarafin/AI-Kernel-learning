# CUDA Debugging Guide

Comprehensive guide for debugging CUDA kernels.

---

## 🔍 Common Errors and Solutions

### 1. CUDA Error: invalid configuration argument

**Cause**: Invalid grid/block dimensions

```cpp
// WRONG - block size > 1024
kernel<<<1024, 2048>>>(args);  // Error!

// FIX - Use valid dimensions
kernel<<<1024, 256>>>(args);   // OK
```

**Solution**: Check your launch configuration
```cpp
printf("Grid: %d, Block: %d\n", gridDim, blockDim);
if (blockDim > 1024) {
    printf("Error: Block size exceeds 1024!\n");
}
```

---

### 2. CUDA Error: an illegal memory access was encountered

**Cause**: Out-of-bounds memory access

```cpp
// WRONG - Missing bounds check
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = value;  // May access beyond n!
}

// FIX - Add bounds check
__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // Bounds check!
        data[idx] = value;
    }
}
```

**Debug with**:
```cpp
// Add printf in kernel (debug build only)
if (idx >= n) {
    printf("Thread %d accessing out of bounds! n=%d\n", idx, n);
}
```

---

### 3. Wrong Results (No Error)

**Cause**: Missing synchronization

```cpp
// WRONG - Reading before all writes complete
__shared__ float shared[256];
shared[tid] = input[tid];
float val = shared[255 - tid];  // May read stale data!

// FIX - Add synchronization
__shared__ float shared[256];
shared[tid] = input[tid];
__syncthreads();  // Wait for all threads
float val = shared[255 - tid];
```

---

### 4. Race Conditions

**Cause**: Multiple threads writing same location

```cpp
// WRONG - Race condition!
__global__ void reduce(float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[0] += input[idx];  // Multiple threads write output[0]!
}

// FIX - Use atomics or proper reduction
__global__ void reduce(float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&output[0], input[idx]);  // Safe!
}
```

---

### 5. Bank Conflicts

**Cause**: Multiple threads accessing same memory bank

```cpp
// WRONG - Strided access causes bank conflicts
__shared__ float shared[256];
float val = shared[tid * 32];  // 32-way bank conflict!

// FIX - Add padding
__shared__ float shared[256 + 8];  // Padding
float val = shared[tid * 32];  // Now conflict-free
```

---

## 🛠️ Debugging Tools

### cuda-gdb

```bash
# Compile with debug symbols
nvcc -G -g kernel.cu -o kernel

# Run with cuda-gdb
cuda-gdb ./kernel

# Common commands
(cuda-gdb) cuda kernels          # List kernels
(cuda-gdb) cuda launch           # Launch kernel
(cuda-gdb) cuda step             # Step through kernel
(cuda-gdb) cuda print locals     # Print local variables
(cuda-gdb) cuda warp             # Show warp info
```

### Compute Sanitizer

```bash
# Run with compute sanitizer
compute-sanitizer ./program

# With specific checks
compute-sanitizer --tool memcheck ./program
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool initcheck ./program
```

### Nsight Compute

```bash
# Profile kernel
ncu ./program

# Specific metrics
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program

# Export to file
ncu --output profile.ncu-rep ./program
```

### Nsight Systems

```bash
# System-wide profiling
nsys profile ./program

# With CUDA API tracing
nsys profile --trace cuda ./program

# Generate report
nsys stats report.qdrep
```

---

## 📝 Debugging Patterns

### 1. Add Error Checking Macro

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel execution error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
kernel<<<grid, block>>>(args);
CUDA_KERNEL_CHECK();
```

---

### 2. Kernel Output Validation

```cpp
__global__ void kernel(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float result = compute(idx);
        
        // Debug output for first few elements
        if (idx < 10) {
            printf("Thread %d: result = %f\n", idx, result);
        }
        
        // Check for NaN/Inf
        if (isnan(result) || isinf(result)) {
            printf("Thread %d: INVALID result!\n", idx);
        }
        
        output[idx] = result;
    }
}
```

---

### 3. Memory Pattern Check

```cpp
// Host-side verification
void verifyMemory(float *d_data, int n, const char *label) {
    float *h_data = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("=== %s ===\n", label);
    printf("First 10 elements: ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%.4f ", h_data[i]);
    }
    printf("\n");
    
    // Check for NaN/Inf
    int nanCount = 0, infCount = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(h_data[i])) nanCount++;
        if (isinf(h_data[i])) infCount++;
    }
    printf("NaN count: %d, Inf count: %d\n", nanCount, infCount);
    
    free(h_data);
}
```

---

### 4. Timing and Performance Debug

```cpp
// Simple timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

printf("Kernel execution time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 🐛 Debugging Checklist

When your kernel produces wrong results:

1. **Check launch configuration**
   - [ ] Grid and block dimensions are valid
   - [ ] Total threads cover all data

2. **Check memory access**
   - [ ] All accesses have bounds checks
   - [ ] No out-of-bounds reads/writes
   - [ ] Memory is properly allocated

3. **Check synchronization**
   - [ ] `__syncthreads()` after shared memory writes
   - [ ] No conditional synchronization

4. **Check for race conditions**
   - [ ] No unsynchronized shared writes
   - [ ] Use atomics for concurrent updates

5. **Check data types**
   - [ ] No integer division when float expected
   - [ ] Proper type casting

6. **Check initialization**
   - [ ] All memory initialized before use
   - [ ] Shared memory initialized

---

## 🔬 Common Debug Scenarios

### Scenario 1: Kernel Returns All Zeros

**Possible causes**:
- Bounds check too restrictive
- Wrong index calculation
- Memory not copied to device

**Debug steps**:
```cpp
// Add debug printf
__global__ void kernel(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: n=%d, idx=%d\n", threadIdx.x, n, idx);
    if (idx < n) {
        output[idx] = 1.0f;
    }
}
```

---

### Scenario 2: Results Change Between Runs

**Possible causes**:
- Race condition
- Uninitialized memory
- Atomic operation order

**Debug steps**:
```cpp
// Initialize all memory
cudaMemset(d_data, 0, size);

// Use cuda-memcheck
cuda-memcheck --tool racecheck ./program
```

---

### Scenario 3: Kernel Works for Small Data, Fails for Large

**Possible causes**:
- Missing grid-stride loop
- Resource exhaustion
- Timeout (TDR on Windows)

**Debug steps**:
```cpp
// Use grid-stride loop
for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    // Process element i
}
```

---

## 📚 Additional Resources

- [CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/)
- [Compute Sanitizer](https://docs.nvidia.com/cuda/compute-sanitizer/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
