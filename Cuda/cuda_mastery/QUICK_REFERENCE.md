# CUDA Mastery - Quick Reference Card

## 🚀 Quick Start

```bash
cd cuda_mastery
make all          # Build all exercises
make run-vector-ops  # Run first exercise
```

## 📁 Exercise Files

| Level | Topic | Exercise File | Solution |
|-------|-------|---------------|----------|
| 1 | Vector Ops | `01_basics/04_exercises_vector_ops.cu` | `solutions/01_vector_ops_solution.cu` |
| 1 | Indexing | `01_basics/05_exercises_thread_indexing.cu` | `solutions/02_thread_indexing_solution.cu` |
| 2 | Coalescing | `02_memory_model/01_exercises_memory_coalescing.cu` | - |
| 3 | Shared Mem | `03_shared_memory/01_exercises_shared_memory_basics.cu` | - |
| 4 | Sync/Atomics | `04_synchronization/01_exercises_sync_atomics.cu` | - |
| 5 | Occupancy | `05_optimization/01_exercises_occupancy_tuning.cu` | - |
| 6 | Streams | `06_advanced/01_exercises_cuda_streams.cu` | - |
| 6 | Unified Mem | `06_advanced/02_exercises_unified_memory.cu` | - |

## 🔑 Key Formulas

### Thread Indexing
```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * height * width + y * width + x;

// Grid-stride loop
int stride = gridDim.x * blockDim.x;
for (int i = idx; i < n; i += stride) { ... }
```

### Memory Operations
```cpp
// Traditional CUDA
cudaMalloc(&d_ptr, size);
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_ptr, n);
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
cudaFree(d_ptr);

// Unified Memory
cudaMallocManaged(&ptr, size);
// Access from CPU and GPU directly
cudaFree(ptr);
```

### Synchronization
```cpp
__syncthreads();              // Block barrier
atomicAdd(&addr, val);        // Atomic add
__shfl_sync(mask, var, lane); // Warp shuffle
```

## 📊 Common Patterns

### Reduction (Sum)
```cpp
__shared__ float sdata[256];
int tid = threadIdx.x;
sdata[tid] = input[idx];
__syncthreads();

for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
if (tid == 0) output[blockIdx.x] = sdata[0];
```

### Tiled Matrix Multiply
```cpp
__shared__ float As[TILE][TILE], Bs[TILE][TILE];
int col = blockIdx.x * TILE + threadIdx.x;
int row = blockIdx.y * TILE + threadIdx.y;

for (int t = 0; t < numTiles; t++) {
    As[threadIdx.y][threadIdx.x] = A[row][t*TILE + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[t*TILE + threadIdx.y][col];
    __syncthreads();
    
    for (int k = 0; k < TILE; k++)
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
}
```

### Warp Reduction
```cpp
if (tid < 32) {
    float val = sdata[tid];
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    if (tid == 0) output = val;
}
```

## 🛠️ Debugging

### Error Checking Macro
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaGetLastError());  // After kernel launch
```

### Timing
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## 📈 Profiling

```bash
# Check register usage
nvcc -ptxas-options=-v file.cu

# Limit registers
nvcc -maxrregcount=32 file.cu

# Profile with nvprof
nvprof ./program
nvprof --metrics all ./program

# Nsight Systems
nsys profile ./program

# Nsight Compute
ncu --set full ./program
```

## ⚡ Optimization Tips

1. **Memory Coalescing**: Consecutive threads → consecutive addresses
2. **Shared Memory**: 100x faster than global, use for data reuse
3. **Occupancy**: Use `cudaOccupancyMaxPotentialBlockSize()`
4. **Streams**: Overlap transfers and compute
5. **Warp Shuffles**: 50x faster than shared memory (intra-warp)
6. **Avoid Bank Conflicts**: Pad shared memory arrays

## 🎯 Exercise Workflow

1. Read worked example (e.g., `01_hello_cuda.cu`)
2. Try exercise file (e.g., `04_exercises_vector_ops.cu`)
3. Fill in TODO sections
4. Compile: `nvcc -o ex file.cu`
5. Run: `./ex`
6. Stuck? Check `solutions/` directory

---

**Happy CUDA Coding!** 🚀
