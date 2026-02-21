# CUDA Programming Cheat Sheet

Quick reference for CUDA kernel programming.

---

## 🧵 Thread Hierarchy

```cpp
// Thread indices
threadIdx.x      // Thread index within block (x dimension)
threadIdx.y      // Thread index within block (y dimension)
threadIdx.z      // Thread index within block (z dimension)

blockIdx.x       // Block index within grid (x dimension)
blockIdx.y       // Block index within grid (y dimension)
blockIdx.z       // Block index within grid (z dimension)

blockDim.x       // Number of threads per block (x dimension)
blockDim.y       // Number of threads per block (y dimension)
blockDim.z       // Number of threads per block (z dimension)

gridDim.x        // Number of blocks in grid (x dimension)
gridDim.y        // Number of blocks in grid (y dimension)
gridDim.z        // Number of blocks in grid (z dimension)

// Common indexing patterns
int tid = blockIdx.x * blockDim.x + threadIdx.x;                    // 1D global
int tid = threadIdx.x + threadIdx.y * blockDim.x;                   // 2D within block
int idx = (blockIdx.x * blockDim.x + threadIdx.x) + 
          (blockIdx.y * blockDim.y + threadIdx.y) * width;          // 2D global

// 2D grid indexing
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// 3D indexing (for volumes)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = x + y * width + z * width * height;
```

---

## 📦 Memory Types

| Type | Scope | Speed | Size | Lifetime |
|------|-------|-------|------|----------|
| Register | Thread | Fastest | 255/thread | Thread |
| Shared | Block | Fast | 48-96 KB/SM | Block |
| Local | Thread | Slow | GB | Thread |
| Global | Grid | Slowest | GB | Grid |
| Constant | Grid | Fast* | 64 KB | Grid |
| Texture | Grid | Fast* | GB | Grid |

*Fast when all threads read same address (broadcast)

```cpp
// Global memory (device memory)
float *d_data;
cudaMalloc(&d_data, size);
cudaFree(d_data);

// Shared memory (on-chip, per-block)
__shared__ float sharedData[256];
extern __shared__ float dynamicShared[];  // Dynamic sizing

// Constant memory (read-only, cached)
__constant__ float constants[100];
cudaMemcpyToSymbol(constants, h_data, size);

// Local memory (automatic, spills to global)
void kernel() {
    float localArray[1000];  // May spill to local memory
}
```

---

## 🔀 Memory Operations

```cpp
// Host to Device
cudaMemcpy(d_dest, h_src, size, cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_dest, d_src, size, cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_dest, d_src, size, cudaMemcpyDeviceToDevice);

// Async (requires pinned memory)
cudaMemcpyAsync(d_dest, h_src, size, cudaMemcpyHostToDevice, stream);

// Set memory
cudaMemset(d_data, 0, size);
cudaMemset2D(d_data, pitch, 0, width, height);

// Pinned (page-locked) host memory
float *h_pinned;
cudaMallocHost(&h_pinned, size);
cudaFreeHost(h_pinned);
```

---

## 🚀 Kernel Launch

```cpp
// Basic launch
kernel<<<gridDim, blockDim>>>(args);

// With shared memory size
kernel<<<gridDim, blockDim, sharedMemSize>>>(args);

// With stream
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);

// Error checking
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}

// Synchronization
cudaDeviceSynchronize();      // Wait for all kernels
cudaStreamSynchronize(stream); // Wait for stream
```

---

## 🔢 Common Patterns

### Grid-Stride Loop
```cpp
__global__ void gridStride(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        data[i] = process(data[i]);
    }
}
```

### Reduction (Sum)
```cpp
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load
    shared[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Write
    if (tid == 0) output[blockIdx.x] = shared[0];
}
```

### Tiled Matrix Multiply
```cpp
__global__ void matMul(float *A, float *B, float *C, int width) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];
    
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0;
    
    for (int t = 0; t < (width + 31) / 32; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * width + t * 32 + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * width + col];
        __syncthreads();
        
        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * width + col] = sum;
}
```

---

## ⚡ Warp Primitives

```cpp
// Shuffle operations (within warp)
float val = __shfl_sync(0xffffffff, src, srcLane);      // From specific lane
float val = __shfl_down_sync(0xffffffff, src, delta);   // From lane + delta
float val = __shfl_up_sync(0xffffffff, src, delta);     // From lane - delta
float val = __shfl_xor_sync(0xffffffff, src, mask);     // From lane ^ mask

// Vote operations
unsigned int mask = __ballot_sync(0xffffffff, pred);    // Bitmask of predicates
int any = __any_sync(0xffffffff, pred);                 // Any thread true?
int all = __all_sync(0xffffffff, pred);                 // All threads true?

// Lane ID
int laneId = threadIdx.x % 32;
int warpId = threadIdx.x / 32;
```

---

## 🔒 Atomic Operations

```cpp
// Basic atomics
atomicAdd(&addr, val);      // Atomic add
atomicSub(&addr, val);      // Atomic subtract
atomicExch(&addr, val);     // Atomic exchange
atomicMin(&addr, val);      // Atomic minimum
atomicMax(&addr, val);      // Atomic maximum
atomicInc(&addr, val);      // Atomic increment
atomicDec(&addr, val);      // Atomic decrement

// Compare and swap
int old = atomicCAS(&addr, compare, val);  // If *addr == compare, set to val

// For floats (compute capability >= 3.5)
float old = atomicAdd(&addr, val);  // Works directly on floats
```

---

## 🌊 Streams

```cpp
// Create stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// Launch kernel in stream
kernel<<<grid, block, 0, stream>>>(args);

// Async memory transfer
cudaMemcpyAsync(d_dest, h_src, size, cudaMemcpyHostToDevice, stream);

// Synchronize
cudaStreamSynchronize(stream);

// Destroy
cudaStreamDestroy(stream);

// Stream with priority
int least, greatest;
cudaDeviceGetStreamPriorityRange(&least, &greatest);
cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatest);
```

---

## 📊 Occupancy

```cpp
// Query max potential block size
int minGrid, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel, 0, 0);

// Query max active blocks per SM
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, 0);

// Calculate occupancy
int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
int activeWarps = numBlocks * blockSize / 32;
float occupancy = (float)activeWarps / maxWarpsPerSM * 100;
```

---

## 🐛 Error Handling

```cpp
// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Check kernel launch errors
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 🔧 Compilation

```bash
# Basic
nvcc file.cu -o output

# With architecture
nvcc file.cu -o output -arch=sm_70

# Debug build
nvcc file.cu -o output -g -G -O0

# Optimized build
nvcc file.cu -o output -O3 -use_fast_math

# Show register usage
nvcc file.cu -o output -Xptxas=-v

# Generate PTX
nvcc file.cu -ptx -o output.ptx

# Compile for multiple architectures
nvcc file.cu -o output -gencode arch=compute_70,code=sm_70 \
                                    -gencode arch=compute_80,code=sm_80
```

---

## 📏 Limits (Typical, Varies by GPU)

```
Max threads per block:     1024
Max block dimensions:      1024 x 1024 x 64
Max grid dimensions:       2^31 - 1 x 65535 x 65535
Registers per thread:      255
Shared memory per SM:      48-96 KB
Constant memory:           64 KB
Warp size:                 32
```

---

## 🎯 Optimization Checklist

- [ ] Use grid-stride loops for large data
- [ ] Coalesce global memory accesses
- [ ] Use shared memory for data reuse
- [ ] Avoid bank conflicts in shared memory
- [ ] Minimize thread divergence
- [ ] Use warp primitives when possible
- [ ] Overlap compute and transfer with streams
- [ ] Tune block size for occupancy
- [ ] Use constant memory for read-only uniform data
- [ ] Profile before optimizing!
