// ============================================================================
// Exercise 2.1: Memory Coalescing - Optimize Memory Access Patterns
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to understand memory coalescing.
//   Good coalescing can improve performance by 2x-16x!
//   Compile with: nvcc -o ex2.1 01_exercises_memory_coalescing.cu
//   Run with: ./ex2.1
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// EXERCISE 1: Coalesced Memory Access
// Implement a kernel where consecutive threads access consecutive addresses
// This is the OPTIMAL pattern!
// ============================================================================
__global__ void coalescedAccess(const float *input, float *output, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Read from input[idx] and write to output[idx]
    // This pattern is COALESCED - consecutive threads access consecutive addresses
}

// ============================================================================
// EXERCISE 2: Uncoalesced Memory Access (Strided)
// Implement a kernel where threads access with a stride
// This is a POOR pattern - DO NOT use in production!
// ============================================================================
__global__ void uncoalescedAccess(const float *input, float *output, int n, int stride) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check (remember: we're accessing idx * stride!)
    
    // TODO: Read from input[idx * stride] - this is UNCOALESCED!
    // Thread 0 reads index 0, Thread 1 reads index stride, etc.
}

// ============================================================================
// EXERCISE 3: Array of Structures (AoS) - Poor Coalescing
// Process only the X component of Point3D structures
// Memory layout: x0,y0,z0, x1,y1,z1, x2,y2,z2...
// ============================================================================
struct Point3D {
    float x, y, z;
};

__global__ void processAoS_XOnly(Point3D *points, float *output, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Access only the .x component
    // Memory layout means threads access: x0, x1, x2... (strided by 3!)
    // This is UNCOALESCED - poor performance!
}

// ============================================================================
// EXERCISE 4: Structure of Arrays (SoA) - Good Coalescing
// Same operation as above, but with SoA layout
// Memory layout: x0,x1,x2... y0,y1,y2... z0,z1,z2...
// ============================================================================
__global__ void processSoA_XOnly(float *xCoords, float *yCoords, float *zCoords,
                                  float *output, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Access xCoords[idx] - this is COALESCED!
    // All x values are contiguous in memory
}

// ============================================================================
// EXERCISE 5: Transpose - Classic Uncoalesced Pattern
// Read rows, write columns (or vice versa)
// One of the accesses will be uncoalesced!
// ============================================================================
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    // TODO: Calculate global x and y coordinates
    
    // TODO: Bounds check for both dimensions
    
    // TODO: Calculate input index (row-major: y * width + x)
    
    // TODO: Calculate output index (transposed: x * height + y)
    
    // TODO: Perform transpose: output[outputIdx] = input[inputIdx]
    // Note: Either the read OR the write will be uncoalesced!
}

// ============================================================================
// EXERCISE 6: Constant Memory Usage
// Store a multiplier in constant memory and apply it
// ============================================================================
// TODO: Declare a constant memory variable called d_multiplier
// Hint: __constant__ float d_multiplier;

__global__ void constantMemoryKernel(const float *input, float *output, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Multiply input by d_multiplier (from constant memory)
    // All threads read the SAME address - optimized for constant cache!
}

// ============================================================================
// TIMING UTILITY
// Measures kernel execution time over multiple iterations
// ============================================================================
float timeKernel(void (*kernel)(const float*, float*, int),
                 float *d_in, float *d_out, int n,
                 int blocks, int threads, int iterations) {
    // Warmup
    kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed;
}

// ============================================================================
// VERIFICATION FUNCTION
// ============================================================================
bool verifyResults(const float *gpu, const float *expected, int n, const char *test) {
    for (int i = 0; i < n; i++) {
        if (gpu[i] != expected[i]) {
            printf("  [FAIL] %s: Mismatch at %d: GPU=%.2f, Expected=%.2f\n",
                   test, i, gpu[i], expected[i]);
            return false;
        }
    }
    printf("  [PASS] %s\n", test);
    return true;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Memory Coalescing Exercises ===\n\n");
    
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Memory Clock Rate: %.0f MHz\n\n", prop.memoryClockRate * 1e-3f);
    
    int n = 1 << 20;  // 1M elements = 4MB
    size_t size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_expected = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i * 1.0f;
    }
    
    // Device arrays
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Array size: %d elements (%.2f MB)\n", n, size / (1024.0 * 1024.0));
    printf("Configuration: %d blocks, %d threads/block\n\n", blocksPerGrid, threadsPerBlock);
    
    // ========================================================================
    // TEST 1: Coalesced Access
    // ========================================================================
    printf("Exercise 1: Coalesced Memory Access\n");
    
    coalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute expected
    for (int i = 0; i < n; i++) {
        h_expected[i] = h_input[i] * 2.0f;
    }
    verifyResults(h_output, h_expected, n, "Coalesced Access");
    
    float coalescedTime = timeKernel(coalescedAccess, d_input, d_output, n,
                                      blocksPerGrid, threadsPerBlock, 100);
    printf("  Time (100 iterations): %.3f ms\n\n", coalescedTime);
    
    // ========================================================================
    // TEST 2: Uncoalesced Access (Stride = 32)
    // ========================================================================
    printf("Exercise 2: Uncoalesced Memory Access (stride=32)\n");
    
    int stride = 32;
    int reducedN = n / stride;  // Smaller array due to stride
    
    uncoalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, reducedN, stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute expected
    for (int i = 0; i < reducedN; i++) {
        h_expected[i] = h_input[i * stride] * 2.0f;
    }
    verifyResults(h_output, h_expected, reducedN, "Uncoalesced Access");
    
    float uncoalescedTime = timeKernel(uncoalescedAccess, d_input, d_output, reducedN,
                                        blocksPerGrid, threadsPerBlock, 100);
    printf("  Time (100 iterations): %.3f ms\n", uncoalescedTime);
    printf("  Slowdown vs coalesced: %.2fx\n\n", uncoalescedTime / coalescedTime);
    
    // ========================================================================
    // TEST 3: AoS vs SoA
    // ========================================================================
    printf("Exercise 3: Array of Structures (AoS)\n");
    
    // Allocate AoS array
    Point3D *d_points;
    size_t aosSize = n * sizeof(Point3D);
    CUDA_CHECK(cudaMalloc(&d_points, aosSize));
    
    // Initialize on host
    Point3D *h_points = (Point3D *)malloc(aosSize);
    for (int i = 0; i < n; i++) {
        h_points[i].x = i * 1.0f;
        h_points[i].y = i * 2.0f;
        h_points[i].z = i * 3.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_points, h_points, aosSize, cudaMemcpyHostToDevice));
    
    processAoS_XOnly<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute expected
    for (int i = 0; i < n; i++) {
        h_expected[i] = h_points[i].x * 2.0f;
    }
    verifyResults(h_output, h_expected, n, "AoS X-Only Processing");
    
    float aosTime = timeKernelAoS(processAoS_XOnly_kernel, d_points, d_output, n,
                                   blocksPerGrid, threadsPerBlock, 100);
    printf("  Time (100 iterations): %.3f ms\n\n", aosTime);
    
    // ========================================================================
    // TEST 4: SoA
    // ========================================================================
    printf("Exercise 4: Structure of Arrays (SoA)\n");
    
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, size));
    CUDA_CHECK(cudaMalloc(&d_y, size));
    CUDA_CHECK(cudaMalloc(&d_z, size));
    
    // Initialize
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);
    float *h_z = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_x[i] = i * 1.0f;
        h_y[i] = i * 2.0f;
        h_z[i] = i * 3.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    
    processSoA_XOnly<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute expected
    for (int i = 0; i < n; i++) {
        h_expected[i] = h_x[i] * 2.0f;
    }
    verifyResults(h_output, h_expected, n, "SoA X-Only Processing");
    
    float soaTime = timeKernel(processSoA_XOnly, d_x, d_output, n,
                                blocksPerGrid, threadsPerBlock, 100);
    printf("  Time (100 iterations): %.3f ms\n", soaTime);
    printf("  SoA vs AoS speedup: %.2fx\n\n", aosTime / soaTime);
    
    // ========================================================================
    // TEST 5: Matrix Transpose
    // ========================================================================
    printf("Exercise 5: Matrix Transpose (Naive)\n");
    
    int width = 1024, height = 1024;
    size_t matrixSize = width * height * sizeof(float);
    
    float *h_matrix = (float *)malloc(matrixSize);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_matrix[i * width + j] = i * 1000 + j;
        }
    }
    
    float *d_matrixIn, *d_matrixOut;
    CUDA_CHECK(cudaMalloc(&d_matrixIn, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_matrixOut, matrixSize));
    
    CUDA_CHECK(cudaMemcpy(d_matrixIn, h_matrix, matrixSize, cudaMemcpyHostToDevice));
    
    dim3 blockSize(32, 32);
    dim3 gridSize((width + 31) / 32, (height + 31) / 32);
    
    printf("  Matrix: %dx%d\n", width, height);
    printf("  Grid: (%d, %d) blocks, Block: (%d, %d) threads\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    transposeNaive<<<gridSize, blockSize>>>(d_matrixIn, d_matrixOut, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float *h_transposed = (float *)malloc(matrixSize);
    CUDA_CHECK(cudaMemcpy(h_transposed, d_matrixOut, matrixSize, cudaMemcpyDeviceToHost));
    
    // Verify a few elements
    printf("  Original[0,0]=%.0f, Transposed[0,0]=%.0f\n",
           h_matrix[0], h_transposed[0]);
    printf("  Original[1,2]=%.0f, Transposed[2,1]=%.0f\n",
           h_matrix[2 * width + 1], h_transposed[1 * width + 2]);
    printf("  (These should match if transpose is correct)\n\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_points);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_matrixIn);
    cudaFree(d_matrixOut);
    free(h_input);
    free(h_output);
    free(h_expected);
    free(h_points);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_matrix);
    free(h_transposed);
    
    printf("=== Memory Coalescing Principles ===\n");
    printf("1. Consecutive threads -> consecutive addresses (COALESCED)\n");
    printf("2. Avoid strided access when possible\n");
    printf("3. Use SoA instead of AoS for SIMD patterns\n");
    printf("4. Transpose requires shared memory for optimal performance\n");
    printf("5. Constant memory is fast when all threads read same address\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. Coalesced access pattern:
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    output[idx] = input[idx] * 2.0f;
//
// 2. Strided (uncoalesced) pattern:
//    output[idx] = input[idx * stride] * 2.0f;
//
// 3. AoS memory layout (bad for component access):
//    struct {x,y,z} arr[n] -> memory: x0,y0,z0,x1,y1,z1...
//
// 4. SoA memory layout (good for component access):
//    float x[n], y[n], z[n] -> memory: x0,x1,x2...,y0,y1,y2...,z0,z1,z2...
//
// 5. 2D indexing for transpose:
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    int rowMajorIdx = y * width + x;
// ============================================================================
