/*
 * Shared Memory Level 3: Bank Conflicts
 *
 * EXERCISE: Learn to identify and resolve bank conflicts in shared memory.
 *
 * CONCEPTS:
 * - Shared memory is divided into 32 banks (on most GPUs)
 * - Bank conflicts occur when multiple threads access same bank
 * - Padding eliminates bank conflicts
 * - Access patterns matter!
 *
 * SKILLS PRACTICED:
 * - Bank conflict detection
 * - Padding strategies
 * - Conflict-free access patterns
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define NUM_BANKS 32

// ============================================================================
// KERNEL 1: Column Access Pattern (Creates Bank Conflicts)
 * This kernel has severe bank conflicts - 32-way conflicts!
 * TODO: Fix by adding padding to eliminate conflicts
// ============================================================================
__global__ void columnAccessBad(float *input, float *output, int width) {
    // TODO: Add padding to shared memory declaration
    // Bad: float sharedData[BLOCK_SIZE];  // Causes 32-way bank conflicts
    // Good: float sharedData[BLOCK_SIZE + 1];  // Padding eliminates conflicts
    __shared__ float sharedData[BLOCK_SIZE];  // TODO: Fix this!

    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + row * blockDim.x + col;

    // Load data - each thread loads one element
    sharedData[row * blockDim.x + col] = input[idx];
    __syncthreads();

    // TODO: This access pattern causes bank conflicts!
    // Each thread reads from a different row but same column
    // All threads in a warp access the same bank
    output[idx] = sharedData[col * blockDim.x + row] * 2.0f;
}

// ============================================================================
// KERNEL 2: Fixed Column Access (With Padding)
 * TODO: Complete the padded version to eliminate bank conflicts
// ============================================================================
__global__ void columnAccessFixed(float *input, float *output, int width) {
    // TODO: Add padding to eliminate bank conflicts
    // Hint: Add 1 element padding: sharedData[BLOCK_SIZE + 1]
    // Or calculate: sharedData[BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS]
    __shared__ float sharedData[/* YOUR CODE HERE - Add padding */];

    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + row * blockDim.x + col;

    // Load with padded indexing
    int loadIdx = row * blockDim.x + col;
    sharedData[loadIdx] = input[idx];
    __syncthreads();

    // Access with padded indexing - now conflict-free!
    int storeIdx = col * (blockDim.x + 1) + row;  // TODO: Fix this line
    output[idx] = sharedData[storeIdx] * 2.0f;
}

// ============================================================================
// KERNEL 3: Strided Access Pattern Analysis
 * Identify which stride values cause bank conflicts
 * TODO: Complete the analysis and fix
// ============================================================================
__global__ void stridedAccess(float *input, float *output, int stride) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data
    sharedData[tid] = input[idx];
    __syncthreads();

    // Strided access - conflict depends on stride value!
    // TODO: Determine which strides cause conflicts:
    // - stride = 1: No conflict (consecutive)
    // - stride = 2: 2-way conflict
    // - stride = 32: 32-way conflict (worst!)
    // - stride = 33: No conflict (coprime with 32)
    int accessIdx = (tid * stride) % BLOCK_SIZE;
    output[idx] = sharedData[accessIdx];
}

// ============================================================================
// KERNEL 4: Matrix Transpose with Bank Conflict Avoidance
 * TODO: Implement conflict-free matrix transpose using padding
// ============================================================================
__global__ void matrixTransposeFixed(float *input, float *output, int width) {
    // TODO: Add padding to shared memory
    // Use: sharedData[TILE_WIDTH][TILE_WIDTH + 1]
    const int TILE_WIDTH = 32;
    __shared__ float sharedData[/* YOUR CODE HERE */];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int rowIn = by * TILE_WIDTH + ty;
    int colIn = bx * TILE_WIDTH + tx;
    int idxIn = rowIn * width + colIn;

    // TODO: Load with padded column index
    sharedData[ty][tx] = input[idxIn];  // TODO: Fix indexing with padding
    __syncthreads();

    // Transpose: read from [tx][ty], write to transposed position
    int rowOut = by * TILE_WIDTH + tx;
    int colOut = bx * TILE_WIDTH + ty;
    int idxOut = rowOut * width + colOut;

    // TODO: Store with padded row index
    output[idxOut] = sharedData[tx][ty];  // TODO: Fix indexing with padding
}

// ============================================================================
// KERNEL 5: Dynamic Shared Memory with Runtime Padding
 * Use dynamic shared memory with padding determined at runtime
// ============================================================================
__global__ void dynamicSharedPadding(float *input, float *output, int size, int padding) {
    // Dynamic shared memory - size determined at kernel launch
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load with padding consideration
    if (idx < size) {
        int paddedIdx = tid + (tid / (NUM_BANKS - 1)) * padding;
        sharedData[paddedIdx] = input[idx];
    }
    __syncthreads();

    // Process with padded access
    if (idx < size) {
        int accessIdx = tid + (tid / (NUM_BANKS - 1)) * padding;
        output[idx] = sharedData[accessIdx] * 2.0f;
    }
}

// Utility functions
void initMatrix(float *mat, int width, float seed) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)(i % 100) / 100.0f + seed;
    }
}

bool verifyTranspose(float *result, float *input, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = input[col * width + row];
            if (fabsf(result[row * width + col] - expected) > 1e-5f) return false;
        }
    }
    return true;
}

// Simple performance comparison helper (counts conflicts conceptually)
void analyzeBankConflicts(int stride) {
    int conflicts = 0;
    for (int warp = 0; warp < BLOCK_SIZE / 32; warp++) {
        int bankAccesses[NUM_BANKS] = {0};
        for (int t = 0; t < 32; t++) {
            int tid = warp * 32 + t;
            int accessIdx = (tid * stride) % BLOCK_SIZE;
            int bank = accessIdx % NUM_BANKS;
            bankAccesses[bank]++;
        }
        for (int b = 0; b < NUM_BANKS; b++) {
            if (bankAccesses[b] > 1) {
                conflicts += bankAccesses[b] - 1;
            }
        }
    }
    printf("  Stride %d: %d bank conflicts per block\n", stride, conflicts);
}

int main() {
    printf("=== Shared Memory Level 3: Bank Conflicts ===\n\n");

    // Part 1: Analyze bank conflicts for different strides
    printf("Part 1: Bank Conflict Analysis\n");
    printf("Analyzing different stride values:\n");
    int strides[] = {1, 2, 4, 8, 16, 32, 33, 64};
    for (int i = 0; i < 8; i++) {
        analyzeBankConflicts(strides[i]);
    }
    printf("\n");

    // Part 2: Test matrix transpose with and without padding
    const int WIDTH = 512;
    const int N = WIDTH * WIDTH;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_expected = (float*)malloc(size);

    initMatrix(h_A, WIDTH, 1.0f);

    // Compute expected transpose
    for (int row = 0; row < WIDTH; row++) {
        for (int col = 0; col < WIDTH; col++) {
            h_expected[row * WIDTH + col] = h_A[col * WIDTH + row];
        }
    }

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(WIDTH / 32, WIDTH / 32);

    // Test bad transpose (with conflicts)
    printf("Part 2: Matrix Transpose\n");
    printf("Testing transpose WITHOUT padding (has bank conflicts)...\n");
    columnAccessBad<<<grid, block>>>(d_A, d_B, WIDTH);
    cudaDeviceSynchronize();

    // Test fixed transpose
    printf("Testing transpose WITH padding (conflict-free)...\n");
    // TODO: Launch columnAccessFixed kernel
    // columnAccessFixed<<<grid, block>>>(d_A, d_B, WIDTH);
    /* YOUR CODE HERE */

    // Verify transpose
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if (verifyTranspose(h_B, h_A, WIDTH)) {
        printf("✓ Matrix transpose PASSED\n");
    } else {
        printf("✗ Matrix transpose FAILED - Fix the padding\n");
    }

    // Part 3: Dynamic shared memory test
    printf("\nPart 3: Dynamic Shared Memory\n");
    const int DYN_SIZE = 10000;
    float *h_dynIn = (float*)malloc(DYN_SIZE * sizeof(float));
    float *h_dynOut = (float*)malloc(DYN_SIZE * sizeof(float));
    for (int i = 0; i < DYN_SIZE; i++) h_dynIn[i] = i * 0.1f;

    float *d_dynIn, *d_dynOut;
    cudaMalloc(&d_dynIn, DYN_SIZE * sizeof(float));
    cudaMalloc(&d_dynOut, DYN_SIZE * sizeof(float));
    cudaMemcpy(d_dynIn, h_dynIn, DYN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int dynBlockSize = 256;
    int dynGridSize = (DYN_SIZE + dynBlockSize - 1) / dynBlockSize;
    int sharedMemSize = (DYN_SIZE + dynBlockSize / NUM_BANKS) * sizeof(float);

    printf("Launching dynamic shared memory kernel...\n");
    // TODO: Launch dynamicSharedPadding with dynamic shared memory
    // dynamicSharedPadding<<<dynGridSize, dynBlockSize, sharedMemSize>>>(
    //     d_dynIn, d_dynOut, DYN_SIZE, dynBlockSize / NUM_BANKS);
    /* YOUR CODE HERE */

    cudaMemcpy(h_dynOut, d_dynOut, DYN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    bool dynPass = true;
    for (int i = 0; i < DYN_SIZE && dynPass; i++) {
        if (fabsf(h_dynOut[i] - h_dynIn[i] * 2.0f) > 1e-5f) dynPass = false;
    }
    if (dynPass) {
        printf("✓ Dynamic shared memory PASSED\n");
    } else {
        printf("✗ Dynamic shared memory FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_expected);
    free(h_dynIn);
    free(h_dynOut);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dynIn);
    cudaFree(d_dynOut);

    printf("\n=== Key Takeaways ===\n");
    printf("- Shared memory has 32 banks (on most GPUs)\n");
    printf("- Bank conflicts serialize memory accesses\n");
    printf("- Add padding to eliminate conflicts\n");
    printf("- Stride values that are multiples of 32 cause worst conflicts\n");
    printf("- Stride values coprime to 32 (like 33) have no conflicts\n");
    printf("\nNext: Try level4_advanced_patterns.cu for complex algorithms\n");

    return 0;
}
