// ============================================================================
// Exercise 1.5: Thread Indexing Challenges - Master CUDA Coordinates
// ============================================================================
// INSTRUCTIONS:
//   Complete the TODO sections to implement various indexing patterns.
//   Understanding thread indexing is CRITICAL for CUDA programming!
//   Compile with: nvcc -o ex1.5 05_exercises_thread_indexing.cu
//   Run with: ./ex1.5
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// EXERCISE 1: 2D Thread Indexing
// Each thread sets its output to: globalX + globalY
// Use 2D blocks within a 1D grid
// ============================================================================
// TODO: Complete the kernel below
__global__ void index2D(float *output, int width, int height) {
    // TODO: Calculate thread's x position within block
    int tx = 0;  // FIXME
    
    // TODO: Calculate thread's y position within block  
    int ty = 0;  // FIXME
    
    // TODO: Calculate block's x position in grid
    int bx = 0;  // FIXME
    
    // TODO: Calculate global X coordinate
    int globalX = 0;  // FIXME
    
    // TODO: Calculate global Y coordinate
    int globalY = 0;  // FIXME
    
    // TODO: Check bounds (both x and y)
    
    // TODO: Calculate 1D index from 2D coordinates (row-major order)
    int idx = 0;  // FIXME
    
    // TODO: Set output to globalX + globalY
}

// ============================================================================
// EXERCISE 2: Full 2D Grid and 2D Blocks
// Process a matrix where each thread handles one element
// Use 2D grid AND 2D blocks
// ============================================================================
__global__ void matrixProcess(float *matrix, int width, int height, float value) {
    // TODO: Calculate global column (x) coordinate
    int col = 0;  // FIXME
    
    // TODO: Calculate global row (y) coordinate
    int row = 0;  // FIXME
    
    // TODO: Bounds check
    
    // TODO: Calculate 1D index (row-major: row * width + col)
    int idx = 0;  // FIXME
    
    // TODO: Set matrix element to: value * (row + col)
}

// ============================================================================
// EXERCISE 3: 3D Thread Indexing for Volume Data
// Process a 3D volume where each thread handles one voxel
// Set each voxel to: x + y + z
// ============================================================================
__global__ void volumeProcess(float *volume, int width, int height, int depth) {
    // TODO: Calculate thread's 3D coordinates within block
    int tx = threadIdx.x;
    int ty = 0;  // FIXME
    int tz = 0;  // FIXME
    
    // TODO: Calculate block's 3D coordinates in grid
    int bx = blockIdx.x;
    int by = 0;  // FIXME
    int bz = 0;  // FIXME
    
    // TODO: Calculate global 3D coordinates
    int x = 0;  // FIXME
    int y = 0;  // FIXME
    int z = 0;  // FIXME
    
    // TODO: Bounds check for all 3 dimensions
    
    // TODO: Calculate 1D index from 3D (z is slowest: z*height*width + y*width + x)
    int idx = 0;  // FIXME
    
    // TODO: Set voxel value to x + y + z
}

// ============================================================================
// EXERCISE 4: Grid-Stride Loop
// Handle arrays larger than total thread count
// Each thread processes multiple elements
// ============================================================================
__global__ void gridStrideAdd(const float *A, const float *B, float *C, int n) {
    // TODO: Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Calculate stride (total threads in grid)
    int stride = 0;  // FIXME: gridDim.x * blockDim.x
    
    // TODO: Implement grid-stride loop
    // for (int i = idx; i < n; i += stride) {
    //     C[i] = A[i] + B[i];
    // }
}

// ============================================================================
// EXERCISE 5: Diagonal Matrix Extraction (Challenge!)
// Extract the diagonal of a matrix (where row == col)
// Only threads on diagonal should write
// ============================================================================
__global__ void extractDiagonal(float *matrix, float *diagonal, int width, int height) {
    // TODO: Calculate global row and column
    
    // TODO: Check if this thread is on the diagonal (row == col)
    
    // TODO: Also check bounds
    
    // TODO: If on diagonal, write to diagonal array
    // Hint: diagonal index = row (or col, they're equal on diagonal)
}

// ============================================================================
// VERIFICATION FUNCTION
// ============================================================================
bool verify2DPattern(const float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float expected = x + y;
            if (output[idx] != expected) {
                printf("  [FAIL] Mismatch at (%d,%d): got %.1f, expected %.1f\n",
                       x, y, output[idx], expected);
                return false;
            }
        }
    }
    printf("  [PASS] 2D pattern verified (%dx%d matrix)\n", width, height);
    return true;
}

bool verify3DPattern(const float *volume, int width, int height, int depth) {
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = z * height * width + y * width + x;
                float expected = x + y + z;
                if (volume[idx] != expected) {
                    printf("  [FAIL] Mismatch at (%d,%d,%d): got %.1f, expected %.1f\n",
                           x, y, z, volume[idx], expected);
                    return false;
                }
            }
        }
    }
    printf("  [PASS] 3D pattern verified (%dx%dx%d volume)\n", width, height, depth);
    return true;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printf("=== Thread Indexing Exercises ===\n\n");
    
    // ========================================================================
    // TEST 1: 2D Thread Indexing
    // ========================================================================
    printf("Exercise 1: 2D Thread Indexing\n");
    printf("  Config: 1D grid, 2D blocks (16x16)\n");
    
    int width = 64, height = 64;
    size_t size2D = width * height * sizeof(float);
    
    float *h_output2D = (float *)malloc(size2D);
    float *d_output2D;
    CUDA_CHECK(cudaMalloc(&d_output2D, size2D));
    
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x);
    
    printf("  Grid: (%d, 1) blocks, Block: (%d, %d) threads\n\n", 
           gridSize2D.x, blockSize2D.x, blockSize2D.y);
    
    index2D<<<gridSize2D, blockSize2D>>>(d_output2D, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output2D, d_output2D, size2D, cudaMemcpyDeviceToHost));
    verify2DPattern(h_output2D, width, height);
    printf("  Sample: output[0,0]=%.0f, output[1,2]=%.0f, output[3,3]=%.0f\n\n",
           h_output2D[0], h_output2D[2 * width + 1], h_output2D[3 * width + 3]);
    
    // ========================================================================
    // TEST 2: Full 2D Grid and 2D Blocks
    // ========================================================================
    printf("Exercise 2: Full 2D Grid and 2D Blocks\n");
    printf("  Config: 2D grid (4x4), 2D blocks (16x16)\n");
    
    float *h_matrix = (float *)malloc(size2D);
    float *d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, size2D));
    
    dim3 gridSize2DFull((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize2DFull(16, 16);
    
    printf("  Grid: (%d, %d) blocks, Block: (%d, %d) threads\n\n",
           gridSize2DFull.x, gridSize2DFull.y, blockSize2DFull.x, blockSize2DFull.y);
    
    matrixProcess<<<gridSize2DFull, blockSize2DFull>>>(d_matrix, width, height, 2.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, size2D, cudaMemcpyDeviceToHost));
    printf("  Sample: matrix[0,0]=%.0f, matrix[1,2]=%.0f, matrix[5,5]=%.0f\n\n",
           h_matrix[0], h_matrix[2 * width + 1], h_matrix[5 * width + 5]);
    
    // ========================================================================
    // TEST 3: 3D Thread Indexing
    // ========================================================================
    printf("Exercise 3: 3D Thread Indexing for Volume Data\n");
    
    int vWidth = 32, vHeight = 32, vDepth = 16;
    size_t size3D = vWidth * vHeight * vDepth * sizeof(float);
    
    float *h_volume = (float *)malloc(size3D);
    float *d_volume;
    CUDA_CHECK(cudaMalloc(&d_volume, size3D));
    
    dim3 blockSize3D(8, 8, 4);
    dim3 gridSize3D((vWidth + 7) / 8, (vHeight + 7) / 8, (vDepth + 3) / 4);
    
    printf("  Volume: %dx%dx%d\n", vWidth, vHeight, vDepth);
    printf("  Grid: (%d, %d, %d) blocks, Block: (%d, %d, %d) threads\n\n",
           gridSize3D.x, gridSize3D.y, gridSize3D.z,
           blockSize3D.x, blockSize3D.y, blockSize3D.z);
    
    volumeProcess<<<gridSize3D, blockSize3D>>>(d_volume, vWidth, vHeight, vDepth);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_volume, d_volume, size3D, cudaMemcpyDeviceToHost));
    verify3DPattern(h_volume, vWidth, vHeight, vDepth);
    printf("  Sample: volume[0,0,0]=%.0f, volume[1,2,3]=%.0f\n\n",
           h_volume[0], h_volume[3 * vHeight * vWidth + 2 * vWidth + 1]);
    
    // ========================================================================
    // TEST 4: Grid-Stride Loop
    // ========================================================================
    printf("Exercise 4: Grid-Stride Loop\n");
    
    int n = 100000;  // Large array
    size_t size = n * sizeof(float);
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Use SMALL configuration to force grid-stride
    int threadsPerBlock = 32;
    int blocksPerGrid = 10;  // Only 10 blocks = 320 threads for 100000 elements!
    
    printf("  Array size: %d elements\n", n);
    printf("  Threads: %d (each processes ~%d elements)\n\n",
           threadsPerBlock * blocksPerGrid, n / (threadsPerBlock * blocksPerGrid));
    
    gridStrideAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("  [%s] Grid-stride loop: All elements = 3.0\n\n",
           correct ? "PASS" : "FAIL");
    
    // ========================================================================
    // TEST 5: Diagonal Extraction (Challenge!)
    // ========================================================================
    printf("Exercise 5: Diagonal Matrix Extraction (Challenge!)\n");
    
    int diagSize = (width < height) ? width : height;
    float *h_diagonal = (float *)malloc(diagSize * sizeof(float));
    float *d_diagonal;
    CUDA_CHECK(cudaMalloc(&d_diagonal, diagSize * sizeof(float)));
    
    // Initialize matrix with known pattern: matrix[i,j] = i*10 + j
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_matrix[i * width + j] = i * 100 + j;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, size2D, cudaMemcpyHostToDevice));
    
    dim3 gridSizeDiag((width + 15) / 16, (height + 15) / 16);
    dim3 blockSizeDiag(16, 16);
    
    extractDiagonal<<<gridSizeDiag, blockSizeDiag>>>(d_matrix, d_diagonal, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_diagonal, d_diagonal, diagSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("  Diagonal values (first 10): ");
    for (int i = 0; i < 10 && i < diagSize; i++) {
        printf("%.0f ", h_diagonal[i]);
    }
    printf("\n  (Expected: 0, 101, 202, 303, 404, ...)\n\n");
    
    // Cleanup
    cudaFree(d_output2D);
    cudaFree(d_matrix);
    cudaFree(d_volume);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_diagonal);
    free(h_output2D);
    free(h_matrix);
    free(h_volume);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_diagonal);
    
    printf("=== All Indexing Exercises Complete! ===\n");
    printf("Next: Move to 02_memory_model/ exercises\n");
    
    return 0;
}

// ============================================================================
// HINTS:
// ============================================================================
// 1. 2D indexing formulas:
//    globalX = blockIdx.x * blockDim.x + threadIdx.x
//    globalY = blockIdx.y * blockDim.y + threadIdx.y
//
// 2. 2D to 1D conversion (row-major):
//    idx = row * width + col
//
// 3. 3D to 1D conversion:
//    idx = z * (height * width) + y * width + x
//
// 4. Grid-stride loop pattern:
//    int stride = gridDim.x * blockDim.x;
//    for (int i = globalIdx; i < n; i += stride) { ... }
//
// 5. Always check bounds for EACH dimension!
// ============================================================================
