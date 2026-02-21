// ============================================================================
// SOLUTION: Exercise 1.5 - Thread Indexing Challenges
// ============================================================================
// Complete working solutions for all thread indexing exercises.
// Compile with: nvcc -o sol1.5 solutions/02_thread_indexing_solution.cu
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
// SOLUTION 1: 2D Thread Indexing
// ============================================================================
__global__ void index2D(float *output, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    
    int globalX = bx * blockDim.x + tx;
    int globalY = ty;  // 1D grid, so globalY = threadIdx.y
    
    if (globalX < width && globalY < height) {
        int idx = globalY * width + globalX;
        output[idx] = globalX + globalY;
    }
}

// ============================================================================
// SOLUTION 2: Full 2D Grid and 2D Blocks
// ============================================================================
__global__ void matrixProcess(float *matrix, int width, int height, float value) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        matrix[idx] = value * (row + col);
    }
}

// ============================================================================
// SOLUTION 3: 3D Thread Indexing for Volume Data
// ============================================================================
__global__ void volumeProcess(float *volume, int width, int height, int depth) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int z = bz * blockDim.z + tz;
    
    if (x < width && y < height && z < depth) {
        int idx = z * height * width + y * width + x;
        volume[idx] = x + y + z;
    }
}

// ============================================================================
// SOLUTION 4: Grid-Stride Loop
// ============================================================================
__global__ void gridStrideAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// SOLUTION 5: Diagonal Matrix Extraction
// ============================================================================
__global__ void extractDiagonal(float *matrix, float *diagonal, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only threads on diagonal write (where row == col)
    if (row == col && row < height && col < width) {
        int matrixIdx = row * width + col;
        diagonal[row] = matrix[matrixIdx];
    }
}

// Verification functions
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

int main() {
    printf("=== Thread Indexing - SOLUTIONS ===\n\n");
    
    // Test 1: 2D Thread Indexing
    printf("1. 2D Thread Indexing\n");
    int width = 64, height = 64;
    size_t size2D = width * height * sizeof(float);
    
    float *h_output2D = (float *)malloc(size2D);
    float *d_output2D;
    CUDA_CHECK(cudaMalloc(&d_output2D, size2D));
    
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x);
    
    index2D<<<gridSize2D, blockSize2D>>>(d_output2D, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output2D, d_output2D, size2D, cudaMemcpyDeviceToHost));
    verify2DPattern(h_output2D, width, height);
    printf("   Sample: output[0,0]=%.0f, output[1,2]=%.0f, output[3,3]=%.0f\n\n",
           h_output2D[0], h_output2D[2 * width + 1], h_output2D[3 * width + 3]);
    
    // Test 2: Full 2D Grid and 2D Blocks
    printf("2. Full 2D Grid and 2D Blocks\n");
    
    float *h_matrix = (float *)malloc(size2D);
    float *d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, size2D));
    
    dim3 gridSize2DFull((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize2DFull(16, 16);
    
    matrixProcess<<<gridSize2DFull, blockSize2DFull>>>(d_matrix, width, height, 2.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, size2D, cudaMemcpyDeviceToHost));
    printf("   Sample: matrix[0,0]=%.0f, matrix[1,2]=%.0f, matrix[5,5]=%.0f\n\n",
           h_matrix[0], h_matrix[2 * width + 1], h_matrix[5 * width + 5]);
    
    // Test 3: 3D Thread Indexing
    printf("3. 3D Thread Indexing for Volume Data\n");
    
    int vWidth = 32, vHeight = 32, vDepth = 16;
    size_t size3D = vWidth * vHeight * vDepth * sizeof(float);
    
    float *h_volume = (float *)malloc(size3D);
    float *d_volume;
    CUDA_CHECK(cudaMalloc(&d_volume, size3D));
    
    dim3 blockSize3D(8, 8, 4);
    dim3 gridSize3D((vWidth + 7) / 8, (vHeight + 7) / 8, (vDepth + 3) / 4);
    
    volumeProcess<<<gridSize3D, blockSize3D>>>(d_volume, vWidth, vHeight, vDepth);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_volume, d_volume, size3D, cudaMemcpyDeviceToHost));
    verify3DPattern(h_volume, vWidth, vHeight, vDepth);
    printf("   Sample: volume[0,0,0]=%.0f, volume[1,2,3]=%.0f\n\n",
           h_volume[0], h_volume[3 * vHeight * vWidth + 2 * vWidth + 1]);
    
    // Test 4: Grid-Stride Loop
    printf("4. Grid-Stride Loop\n");
    
    int n = 100000;
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
    
    int threadsPerBlock = 32;
    int blocksPerGrid = 10;
    
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
    printf("   [%s] Grid-stride loop: All elements = 3.0\n\n",
           correct ? "PASS" : "FAIL");
    
    // Test 5: Diagonal Extraction
    printf("5. Diagonal Matrix Extraction\n");
    
    int diagSize = (width < height) ? width : height;
    float *h_diagonal = (float *)malloc(diagSize * sizeof(float));
    float *d_diagonal;
    CUDA_CHECK(cudaMalloc(&d_diagonal, diagSize * sizeof(float)));
    
    // Initialize matrix: matrix[i,j] = i*100 + j
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
    
    printf("   Diagonal values (first 10): ");
    for (int i = 0; i < 10 && i < diagSize; i++) {
        printf("%.0f ", h_diagonal[i]);
    }
    printf("\n   Expected: 0, 101, 202, 303, 404, 505, 606, 707, 808, 909\n\n");
    
    // Cleanup
    cudaFree(d_output2D);
    cudaFree(d_matrix);
    cudaFree(d_volume);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_diagonal);
    free(h_output2D);
    free(h_matrix);
    free(h_volume);
    free(h_A); free(h_B); free(h_C);
    free(h_diagonal);
    
    printf("=== All Solutions Complete! ===\n");
    
    return 0;
}
