/*
 * Level 1: Basic Thread Indexing - Kernel 4: Block Index Calculation
 *
 * This kernel demonstrates calculating block IDs and thread positions
 * within a 2D grid configuration.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// KERNEL 4: Block Index Calculation (Challenge)
// Calculate which block a thread belongs to in a 2D grid
// ============================================================================
__global__ void blockInfo2D(int *blockIds, int gridWidth, int gridHeight) {
    // Get this thread's block coordinates
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    // Get this thread's position within the block
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Calculate unique block ID (row-major order)
    int blockId = blockY * gridWidth + blockX;

    // Each thread in the block writes the same block ID
    int idx = blockY * gridWidth + blockX;
    blockIds[idx] = blockId;
}

// Utility functions
bool verifyBlockInfo(int *result, int gridWidth, int gridHeight) {
    for (int blockY = 0; blockY < gridHeight; blockY++) {
        for (int blockX = 0; blockX < gridWidth; blockX++) {
            int idx = blockY * gridWidth + blockX;
            int expectedBlockId = blockY * gridWidth + blockX;
            if (result[idx] != expectedBlockId) return false;
        }
    }
    return true;
}

void printBlockIds(int *arr, int gridWidth, int gridHeight, const char *label) {
    printf("%s (block IDs):\n", label);
    for (int blockY = 0; blockY < gridHeight && blockY < 5; blockY++) {
        printf("  Block Row %d: ", blockY);
        for (int blockX = 0; blockX < gridWidth && blockX < 5; blockX++) {
            int idx = blockY * gridWidth + blockX;
            printf("%4d ", arr[idx]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Thread Hierarchy Level 1: Block Index Calculation ===\n\n");

    // Test block info calculation
    const int GRID_WIDTH = 4, GRID_HEIGHT = 4;
    const int NUM_BLOCKS = GRID_WIDTH * GRID_HEIGHT;
    const int BLOCKS_PER_ROW = GRID_WIDTH;
    
    int *d_blockIds;
    cudaMalloc(&d_blockIds, NUM_BLOCKS * sizeof(int));
    cudaMemset(d_blockIds, 0, NUM_BLOCKS * sizeof(int));

    // Use a small block size for clarity
    dim3 block2D(8, 8);
    dim3 grid2D(GRID_WIDTH, GRID_HEIGHT);

    printf("Launching blockInfo2D kernel...\n");
    printf("  Grid size: %d x %d blocks\n", GRID_WIDTH, GRID_HEIGHT);
    printf("  Total blocks: %d\n", NUM_BLOCKS);
    printf("  Block size: %d x %d threads\n\n", block2D.x, block2D.y);

    blockInfo2D<<<grid2D, block2D>>>(d_blockIds, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_blockIds);
        return 1;
    }

    int *h_result = (int*)malloc(NUM_BLOCKS * sizeof(int));
    cudaMemcpy(h_result, d_blockIds, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    printBlockIds(h_result, GRID_WIDTH, GRID_HEIGHT, "Results");

    if (verifyBlockInfo(h_result, GRID_WIDTH, GRID_HEIGHT)) {
        printf("\n✓ Block info calculation PASSED\n");
    } else {
        printf("\n✗ Block info calculation FAILED - Check your block ID calculation\n");
    }

    // Cleanup
    free(h_result);
    cudaFree(d_blockIds);

    printf("\n=== Level 1.4 Complete ===\n");
    printf("Next: Try level2_grid_stride.cu for scalable kernel patterns\n");

    return 0;
}
