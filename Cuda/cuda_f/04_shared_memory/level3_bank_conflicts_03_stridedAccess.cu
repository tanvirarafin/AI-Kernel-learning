/**
 * Strided Access - Kernel 3 from level3_bank_conflicts.cu
 * 
 * This kernel demonstrates how stride values affect bank conflicts.
 * Different strides cause different levels of bank conflicts.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256
#define NUM_BANKS 32

__global__ void stridedAccess(float *input, float *output, int stride) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data
    sharedData[tid] = input[idx];
    __syncthreads();

    // Strided access - conflict depends on stride value!
    // stride = 1: No conflict (consecutive)
    // stride = 2: 2-way conflict
    // stride = 32: 32-way conflict (worst!)
    // stride = 33: No conflict (coprime with 32)
    int accessIdx = (tid * stride) % BLOCK_SIZE;
    output[idx] = sharedData[accessIdx];
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 0.5f;
    }
}

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
    printf("=== Strided Access Bank Conflict Analysis ===\n\n");

    printf("Bank Conflict Analysis for different strides:\n");
    int strides[] = {1, 2, 4, 8, 16, 32, 33, 64};
    for (int i = 0; i < 8; i++) {
        analyzeBankConflicts(strides[i]);
    }

    printf("\n=== Key Takeaways ===\n");
    printf("- stride = 1: No conflict (consecutive access)\n");
    printf("- stride = 32: 32-way conflict (worst case!)\n");
    printf("- stride = 33: No conflict (coprime with 32)\n");
    printf("- Stride values that are multiples of 32 cause worst conflicts\n");

    return 0;
}
