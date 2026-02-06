#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// Counting phase: count occurrences of each digit
__global__ void countDigits(int* input, int* counts, int n, int shift, int mask) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Shared memory for block-level counts
    extern __shared__ int sCounts[];
    
    // Initialize shared memory
    for(int i = tid; i < 256; i += blockDim.x) {  // Assuming 8-bit radix (256 bins)
        sCounts[i] = 0;
    }
    __syncthreads();
    
    // Count digits in parallel
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        atomicAdd(&sCounts[digit], 1);
    }
    __syncthreads();
    
    // Write block counts to global memory
    for(int i = tid; i < 256; i += blockDim.x) {
        counts[bid * 256 + i] = sCounts[i];
    }
}

// Prefix sum for calculating starting positions
__global__ void prefixSum(int* counts, int* prefixSums, int numBlocks, int numBins) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if(bid == 0 && tid < numBins) {
        // Calculate prefix sums across blocks
        int total = 0;
        for(int i = 0; i < numBlocks; i++) {
            int temp = counts[i * numBins + tid];
            prefixSums[i * numBins + tid] = total;
            total += temp;
        }
    }
}

// Scatter phase: place elements in correct positions
__global__ void scatter(int* input, int* output, int* prefixSums, 
                       int n, int shift, int mask, int numBlocks) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if(idx < n) {
        int value = input[idx];
        int digit = (value >> shift) & mask;
        
        // Calculate position using prefix sum
        int pos = prefixSums[bid * 256 + digit];
        
        // Add offset within block
        extern __shared__ int sCounts[];
        for(int i = 0; i < 256; i++) sCounts[i] = 0;  // Initialize
        __syncthreads();
        
        // Count how many of the same digit appear before this thread in the block
        for(int i = 0; i < blockDim.x && (bid * blockDim.x + i) < n; i++) {
            int otherValue = input[bid * blockDim.x + i];
            int otherDigit = (otherValue >> shift) & mask;
            if(otherDigit == digit && i < tid) {
                atomicAdd(&sCounts[digit], 1);
            }
        }
        __syncthreads();
        
        // Calculate final position
        int finalPos = prefixSums[bid * 256 + digit] + sCounts[digit];
        output[finalPos] = value;
    }
}

// Complete radix sort pass
__global__ void radixSortPass(int* input, int* output, int* tempStorage, 
                            int n, int shift, int mask) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Shared memory for counts and positions
    extern __shared__ int sharedMem[];
    int* sCounts = sharedMem;
    int* sPositions = &sharedMem[256]; // Assuming 256 bins for 8-bit radix
    
    // Initialize counts
    for(int i = tid; i < 256; i += blockDim.x) {
        sCounts[i] = 0;
    }
    __syncthreads();
    
    // Count digits in this block
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        atomicAdd(&sCounts[digit], 1);
    }
    __syncthreads();
    
    // Store block-level counts in global memory
    if(tid < 256) {
        tempStorage[bid * 256 + tid] = sCounts[tid];
    }
    __syncthreads();
    
    // After global prefix sum is computed externally:
    __syncthreads();
    
    // Calculate local positions within block
    if(idx < n) {
        int digit = (input[idx] >> shift) & mask;
        
        // Reset position counters
        if(tid < 256) {
            sPositions[digit] = 0;
        }
        __syncthreads();
        
        // Calculate local offset
        if(digit == (input[idx] >> shift) & mask) {
            // Count how many of the same digit come before this element in the block
            for(int i = 0; i < tid && (bid * blockDim.x + i) < n; i++) {
                if(((input[bid * blockDim.x + i] >> shift) & mask) == digit) {
                    sPositions[digit]++;
                }
            }
        }
        __syncthreads();
        
        // Calculate final position
        int globalPos = tempStorage[bid * 256 + digit] + sPositions[digit];
        output[globalPos] = input[idx];
    }
}

// Simple bitonic sort for small arrays as a comparison
__global__ void bitonicSortStep(int* arr, int n, int k, int j) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = idx ^ j;  // XOR operation
    
    if(ixj > idx && idx < n && ixj < n) {
        if((idx & k) == 0) {
            // Sort in ascending order
            if(arr[idx] > arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            // Sort in descending order
            if(arr[idx] < arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void gpuBitonicSort(int* d_arr, int n) {
    // Ensure n is a power of 2
    int powerOf2 = 1;
    while(powerOf2 < n) powerOf2 <<= 1;
    
    dim3 block(256);
    dim3 grid((powerOf2 + block.x - 1) / block.x);
    
    // Perform bitonic sort
    for(int k = 2; k <= powerOf2; k <<= 1) {
        for(int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<grid, block>>>(d_arr, n, k, j);
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const int numBits = 32;      // For 32-bit integers
    const int radixBits = 8;     // 8 bits at a time (256 bins)
    const int numRadixPasses = numBits / radixBits;  // 4 passes for 32-bit integers
    const int mask = (1 << radixBits) - 1;  // 0xFF for 8-bit radix
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    
    // Host memory allocation
    std::vector<int> h_input(N);
    std::vector<int> h_output(N);
    
    // Initialize input with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000000);
    
    for(int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }
    
    // Device memory allocation
    int *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_temp, gridSize * 256 * sizeof(int));  // Temporary storage for counts
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Perform radix sort with multiple passes
    int* current = d_input;
    int* next = d_output;
    
    for(int pass = 0; pass < numRadixPasses; pass++) {
        int shift = pass * radixBits;
        
        // Count phase
        countDigits<<<gridSize, blockSize, 256 * sizeof(int)>>>(current, d_temp, N, shift, mask);
        cudaDeviceSynchronize();
        
        // Prefix sum phase (simplified - in practice, this would be a more complex scan)
        prefixSum<<<1, 256>>>(d_temp, d_temp, gridSize, 256);
        cudaDeviceSynchronize();
        
        // Scatter phase
        scatter<<<gridSize, blockSize, 256 * sizeof(int)>>>(current, next, d_temp, N, shift, mask, gridSize);
        cudaDeviceSynchronize();
        
        // Swap pointers for next iteration
        int* temp_ptr = current;
        current = next;
        next = temp_ptr;
    }
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), current, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify the result is sorted
    bool isSorted = std::is_sorted(h_output.begin(), h_output.end());
    std::cout << "Radix sort result is " << (isSorted ? "correct" : "incorrect") << std::endl;
    
    // Print first 10 elements
    std::cout << "First 10 elements after radix sort: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Print last 10 elements
    std::cout << "Last 10 elements after radix sort: ";
    for(int i = N-10; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    return 0;
}