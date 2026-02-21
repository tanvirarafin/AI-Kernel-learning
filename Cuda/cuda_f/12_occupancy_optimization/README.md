# Occupancy Optimization in CUDA

Master occupancy tuning for maximum GPU utilization.

## Concepts Covered
- Occupancy calculation
- Register usage
- Shared memory limits
- Block size optimization
- Launch configuration

## Levels

### Level 1: Occupancy Basics (`level1_occupancy_basics.cu`)
- **Goal**: Understand occupancy calculation
- **Missing**: cudaOccupancyMaxActiveBlocksPerMultiprocessor
- **Concepts**: Theoretical vs achieved occupancy

### Level 2: Register Optimization (`level2_register_tuning.cu`)
- **Goal**: Optimize register usage
- **Missing**: --maxrregcount, register spilling
- **Concepts**: Register pressure, spill to local memory

### Level 3: Block Size Tuning (`level3_block_size.cu`)
- **Goal**: Find optimal block size
- **Missing**: Occupancy query API
- **Concepts**: Block size vs occupancy

### Level 4: Shared Memory Impact (`level4_shared_memory.cu`)
- **Goal**: Understand shared memory limits
- **Missing**: Shared memory per block tuning
- **Concepts**: Shared memory partitioning

### Level 5: Launch Configuration (`level5_launch_config.cu`)
- **Goal**: Optimize launch parameters
- **Missing**: cudaOccupancyMaxPotentialBlockSize
- **Concepts**: Automatic occupancy-based configuration

## Key Principles
1. **Occupancy**: Active warps / Maximum warps
2. **Registers**: 255 per thread max, affects occupancy
3. **Shared Memory**: 48KB-96KB per SM (depends on GPU)
4. **Block Size**: Must be multiple of warpSize (32)
5. **Sweet Spot**: Often 64-256 threads per block

## Occupancy Calculator
Use the CUDA Occupancy Calculator spreadsheet or:
- cudaOccupancyMaxActiveBlocksPerMultiprocessor
- cudaOccupancyMaxPotentialBlockSize
