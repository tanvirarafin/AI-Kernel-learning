# Memory Hierarchy in CUDA

CUDA GPUs have a memory hierarchy that significantly impacts performance. Understanding when and how to use each memory type is crucial for optimization.

## Memory Types

| Memory Type | Scope | Speed | Size | Cached |
|-------------|-------|-------|------|--------|
| Global      | Grid  | Slow  | GB   | L1/L2  |
| Shared      | Block | Fast  | KB   | Yes    |
| Local       | Thread| Slow  | GB   | L1/L2  |
| Constant    | Grid  | Fast* | 64KB | Yes    |
| Registers   | Thread| Fastest| 255  | N/A    |

*Constant memory is fast when all threads read the same address

## Levels

### Level 1: Global Memory Basics
Learn to allocate and transfer data to/from global memory.

### Level 2: Shared Memory Introduction
Use shared memory for thread-block level data sharing.

### Level 3: Local Memory and Register Pressure
Understand when variables spill to local memory.

### Level 4: Constant Memory Optimization
Use constant memory for read-only uniform data.

### Level 5: Memory Hierarchy Combined
Combine all memory types for optimal performance.
