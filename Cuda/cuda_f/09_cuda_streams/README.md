# CUDA Streams

Master concurrent execution with CUDA streams for improved throughput.

## Concepts Covered
- Stream creation and management
- Concurrent kernel execution
- Async memory transfers
- Stream synchronization
- Multi-stream patterns

## Levels

### Level 1: Stream Basics (`level1_stream_basics.cu`)
- **Goal**: Learn stream creation and basic usage
- **Missing**: cudaStreamCreate, cudaStreamSynchronize
- **Concepts**: Stream abstraction, default vs non-default streams
- **API**: cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize

### Level 2: Concurrent Kernels (`level2_concurrent_kernels.cu`)
- **Goal**: Execute multiple kernels concurrently
- **Missing**: Multi-stream kernel launches
- **Concepts**: Kernel concurrency, GPU utilization
- **API**: Multiple streams, concurrent execution

### Level 3: Async Memory Transfers (`level3_async_memcpy.cu`)
- **Goal**: Overlap computation with data transfer
- **Missing**: cudaMemcpyAsync, pinned memory
- **Concepts**: H2D/D2H async transfers, pinned memory
- **API**: cudaMallocHost, cudaMemcpyAsync

### Level 4: Stream Callbacks (`level4_stream_callbacks.cu`)
- **Goal**: Use callbacks for stream synchronization
- **Missing**: cudaStreamAddCallback
- **Concepts**: Event-driven programming, completion notification
- **API**: cudaStreamAddCallback

### Level 5: Advanced Stream Patterns (`level5_advanced_streams.cu`)
- **Goal**: Complex multi-stream algorithms
- **Missing**: Stream prioritization, dependency management
- **Concepts**: Priority streams, multi-GPU preparation
- **API**: cudaStreamCreateWithPriority

## Compilation
```bash
nvcc level1_stream_basics.cu -o level1
nvcc level2_concurrent_kernels.cu -o level2
nvcc level3_async_memcpy.cu -o level3
nvcc level4_stream_callbacks.cu -o level4
nvcc level5_advanced_streams.cu -o level5
```

## Key Principles
1. **Default Stream**: Blocking, synchronized with host
2. **Non-default Streams**: Can execute concurrently
3. **Pinned Memory**: Required for async transfers
4. **Synchronization**: cudaStreamSynchronize or events
5. **Resources**: Concurrent kernels share GPU resources

## Important Notes
- Concurrent execution requires compute capability 3.0+
- Use cudaDeviceCanAccessPeer for multi-GPU
- Pinned memory is limited - use carefully
- Stream priority is hardware-dependent
