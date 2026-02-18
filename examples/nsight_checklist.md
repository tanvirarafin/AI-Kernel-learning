# Nsight Profiling Checklist

A compact, practical checklist for using NVIDIA Nsight Compute (NCU) and Nsight Systems (NSYS) to profile GPU kernels, interpret the results, and iterate on optimizations. Keep this file in your repository's `examples/` folder so you can copy commands and notes into experiments and PR descriptions.

---

## Table of Contents
- Overview
- Preparation
- Quick profiling commands
  - Nsight Compute (NCU)
  - Nsight Systems (NSYS)
- Recommended workflow (step-by-step)
- Key metrics to inspect and what they mean
- Common bottlenecks and targeted actions
- NVTX instrumentation and range usage
- Practical tips and gotchas

---

## Overview
- Use Nsight Systems for system-level timeline analysis (CPU/GPU interactions, launch/host stalls, concurrency).
- Use Nsight Compute for deep kernel-level metrics (memory throughput, occupancy, source counters, line-level hotspots).
- Profile representative workloads (realistic shapes, batch sizes) and run warm-up iterations before measuring.
- Always compile kernels with line information when you want source-level insights:
  - Add `-lineinfo` to `nvcc` for device source line correlation (or appropriate flags for your build system).

---

## Preparation checklist
- Build with debug/line info for source mapping:
```AI-Kernel-learning/examples/nsight_checklist.md#L1-4
# nvcc compile flags
nvcc -O3 -lineinfo -arch=sm_80 -std=c++17 my_kernel.cu -o my_kernel
```
- Add NVTX ranges in host code for meaningful timeline regions (see NVTX section).
- Make sure the workload is large enough to be measurable but representative.
- Minimize background activity on the host machine (close unrelated programs, isolate GPUs if possible).
- Warm-up GPU (launch kernels few times before measuring) to avoid measuring one-time initialization costs.

---

## Quick profiling commands

### Nsight Compute (NCU)
- Basic (collect summary, fast):
```AI-Kernel-learning/examples/nsight_checklist.md#L21-26
# Profile and create a report (default set)
ncu -o report ./my_application
```

- Full metrics (expensive but thorough):
```AI-Kernel-learning/examples/nsight_checklist.md#L27-33
# Collect full metrics (many counters)
ncu --set full -o report_full ./my_application
```

- Profile a specific kernel by name:
```AI-Kernel-learning/examples/nsight_checklist.md#L34-38
# Profile kernels whose name matches "myKernel"
ncu --kernel-name "myKernel" -o report_myKernel ./my_application
```

- Profile only the first kernel launch (avoid long replay times):
```AI-Kernel-learning/examples/nsight_checklist.md#L39-43
# Collect metrics for first kernel launch only
ncu --kernel-id :::1 -o report_first ./my_application
```

- Collect a specific subset of metrics:
```AI-Kernel-learning/examples/nsight_checklist.md#L44-48
# Pick specific metrics (example: achieved_occupancy, dram_read_throughput)
ncu --metrics achieved_occupancy,dram__bytes_read.throughput -o small_report ./my_application
```

- Use NVTX filtering to focus on regions:
```AI-Kernel-learning/examples/nsight_checklist.md#L49-53
# Use NVTX ranges and include only relevant NVTX tags
ncu --nvtx --nvtx-include "my_range" -o nvtx_report ./my_application
```

Notes:
- Use `--target-processes all` if profiling multi-process workloads.
- Use `--import`/`--export` for session reuse (advanced workflows).

### Nsight Systems (NSYS)
- Timeline profiling (default capture):
```AI-Kernel-learning/examples/nsight_checklist.md#L61-65
# Capture a system-level timeline
nsys profile --stats=true -o timeline_report ./my_application
```

- Limit capture to CUDA/NVTX events and avoid excessive CPU traces:
```AI-Kernel-learning/examples/nsight_checklist.md#L66-70
# Capture only NVTX and CUDA events
nsys profile -o timeline_nvtx -t nvtx,cuda ./my_application
```

- Use capture-range with NVTX markers (start/stop capture programmatically):
```AI-Kernel-learning/examples/nsight_checklist.md#L71-75
# Capture only when application uses cudaProfilerStart / cudaProfilerStop or NVTX ranges
nsys profile --capture-range=cudaProfilerApi -o partial_report ./my_application
```

---

## Recommended profiling workflow (concise)
1. Baseline run: run the app and ensure correctness + representative performance numbers.
2. Lightweight NCU run: `ncu -o quick ./app` to get a high-level picture and detect hot kernels.
3. NSYS timeline: `nsys profile -o timeline ./app` to see host/GPU interactions and concurrency issues.
4. Focused NCU: profile the top hot kernel(s) with `--kernel-name` or `--kernel-id` and `--set full` if necessary.
5. Iterate edits (tile sizes, shared memory, data layout, loop unrolling), re-profile, and compare reports.
6. Keep a journal of runs (script names, flags, inputs, environment) for reproducibility.

---

## Key metrics to inspect and how to interpret them

- Throughput & occupancy:
  - Achieved Occupancy: fraction of theoretical occupancy — low occupancy may indicate register/shared-memory pressure.
  - SM Efficiency / SM Activity: how busy SMs were during kernel execution.

- Memory:
  - DRAM Throughput (GB/s): how much bandwidth kernel used vs device peak — if close to peak, kernel is memory-bound.
  - L2 Hit Rate / Global Load Efficiency: shows whether loads are coalesced and using caches effectively.
  - Global Load/Store Transactions: large number indicates uncoalesced/staggered accesses.

- Compute:
  - FLOPs or Tensor Core utilization (when available): indicates compute intensity and whether Tensor Cores are used.
  - IPC / Warp Execution Efficiency: low values imply diverging warps or pipeline stalls.

- Stalls and warp state:
  - Warp Stall Reasons (memory, synchronization, execution dependency): primary reason for stalls helps choose optimization focus.
  - Warp Occupancy vs Active Warps: indicates inefficiency due to divergence or serialization.

- Source-level hotspots:
  - With `-lineinfo` build flag, NCU can attribute time to CUDA source lines; use this to find exact hotspots.

---

## Common bottlenecks → targeted actions

- Memory-bound (high DRAM throughput, low compute utilization)
  - Action: Reorder loops for coalescing, use vectorized loads, use shared memory tiling, reduce memory traffic (fusion).
  - Consider lower precision (FP16/BF16) if accuracy permits.

- Compute-bound (high compute usage, low memory usage)
  - Action: Use Tensor Cores (if suitable), increase ILP via loop unrolling, use warp-level intrinsics, ensure enough independent work per thread.

- Low occupancy / register pressure
  - Action: Reduce registers per thread (limit inlining, avoid large stack arrays), increase block size (if beneficial), split kernel into stages to reduce per-kernel resource usage.

- High latency due to host-GPU sync or small kernels
  - Action: Batch work, fuse kernels, use streams for concurrency, reduce host-side synchronization and frequent small kernel launches.

- Warp divergence
  - Action: Re-structure conditionals to minimize divergence, sort data by branch, use predication carefully.

---

## NVTX instrumentation
- NVTX lets you annotate host ranges which appear in NSYS timeline and NCU NVTX filters.
- Example (C++):
```AI-Kernel-learning/examples/nsight_checklist.md#L121-126
// C++ NVTX example (requires nvToolsExt)
nvtxRangeId_t id = nvtxRangeStartA("forward_pass");
// ... launch kernels, copy memory ...
nvtxRangeEnd(id);
```
- Example (Python + PyTorch):
```AI-Kernel-learning/examples/nsight_checklist.md#L127-131
import torch
import nvtx
with nvtx.annotate("forward", color="blue"):
    output = model(input)
```
- Use NVTX tags to focus timeline capture (nsys `--nvtx-include` or ncu `--nvtx-include`).

---

## Practical tips & gotchas
- Warm-up: GPU kernels often incur one-time overheads (JIT, allocator) — skip initial iterations.
- Noise: Run multiple iterations and take median/mean; disable SM-clock scaling (use nvidia-settings/persistence) for stable runs.
- File sizes: `--set full` can produce very large reports — use targeted metric sets when possible.
- Compare releases: Keep a baseline before making changes; compare NCU `--session`/`--import` results or export metrics into CSV for analysis.
- Use NVTX to separate phases (data transfer, kernel compute, epilogue) to see where time is spent.
- When profiling multi-GPU or multi-process apps, ensure you include `--target-processes all` for NCU and appropriate NSYS options.

---

## Example Nsight checklist you can copy into experiment notes
```AI-Kernel-learning/examples/nsight_checklist.md#L157-170
1) Build:
   nvcc -O3 -lineinfo -arch=sm_80 -std=c++17 src/my_app.cu -o bin/my_app

2) Warm-up:
   ./bin/my_app --mode warmup

3) Quick NCU:
   ncu -o ncu_quick ./bin/my_app --config args

4) Timeline NSYS:
   nsys profile -o nsys_timeline -t nvtx,cuda ./bin/my_app --config args

5) Focused NCU on top kernel:
   ncu --kernel-name "hotKernel" --set full -o ncu_hot ./bin/my_app --config args

6) Analyze and iterate:
   - Check DRAM throughput vs device peak
   - Inspect warp stall reasons
   - Use lineinfo to map to source lines and optimize
```

---

If you want, I can also:
- Add a small script to automate NCU/NSYS runs and save metadata (git sha, CUDA version, GPU model).
- Generate a short example NVTX-instrumented wrapper for your existing test harness.
- Provide a sample "compare report" checklist (which fields to snapshot before and after a change).

Pick one and I will add it to `examples/` as a convenience script or snippet.