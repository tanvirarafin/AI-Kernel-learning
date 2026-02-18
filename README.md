# AI Kernel Learning — Quick Start

This repository collects curated resources and learning paths for GPU kernel programming, concurrency, and NVIDIA libraries.

See the full resource list at:
- `learning_material.md` — a curated list of tutorials, docs, and examples (CUDA, CuTe, CUTLASS, Triton, Nsight, and more).

What you'll find in `learning_material.md`
- **CUDA** — Programming Guide, tutorials, and GEMM optimization walkthroughs to learn the execution model and memory hierarchy.
- **Concurrency (C++)** — Host-side multithreading and synchronization patterns to safely drive GPU workloads.
- **CuTe** — Layout and tensor-abstraction tutorials for composing efficient tensor operations.
- **CUTLASS** — Templates and examples for high-performance GEMM and Tensor Core usage.
- **Triton** — Python-first kernel programming with autotuning and block-model guidance.
- **NVIDIA Nsight Tools** — Links and tips for Nsight Compute (NCU) and Nsight Systems (Nsys) profiling.

Suggested next steps
1. Open `learning_material.md` and skim the sections to pick a starting area.
2. Start with the CUDA Programming Guide to understand the fundamentals.
3. Implement a small GEMM (naive → tiled → optimized) and profile with Nsight Compute / Nsight Systems.
4. Compare a Triton or CUTLASS implementation once correctness is verified.

Want me to add:
- Short example snippets (Triton kernel, simple CUDA GEMM)?
- A checklist of Nsight commands and profiling workflow?

I'm here to help — tell me which example or checklist you'd like next.