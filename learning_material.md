# Learning Material

A curated list of resources for GPU programming, concurrency, and NVIDIA libraries (CuTe / CUTLASS). Use this as a quick reference for tutorials, docs, and deeper reading.

---

## CUDA
1. NVIDIA CUDA Programming Guide  
   https://docs.nvidia.com/cuda/cuda-programming-guide/  
   - Official programming reference. Required reading for CUDA concepts, memory model, and performance guidance.

2. Programming Massively Parallel Processors — Notes (Steven Gong)  
   https://stevengong.co/notes/Programming-Massively-Parallel-Processors  
   - Concise notes and explanations covering parallel programming fundamentals and GPU-specific design.

3. NVIDIA Developer Blog — "An even easier introduction to CUDA"  
   https://developer.nvidia.com/blog/even-easier-introduction-cuda/  
   - Beginner-friendly introduction and practical examples.

4. CUDA Tutorial (Read the Docs)  
   https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/  
   - Hands-on tutorial covering basic CUDA setup and simple kernels.

5. CUDA MMM (Matrix-Matrix Multiplication) — Simon Boehm  
   https://siboehm.com/articles/22/CUDA-MMM  
   - Walkthrough of optimizing matrix multiply on CUDA — great for learning tiling/shared-memory strategies.

---

## Concurrency (C++)
1. Modern Multithreading and Concurrency in C++ (Educative)  
   https://www.educative.io/blog/modern-multithreading-and-concurrency-in-cpp  
   - Practical overview of modern C++ concurrency features and patterns.

2. GeeksforGeeks — C++ Concurrency  
   https://www.geeksforgeeks.org/cpp/cpp-concurrency/  
   - Reference-style examples for threads, mutexes, condition variables, and more.

3. Medium — "Concurrency in C++: multithreading and concurrent programming"  
   https://medium.com/@lfoster49203/concurrency-in-c-multithreading-and-concurrent-programming-ccf81110c284  
   - A readable primer on common concurrency pitfalls and idioms.

4. University of Chicago — Concurrency lab (archive)  
   https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/labs/lab6/  
   - Educational lab exercises for hands-on learning.

5. "C++ Concurrency In Action" (PDF)  
   https://www.bogotobogo.com/cplusplus/files/CplusplusConcurrencyInAction_PracticalMultithreading.pdf  
   - Deep dive book covering theory and practical examples.

---

## CuTe (CUDA Template Library for Tensors)
1. "Hello Layout" — CuTe layout tutorial  
   https://www.dcbaslani.xyz/blog.html?post=01_hello_layout  
   - Intro to CuTe layout abstractions and indexing.

2. CuTe C++ Quickstart (CUTLASS docs)  
   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html  
   - Official quickstart for C++ CuTe usage.

3. CuTe Python DSL Quickstart (CUTLASS docs)  
   https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html  
   - Quickstart for the Python-based domain-specific language.

---

## CUTLASS
1. CUTLASS GitHub (NVIDIA)  
   https://github.com/NVIDIA/cutlass  
   - Source, examples, and templates for high-performance GEMM and tensor ops.

2. CUTLASS Documentation (NVIDIA)  
   https://docs.nvidia.com/cutlass/latest/  
   - API docs, design guides, and developer notes.

3. CUDA Mode Notes — Lecture (Christian J. Mills)  
   https://christianjmills.com/posts/cuda-mode-notes/lecture-015/  
   - Lecture notes and explanations relevant to CUDA/CUTLASS concepts.

4. CUTLASS tutorial — WMMA / Hopper (Colfax Research)  
   https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/  
   - Practical tutorial focused on using CUTLASS with WMMA and Hopper architectures.

---

## Usage tips
- Start with the CUDA Programming Guide to understand the memory hierarchy and execution model.  
- Practice by implementing and optimizing a simple GEMM; use the CUDA MMM and CUTLASS examples to compare approaches.  
- Use Concurrency/C++ resources to write safe host-side code that efficiently drives GPU workloads (threads, streams, and synchronization).  
- When performance matters, profile with NVIDIA Nsight (NCU / Nsys) and iterate: optimize memory access patterns, then compute.

---

## Notes / TODO
- Consider adding Triton, Tensor Cores, and profiling resources (Nsight Compute / Nsight Systems) to expand the list.  
- Rename file to `learning_material.md` if you prefer the corrected spelling.