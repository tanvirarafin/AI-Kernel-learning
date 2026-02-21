# CUDA Mastery Path

A structured, **interactive** journey from CUDA basics to advanced optimization techniques.

## 🎯 How This Works

Each lesson has:
1. **Worked examples** - Study these first (e.g., `01_hello_cuda.cu`)
2. **Exercise files** - Fill in the TODO sections yourself ✏️
3. **Solutions** - Check when stuck (`solutions/` directory)

## 📚 Curriculum

### Level 1: Basics ✓
| File | Topic | Type |
|------|-------|------|
| `01_hello_cuda.cu` | Hello World | Example |
| `02_vector_add.cu` | Vector Addition | Example |
| `03_thread_indexing.cu` | Thread Indexing | Example |
| `04_exercises_vector_ops.cu` | Vector Operations | **Exercise** ✏️ |
| `05_exercises_thread_indexing.cu` | Indexing Challenges | **Exercise** ✏️ |

### Level 2: Memory Model ✓
| File | Topic | Type |
|------|-------|------|
| `01_memory_types.cu` | Memory Types | Example |
| `02_memory_coalescing.cu` | Memory Coalescing | Example |
| `01_exercises_memory_coalescing.cu` | Coalescing Practice | **Exercise** ✏️ |

### Level 3: Shared Memory ✓
| File | Topic | Type |
|------|-------|------|
| `01_shared_memory_basics.cu` | Shared Memory Basics | Example |
| `02_tiled_matrix_multiply.cu` | Tiled Matrix Multiply | Example |
| `01_exercises_shared_memory_basics.cu` | Shared Memory Practice | **Exercise** ✏️ |

### Level 4: Synchronization ✓
| File | Topic | Type |
|------|-------|------|
| `01_sync_and_atomics.cu` | Sync & Atomics | Example |
| `01_exercises_sync_atomics.cu` | Sync Practice | **Exercise** ✏️ |

### Level 5: Optimization ✓
| File | Topic | Type |
|------|-------|------|
| `01_occupancy_tuning.cu` | Occupancy Tuning | Example |
| `01_exercises_occupancy_tuning.cu` | Occupancy Practice | **Exercise** ✏️ |

### Level 6: Advanced ✓
| File | Topic | Type |
|------|-------|------|
| `01_cuda_streams.cu` | CUDA Streams | Example |
| `02_unified_memory.cu` | Unified Memory | Example |
| `01_exercises_cuda_streams.cu` | Streams Practice | **Exercise** ✏️ |
| `02_exercises_unified_memory.cu` | Unified Memory Practice | **Exercise** ✏️ |

## 🚀 Quick Start

```bash
cd cuda_mastery

# Build all exercises
make all

# Run first exercise
make run-vector-ops

# Or compile manually
nvcc -o ex 01_basics/04_exercises_vector_ops.cu
./ex
```

## 📁 Directory Structure

```
cuda_mastery/
├── 01_basics/           # Level 1: Basics
├── 02_memory_model/     # Level 2: Memory
├── 03_shared_memory/    # Level 3: Shared Memory
├── 04_synchronization/  # Level 4: Synchronization
├── 05_optimization/     # Level 5: Optimization
├── 06_advanced/         # Level 6: Advanced
├── solutions/           # Solution files
├── Makefile             # Build system
├── QUICK_REFERENCE.md   # Formula cheat sheet
└── exercises.md         # Detailed exercise descriptions
```

## 🔧 Prerequisites

- Basic C/C++ knowledge
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

## 📖 Resources

| Resource | Description |
|----------|-------------|
| `exercises.md` | Detailed exercise descriptions |
| `QUICK_REFERENCE.md` | Formulas and patterns cheat sheet |
| `cuda_syntax_reference.cu` | Complete CUDA syntax reference |
| `solutions/README.md` | How to use solutions |

## 🎯 Learning Path

1. **Read** the worked example first
2. **Try** the corresponding exercise
3. **Compile** and test your solution
4. **Verify** with the provided test cases
5. **Check** solutions if stuck
6. **Profile** with nvprof/nsys for advanced exercises

## 🛠️ Makefile Targets

```bash
make all              # Build all exercises
make run-vector-ops   # Run vector operations exercise
make run-indexing     # Run thread indexing exercise
make run-coalescing   # Run memory coalescing exercise
make run-shared       # Run shared memory exercise
make run-sync         # Run sync/atomics exercise
make run-occupancy    # Run occupancy tuning exercise
make run-streams      # Run CUDA streams exercise
make run-unified      # Run unified memory exercise
make solutions        # Build solution files
make clean            # Remove built files
```

---

**Ready to master CUDA?** Start with `01_basics/01_hello_cuda.cu` and work your way through! 🚀
