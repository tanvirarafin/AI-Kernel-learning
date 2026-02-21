# CUDA Mastery Exercises Makefile
# Simplifies compilation and testing of exercises

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_50

# Directories
BASICS_DIR = 01_basics
MEMORY_DIR = 02_memory_model
SHARED_DIR = 03_shared_memory
SYNC_DIR = 04_synchronization
OPT_DIR = 05_optimization
ADVANCED_DIR = 06_advanced
SOLUTIONS_DIR = solutions

# Exercise files
EXERCISES = \
    $(BASICS_DIR)/04_exercises_vector_ops \
    $(BASICS_DIR)/05_exercises_thread_indexing \
    $(MEMORY_DIR)/01_exercises_memory_coalescing \
    $(SHARED_DIR)/01_exercises_shared_memory_basics \
    $(SYNC_DIR)/01_exercises_sync_atomics \
    $(OPT_DIR)/01_exercises_occupancy_tuning \
    $(ADVANCED_DIR)/01_exercises_cuda_streams \
    $(ADVANCED_DIR)/02_exercises_unified_memory

# Solution files
SOLUTIONS = \
    $(SOLUTIONS_DIR)/01_vector_ops_solution \
    $(SOLUTIONS_DIR)/02_thread_indexing_solution

# Default target
all: $(EXERCISES)

# Build all exercises
$(BASICS_DIR)/04_exercises_vector_ops: $(BASICS_DIR)/04_exercises_vector_ops.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BASICS_DIR)/05_exercises_thread_indexing: $(BASICS_DIR)/05_exercises_thread_indexing.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(MEMORY_DIR)/01_exercises_memory_coalescing: $(MEMORY_DIR)/01_exercises_memory_coalescing.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SHARED_DIR)/01_exercises_shared_memory_basics: $(SHARED_DIR)/01_exercises_shared_memory_basics.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SYNC_DIR)/01_exercises_sync_atomics: $(SYNC_DIR)/01_exercises_sync_atomics.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(OPT_DIR)/01_exercises_occupancy_tuning: $(OPT_DIR)/01_exercises_occupancy_tuning.cu
	$(NVCC) $(NVCCFLAGS) -ptxas-options=-v -o $@ $<

$(ADVANCED_DIR)/01_exercises_cuda_streams: $(ADVANCED_DIR)/01_exercises_cuda_streams.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(ADVANCED_DIR)/02_exercises_unified_memory: $(ADVANCED_DIR)/02_exercises_unified_memory.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Build solutions
solutions: $(SOLUTIONS)

$(SOLUTIONS_DIR)/01_vector_ops_solution: $(SOLUTIONS_DIR)/01_vector_ops_solution.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(SOLUTIONS_DIR)/02_thread_indexing_solution: $(SOLUTIONS_DIR)/02_thread_indexing_solution.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Run individual exercises
run-vector-ops: $(BASICS_DIR)/04_exercises_vector_ops
	./$(BASICS_DIR)/04_exercises_vector_ops

run-indexing: $(BASICS_DIR)/05_exercises_thread_indexing
	./$(BASICS_DIR)/05_exercises_thread_indexing

run-coalescing: $(MEMORY_DIR)/01_exercises_memory_coalescing
	./$(MEMORY_DIR)/01_exercises_memory_coalescing

run-shared: $(SHARED_DIR)/01_exercises_shared_memory_basics
	./$(SHARED_DIR)/01_exercises_shared_memory_basics

run-sync: $(SYNC_DIR)/01_exercises_sync_atomics
	./$(SYNC_DIR)/01_exercises_sync_atomics

run-occupancy: $(OPT_DIR)/01_exercises_occupancy_tuning
	./$(OPT_DIR)/01_exercises_occupancy_tuning

run-streams: $(ADVANCED_DIR)/01_exercises_cuda_streams
	./$(ADVANCED_DIR)/01_exercises_cuda_streams

run-unified: $(ADVANCED_DIR)/02_exercises_unified_memory
	./$(ADVANCED_DIR)/02_exercises_unified_memory

# Run solutions
run-sol-vector: $(SOLUTIONS_DIR)/01_vector_ops_solution
	./$(SOLUTIONS_DIR)/01_vector_ops_solution

run-sol-indexing: $(SOLUTIONS_DIR)/02_thread_indexing_solution
	./$(SOLUTIONS_DIR)/02_thread_indexing_solution

# Clean build artifacts
clean:
	rm -f $(BASICS_DIR)/*_exercises_*
	rm -f $(MEMORY_DIR)/*_exercises_*
	rm -f $(SHARED_DIR)/*_exercises_*
	rm -f $(SYNC_DIR)/*_exercises_*
	rm -f $(OPT_DIR)/*_exercises_*
	rm -f $(ADVANCED_DIR)/*_exercises_*
	rm -f $(SOLUTIONS_DIR)/*_solution

# Help target
help:
	@echo "CUDA Mastery Exercises - Makefile Targets"
	@echo "=========================================="
	@echo ""
	@echo "Build all exercises:"
	@echo "  make all"
	@echo ""
	@echo "Build solutions:"
	@echo "  make solutions"
	@echo ""
	@echo "Run individual exercises:"
	@echo "  make run-vector-ops      - Vector operations (1.4)"
	@echo "  make run-indexing        - Thread indexing (1.5)"
	@echo "  make run-coalescing      - Memory coalescing (2.1)"
	@echo "  make run-shared          - Shared memory (3.1)"
	@echo "  make run-sync            - Sync and atomics (4.1)"
	@echo "  make run-occupancy       - Occupancy tuning (5.1)"
	@echo "  make run-streams         - CUDA streams (6.1)"
	@echo "  make run-unified         - Unified memory (6.2)"
	@echo ""
	@echo "Run solutions:"
	@echo "  make run-sol-vector      - Vector ops solution"
	@echo "  make run-sol-indexing    - Thread indexing solution"
	@echo ""
	@echo "Clean:"
	@echo "  make clean               - Remove all built files"
	@echo ""
	@echo "Compile with verbose PTX (for register info):"
	@echo "  nvcc -ptxas-options=-v -o output file.cu"

.PHONY: all solutions clean help run-vector-ops run-indexing run-coalescing \
        run-shared run-sync run-occupancy run-streams run-unified \
        run-sol-vector run-sol-indexing
