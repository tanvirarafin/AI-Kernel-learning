# CUDA Learning Curriculum - Makefile
# Usage: make module=01_thread_hierarchy
#        make level=level1_basic_indexing
#        make all
#        make clean

CUDA_ARCH ?= sm_70
NVCC := nvcc
NVCC_FLAGS := -arch=$(CUDA_ARCH) -O2

# Find all .cu files
CU_FILES := $(shell find . -name "*.cu" -type f)
EXECUTABLES := $(CU_FILES:.cu=)

.PHONY: all clean help modules

# Default target - build everything
all: $(EXECUTABLES)

# Pattern rule for compiling .cu files
%: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "  -> Created $@"

# Build specific module
module:
	@echo "Building module $(module)..."
	@find $(module) -name "*.cu" -exec $(NVCC) $(NVCC_FLAGS) {} -o {}.out \;

# Build specific level
level:
	@echo "Building $(level)..."
	@find . -name "$(level).cu" -exec $(NVCC) $(NVCC_FLAGS) {} -o {}.out \;

# Run a specific level
run:
	@echo "Running $(level)..."
	@./$(level).out

# Clean all executables
clean:
	@echo "Cleaning build artifacts..."
	@find . -type f -executable -not -name "*.sh" -delete
	@find . -name "*.out" -delete
	@echo "Clean complete!"

# Show help
help:
	@echo "CUDA Learning Curriculum - Build System"
	@echo "========================================"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all CUDA files"
	@echo "  module=X         - Build specific module (e.g., make module=01_thread_hierarchy)"
	@echo "  level=X          - Build specific level (e.g., make level=level1_basic_indexing)"
	@echo "  run              - Run a compiled level"
	@echo "  clean            - Remove all build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH        - GPU architecture (default: sm_70)"
	@echo "  NVCC_FLAGS       - Additional NVCC flags"
	@echo ""
	@echo "Examples:"
	@echo "  make all                              # Build everything"
	@echo "  make module=02_memory_hierarchy       # Build module 2"
	@echo "  make level=level1_global_memory       # Build specific level"
	@echo "  make CUDA_ARCH=sm_80 all              # Build for Ampere"
	@echo "  make clean                            # Clean build"
	@echo ""
	@echo "Available Modules:"
	@ls -1 */README.md 2>/dev/null | sed 's/\/README.md//' | while read mod; do echo "  - $$mod"; done

# Show module info
info:
	@echo "CUDA Learning Curriculum - Module Information"
	@echo "=============================================="
	@echo ""
	@for dir in */; do \
		if [ -f "$$dir/README.md" ]; then \
			echo "=== $$dir ==="; \
			head -10 "$$dir/README.md" | grep -v "^#"; \
			echo ""; \
		fi; \
	done

# Quick start - build and run first module
quickstart:
	@echo "Quick Start - Building and running first level..."
	@$(NVCC) $(NVCC_FLAGS) 01_thread_hierarchy/level1_basic_indexing.cu -o 01_thread_hierarchy/level1_basic_indexing
	@echo ""
	@echo "Running level1_basic_indexing..."
	@./01_thread_hierarchy/level1_basic_indexing

# Build with debug symbols
debug: NVCC_FLAGS := -arch=$(CUDA_ARCH) -g -G -O0
debug: all

# Build with verbose output
verbose: NVCC_FLAGS := -arch=$(CUDA_ARCH) -v
verbose: all

# Show register usage
registers: NVCC_FLAGS := -arch=$(CUDA_ARCH) -Xptxas=-v
registers: all

# Generate PTX
ptx:
	@echo "Generating PTX for all files..."
	@for f in $(CU_FILES); do \
		$(NVCC) $(NVCC_FLAGS) -ptx $$f -o $$f.ptx; \
	done

# Run all tests (basic smoke test)
test: all
	@echo "Running smoke tests..."
	@for exe in $(EXECUTABLES); do \
		echo "Testing $$exe..."; \
		timeout 10s ./$$exe > /dev/null 2>&1 && echo "  ✓ PASS" || echo "  ✗ FAIL (or timeout)"; \
	done
