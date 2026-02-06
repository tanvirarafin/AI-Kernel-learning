# Makefile for CUTLASS Learning Repository

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -arch=sm_89 -O3 -use_fast_math -lineinfo -std=c++17 -DNDEBUG
INCLUDES = -I./third_party/cutlass/include -I./third_party/cutlass/examples/common

# Directories
BUILD_DIR = build
MODULE1_DIR = module1

# Targets
.PHONY: all clean module1 setup-submodule

all: module1

# Setup CUTLASS as submodule
setup-submodule:
	@if [ ! -d "third_party/cutlass" ]; then \
		echo "Initializing CUTLASS submodule..."; \
		git submodule update --init --recursive; \
	else \
		echo "CUTLASS submodule already exists."; \
	fi

# Build Module 1
module1: setup-submodule
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(MODULE1_DIR)/main.cu -o $(BUILD_DIR)/module1
	@echo "Module 1 built successfully!"

# Run Module 1
run-module1: module1
	@echo "Running Module 1: Layout Algebra Foundation"
	./$(BUILD_DIR)/module1

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Utility to check CUDA installation
check-cuda:
	nvcc --version
	@echo "Checking for CUTLASS submodule..."
	@if [ ! -d "third_party/cutlass" ]; then \
		echo "WARNING: CUTLASS submodule not found. Run 'make setup-submodule'"; \
	else \
		echo "CUTLASS submodule found."; \
	fi

# Default compilation command for individual modules if needed
compile-module1-direct:
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) module1/main.cu -o module1_exec
	@echo "Module 1 compiled directly!"