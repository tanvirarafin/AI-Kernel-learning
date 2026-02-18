#!/bin/bash
# CUDA Learning Curriculum - Quick Start Script
# Usage: ./quickstart.sh

set -e

echo "=============================================="
echo "  CUDA Kernel Programming - Quick Start"
echo "=============================================="
echo ""

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found!"
    echo "Please install CUDA Toolkit from:"
    echo "https://developer.nvidia.com/cuda-toolkit"
    exit 1
fi

echo "✓ CUDA found: $(nvcc --version | grep release | cut -d',' -f1)"
echo ""

# Detect GPU
echo "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ GPU: $GPU_NAME"
else
    echo "⚠ Could not detect GPU (nvidia-smi not available)"
fi
echo ""

# Get compute capability
echo "Recommended architecture flags:"
echo "  -arch=sm_70  (Volta)"
echo "  -arch=sm_75  (Turing)"
echo "  -arch=sm_80  (Ampere)"
echo "  -arch=sm_86  (Ampere)"
echo "  -arch=sm_89  (Ada)"
echo "  -arch=sm_90  (Hopper)"
echo ""

# Create build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR
echo "✓ Created build directory: $BUILD_DIR"
echo ""

# Show module structure
echo "=============================================="
echo "  Available Modules"
echo "=============================================="
echo ""

MODULES=(
    "01_thread_hierarchy:Thread Basics"
    "02_memory_hierarchy:Memory Types"
    "03_memory_coalescing:Coalescing"
    "04_shared_memory:Shared Memory"
    "05_reduction_patterns:Reduction"
    "06_matrix_multiplication:Matrix Multiply"
    "07_atomic_operations:Atomics"
    "08_warp_primitives:Warp Ops"
    "09_cuda_streams:Streams"
    "10_constant_memory:Constant Memory"
    "11_texture_memory:Texture Memory"
    "12_occupancy_optimization:Occupancy"
)

for i in "${!MODULES[@]}"; do
    IFS=':' read -r dir name <<< "${MODULES[$i]}"
    if [ -d "$dir" ]; then
        FILES=$(find "$dir" -name "*.cu" | wc -l)
        printf "%2d. %-30s (%d files)\n" $((i+1)) "$name" "$FILES"
    fi
done

echo ""
echo "=============================================="
echo "  Quick Start Commands"
echo "=============================================="
echo ""
echo "1. Build a specific module:"
echo "   make module=01_thread_hierarchy"
echo ""
echo "2. Build a specific level:"
echo "   make level=level1_basic_indexing"
echo ""
echo "3. Build everything:"
echo "   make all"
echo ""
echo "4. Build and run first exercise:"
echo "   make quickstart"
echo ""
echo "5. Clean build artifacts:"
echo "   make clean"
echo ""
echo "=============================================="
echo "  Recommended Learning Path"
echo "=============================================="
echo ""
echo "Week 1-2: Foundations"
echo "  → 01_thread_hierarchy"
echo "  → 02_memory_hierarchy"
echo "  → 03_memory_coalescing"
echo ""
echo "Week 3-4: Core Patterns"
echo "  → 04_shared_memory"
echo "  → 05_reduction_patterns"
echo "  → 06_matrix_multiplication"
echo ""
echo "Week 5-6: Advanced Topics"
echo "  → 07_atomic_operations"
echo "  → 08_warp_primitives"
echo "  → 09_cuda_streams"
echo ""
echo "Week 7-8: Specialization"
echo "  → 10_constant_memory"
echo "  → 11_texture_memory"
echo "  → 12_occupancy_optimization"
echo ""
echo "=============================================="
echo ""
echo "📚 Read README.md for detailed guide"
echo "📋 Use PROGRESS_TRACKER.md to track progress"
echo "🔧 Use CHEATSHEET.md for quick reference"
echo "🐛 Use DEBUG_GUIDE.md for debugging help"
echo ""
echo "Happy CUDA Learning! 🚀"
echo ""
