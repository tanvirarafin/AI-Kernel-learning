#!/bin/bash

# Build script for Module 4: MMA Atoms - Spatial Tensor Core Mastery

echo "Building Module 4: MMA Atoms - Spatial Tensor Core Mastery"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
cmake .. -DCUTLASS_DIR=../../third_party/cutlass

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo "Building the project..."
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"

# Optionally run the executable
if [ "$1" == "run" ]; then
    echo "Running the executable..."
    ./mma_atom_spatial
fi