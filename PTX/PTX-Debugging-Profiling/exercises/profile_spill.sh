#!/bin/bash

echo "Profiling register usage for spilling vs optimized kernels"

echo "Compiling PTX files..."
nvcc -ptx spilling_kernel.ptx -o spilling_kernel.ptx  # Just validate syntax
nvcc -ptx optimized_kernel.ptx -o optimized_kernel.ptx  # Just validate syntax

echo "Analyzing register usage with cuobjdump..."
echo "Spilling kernel register info:"
cuobjdump -ptx spilling_kernel.ptx | grep -A 10 -B 10 ".visible .entry spilling_kernel"

echo ""
echo "Optimized kernel register info:"
cuobjdump -ptx optimized_kernel.ptx | grep -A 10 -B 10 ".visible .entry optimized_kernel"

echo ""
echo "Run with nvprof to compare performance:"
echo "nvprof --print-gpu-trace ./test_spill"