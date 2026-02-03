# Setting up CUTLASS for CuTe Learning

## Prerequisites
To use this CuTe learning repository, you need to have CUTLASS 3.x installed with CuTe headers accessible.

## Installing CUTLASS

### Option 1: Clone from GitHub
```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout 3.x  # Use the latest 3.x branch
```

### Option 2: Download Release
Download the latest CUTLASS 3.x release from NVIDIA's GitHub repository.

## Building CUTLASS (Optional)
While CuTe is header-only, you may want to build CUTLASS to run tests:

```bash
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="89" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Using CUTLASS Headers
CuTe is part of CUTLASS 3.x and uses header-only components. You only need to make the headers accessible:

1. If you cloned CUTLASS to a local directory:
   ```bash
   # When compiling, use:
   nvcc -I/path/to/cutlass/include ...
   ```

2. If using CMake, update the CUTLASS_INCLUDE_DIR:
   ```bash
   cmake -DCUTLASS_INCLUDE_DIR=/path/to/cutlass/include ..
   ```

## Verifying Installation
To verify CuTe headers are accessible:
```bash
# Look for cute headers
ls /path/to/cutlass/include/cute/
# Should show files like: layout.hpp, tensor.hpp, etc.
```

## For Module 01
For the current module, ensure that the include path points to the CUTLASS directory containing the "cute" subdirectory with layout.hpp, tensor.hpp, etc.

## Troubleshooting
- If you see "fatal error: cute/layout.hpp: No such file or directory", verify your CUTLASS path
- Make sure you're using CUTLASS 3.x, not 2.x - CuTe is not available in older versions
- Ensure the cute/ subdirectory exists in your CUTLASS include directory