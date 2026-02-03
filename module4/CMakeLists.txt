cmake_minimum_required(VERSION 3.20)
project(Module4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUTLASS as a submodule
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass)

# Set CUDA architectures (sm_89 for Ada Lovelace RTX 4060)
set(CMAKE_CUDA_ARCHITECTURES "89")

# Enable CUDA compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -DNDEBUG")

# Add executable
add_executable(mma_atom_spatial main.cu)

# Link against CUDA runtime
target_link_libraries(mma_atom_spatial ${CUDA_LIBRARIES})

# Include directories
target_include_directories(mma_atom_spatial PRIVATE ${CUTLASS_DIR}/include)
target_include_directories(mma_atom_spatial PRIVATE ${CUTLASS_DIR}/examples/common)