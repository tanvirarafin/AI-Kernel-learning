cmake_minimum_required(VERSION 3.20)
project(module2_tiled_copy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture to target sm_89 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include CUTLASS from third_party submodule
include_directories(${PROJECT_SOURCE_DIR}/../third_party/cutlass/include)

# Add executable for Module 2
add_executable(module2 main.cu)

# Set CUDA properties
set_property(TARGET module2 PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET module2 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Additional CUDA flags for CUTLASS compatibility
target_compile_options(module2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)