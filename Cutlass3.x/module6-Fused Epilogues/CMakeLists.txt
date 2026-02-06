# CMakeLists.txt for Module 6: Fused Epilogues - Functional Avoiding VRAM Roundtrips
cmake_minimum_required(VERSION 3.20)
project(module6 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture to target sm_89 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include CUTLASS from third_party submodule
include_directories(${PROJECT_SOURCE_DIR}/../third_party/cutlass/include)

# Add executable for Module 6
add_executable(module6 main.cu)

# Set CUDA properties
set_property(TARGET module6 PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET module6 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Additional CUDA flags for CUTLASS compatibility
target_compile_options(module6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Link libraries if needed
target_link_libraries(module6 ${CUDA_CUBLAS_LIBRARIES})