# CMakeLists.txt for Module 5: Mainloop Pipelining - Temporal Overlap & Throughput
cmake_minimum_required(VERSION 3.20)
project(module5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture to target sm_89 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include CUTLASS from third_party submodule
include_directories(${PROJECT_SOURCE_DIR}/../third_party/cutlass/include)

# Add executable for Module 5
add_executable(module5 main.cu)

# Set CUDA properties
set_property(TARGET module5 PROPERTY CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET module5 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Additional CUDA flags for CUTLASS compatibility
target_compile_options(module5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Link libraries if needed
target_link_libraries(module5 ${CUDA_CUBLAS_LIBRARIES})