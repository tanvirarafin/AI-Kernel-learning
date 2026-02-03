cmake_minimum_required(VERSION 3.18)
project(ModularKernels LANGUAGES CXX CUDA)

# 1. Target your RTX 4060
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)

# 2. Define the path to the submodule
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")

# 3. Add CUTLASS include directories (Header-only usage)
add_library(cutlass_interface INTERFACE)
target_include_directories(cutlass_interface INTERFACE 
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/common
)

# 4. Your executable
add_executable(my_kernel main.cu)
target_link_libraries(my_kernel PRIVATE cutlass_interface)

# 5. Required flags for CUTLASS
target_compile_options(my_kernel PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
