# Module 3 CMakeLists.txt - Tiled Copy with Spatial Async Data Movement

cmake_minimum_required(VERSION 3.20)

project(Module3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Define the executable
add_executable(module3_main main.cu)

# Set CUDA properties
set_property(TARGET module3_main PROPERTY CUDA_ARCHITECTURES 89)

# Include CUTLASS headers
target_include_directories(module3_main PRIVATE
    ${PROJECT_SOURCE_DIR}/../../third_party/cutlass/include
    ${PROJECT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include
)