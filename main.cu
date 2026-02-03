#include <iostream>
#include <vector>

// Include CUTLASS and CuTe headers
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// Include CuTe (Composable Layouts and Operations)
#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/tensor.hpp"
#include "cute/container/tuple.hpp"

using namespace cute;

/**
 * @brief Module 1: Layouts and Tensors
 * 
 * FIRST PRINCIPLES EXPLANATION:
 * 
 * In CuTe, everything revolves around the concept of LAYOUTS - mathematical mappings
 * from logical coordinates to physical memory addresses. A layout defines how 
 * multi-dimensional indices map to linear memory offsets.
 * 
 * Key concepts:
 * - Shape: The dimensions of the tensor (e.g., 4x4 matrix has shape (4,4))
 * - Stride: The step size between elements in each dimension
 * - Layout = Shape + Stride (defines the complete mapping)
 * 
 * CuTe's power comes from its ALGEBRAIC approach - layouts can be composed,
 * nested, and manipulated using operators like product (composition) and sum (union).
 * 
 * Instead of manual indexing, we think in terms of "partitioning tensors for threads"
 * - Each thread gets a subset of the tensor based on the layout decomposition
 */

void hello_world_cute() {
    std::cout << "=== Hello World of CuTe: Layout Algebra ===" << std::endl;
    
    /**
     * Create a simple 2D layout: 4x4 matrix with row-major ordering
     * In CuTe, we define this using Shape and Stride
     */
    auto simple_layout = make_layout(make_shape(_4{}, _4{}));  // 4x4 matrix, row-major by default
    
    std::cout << "Simple 4x4 layout: " << simple_layout << std::endl;
    
    // Access individual elements
    auto coord_2_3 = make_coord(_2{}, _3{});  // Row 2, Column 3
    int physical_addr = simple_layout(coord_2_3);
    std::cout << "Logical coordinate (2,3) maps to physical address: " << physical_addr << std::endl;
    
    // Create a more complex layout with custom strides (column-major)
    auto col_major_layout = make_layout(
        make_shape(_4{}, _4{}),
        make_stride(_1{}, _4{})  // Stride of 1 for rows, 4 for columns (column-major)
    );
    
    std::cout << "Column-major 4x4 layout: " << col_major_layout << std::endl;
    
    // Show the difference in mapping
    std::cout << "\nRow-major mapping:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            auto coord = make_coord(i, j);
            std::cout << simple_layout(coord) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nColumn-major mapping:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            auto coord = make_coord(i, j);
            std::cout << col_major_layout(coord) << " ";
        }
        std::cout << std::endl;
    }
}

void basic_nested_layouts() {
    std::cout << "\n\n=== Basic Nested Layouts Example ===" << std::endl;
    
    /**
     * Simple nested layout example - creating a 2D layout from 1D components
     */
    
    // Create a 2D layout representing a 2x3 grid
    auto basic_2d_layout = make_layout(
        make_shape(_2{}, _3{}),  // 2 rows, 3 columns
        make_stride(_3{}, _1{})  // stride of 3 for rows, 1 for columns (row-major)
    );
    
    std::cout << "Basic 2x3 layout: " << basic_2d_layout << std::endl;
    
    // Show how coordinates map to physical addresses
    std::cout << "\n2x3 layout mapping:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            auto coord = make_coord(i, j);
            std::cout << basic_2d_layout(coord) << " ";
        }
        std::cout << std::endl;
    }
}

void cute_basic_operations() {
    std::cout << "\n\n=== CuTe Basic Operations ===" << std::endl;
    
    /**
     * Basic operations on layouts
     */
    
    // Simple 1D layout
    auto layout_1d = make_layout(make_shape(_6{}));  // 6 elements in a row
    std::cout << "1D layout (6 elements): " << layout_1d << std::endl;
    
    // 2D layout: 3x2
    auto layout_2d = make_layout(make_shape(_3{}, _2{}));
    std::cout << "2D layout (3x2): " << layout_2d << std::endl;
    
    // Show how each coordinate maps to a physical address
    std::cout << "Mapping for 3x2 layout:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto coord = make_coord(i, j);
            std::cout << "(" << i << "," << j << ")->" << layout_2d(coord) << "  ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "CUTLASS 3.x & CuTe Learning - Module 1: Layouts and Tensors" << std::endl;
    std::cout << "Target Architecture: sm_89 (NVIDIA RTX 4060)" << std::endl;
    
    // Run the "Hello World" of CuTe
    hello_world_cute();
    
    // Demonstrate basic nested layouts
    basic_nested_layouts();
    
    // Show basic CuTe operations
    cute_basic_operations();
    
    std::cout << "\n\nModule 1 completed successfully!" << std::endl;
    std::cout << "Key concepts covered:" << std::endl;
    std::cout << "- Layout algebra (Shape and Stride)" << std::endl;
    std::cout << "- Logical-to-physical mapping" << std::endl;
    std::cout << "- Basic layout composition" << std::endl;
    std::cout << "- Composable abstractions instead of manual indexing" << std::endl;
    
    return 0;
}