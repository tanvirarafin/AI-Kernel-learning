#include <iostream>
#include <vector>
#include <iomanip>

// Include CuTe headers
#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/stride.hpp"

using namespace cute;

/*
 * Module 1: Layout Algebra Foundation
 * Composable Data Access
 *
 * This kernel demonstrates the fundamental concept of CuTe layouts by showing
 * how a 2D tile maps to memory using mathematical layout algebra instead of
 * traditional nested loops. We'll compare row-major vs column-major layouts
 * by computing the logical-to-physical mapping mathematically.
 */

template<class Layout>
void print_layout_mapping(const Layout& layout, const char* layout_name) {
    // Print the layout shape and strides
    std::cout << "\n" << layout_name << " Layout:" << std::endl;
    std::cout << "Shape: " << layout.shape() << std::endl;
    std::cout << "Stride: " << layout.stride() << std::endl;

    // Calculate and print the mapping from logical to physical indices
    std::cout << "Logical -> Physical Mapping:" << std::endl;
    std::cout << "Format: (row, col) -> physical_address" << std::endl;

    // Iterate through all logical coordinates in the layout
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            // Create a coordinate tuple (i, j)
            auto coord = make_tuple(i, j);

            // Map logical coordinate to physical address using layout
            int physical_addr = layout(coord);

            std::cout << "(" << i << "," << j << ") -> " << physical_addr << std::endl;
        }
    }
}

int main() {
    std::cout << "=== CUTLASS 3.x CuTe Layout Algebra Demo ===" << std::endl;
    std::cout << "Demonstrating 2D layout mapping with mathematical formulas" << std::endl;

    // Define a 3x4 matrix layout
    constexpr int rows = 3;
    constexpr int cols = 4;

    /*
     * Row-Major Layout: Elements stored row by row
     * For a 3x4 matrix, the layout is defined as:
     * - Shape: (3, 4) - 3 rows, 4 columns
     * - Stride: (4, 1) - stride 4 to move to next row, stride 1 to move to next column
     * Address formula: addr = row_idx * 4 + col_idx * 1
     */
    auto row_major_layout = make_layout(make_shape(Int<rows>{}, Int<cols>{}),
                                        make_stride(Int<cols>{}, Int<1>{}));

    print_layout_mapping(row_major_layout, "Row-Major");

    /*
     * Column-Major Layout: Elements stored column by column
     * For a 3x4 matrix, the layout is defined as:
     * - Shape: (3, 4) - 3 rows, 4 columns
     * - Stride: (1, 3) - stride 1 to move to next row, stride 3 to move to next column
     * Address formula: addr = row_idx * 1 + col_idx * 3
     */
    auto col_major_layout = make_layout(make_shape(Int<rows>{}, Int<cols>{}),
                                        make_stride(Int<1>{}, Int<rows>{}));

    print_layout_mapping(col_major_layout, "Column-Major");

    // Demonstrate nested layouts - a more advanced concept
    std::cout << "\n=== Nested Layout Example ===" << std::endl;
    std::cout << "Creating a 2x2 tile repeated 2 times in each dimension (4x4 total)" << std::endl;

    // Create a 2x2 tile layout
    auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}),
                                   make_stride(Int<2>{}, Int<1>{}));  // Row-major within tile

    // Create a layout that represents 2x2 tiles arranged in a 2x2 grid
    auto tiled_2x2_grid = tile_to_shape(tile_layout, make_shape(Int<4>{}, Int<4>{}));

    std::cout << "Tiled Layout (2x2 tiles in 2x2 grid):" << std::endl;
    std::cout << "Shape: " << tiled_2x2_grid.shape() << std::endl;
    std::cout << "Total size: " << size(tiled_2x2_grid) << std::endl;

    // Print the mapping for the tiled layout
    std::cout << "Logical -> Physical Mapping for Tiled Layout:" << std::endl;
    for (int i = 0; i < size<0>(tiled_2x2_grid); ++i) {
        for (int j = 0; j < size<1>(tiled_2x2_grid); ++j) {
            auto coord = make_tuple(i, j);
            int physical_addr = tiled_2x2_grid(coord);
            std::cout << "(" << std::setw(1) << i << "," << std::setw(1) << j << ") -> "
                      << std::setw(2) << physical_addr;

            // Add a newline every 4 elements to visualize the 2x2 tile structure
            if ((j + 1) % 4 == 0) std::cout << std::endl;
        }
    }

    // Show how layouts can be composed mathematically
    std::cout << "\n=== Layout Composition ===" << std::endl;
    std::cout << "CuTe allows mathematical operations on layouts:" << std::endl;

    // Create a simple 4-element vector layout
    auto vector_layout = make_layout(Int<4>{});
    std::cout << "1D Vector Layout - Shape: " << vector_layout.shape()
              << ", Stride: " << vector_layout.stride() << std::endl;

    // Reshape it to a 2x2 matrix using composition
    auto product_layout = make_layout(make_shape(Int<2>{}, Int<2>{}),
                                      make_stride(Int<2>{}, Int<1>{}));
    std::cout << "Reshaped 2x2 Layout - Shape: " << product_layout.shape()
              << ", Stride: " << product_layout.stride() << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "CuTe layouts provide a mathematical way to define memory mappings" << std::endl;
    std::cout << "- No manual indexing required" << std::endl;
    std::cout << "- Composable through mathematical operations" << std::endl;
    std::cout << "- Enable thread-level tensor partitioning" << std::endl;
    std::cout << "- Eliminate bounds checking overhead" << std::endl;

    return 0;
}