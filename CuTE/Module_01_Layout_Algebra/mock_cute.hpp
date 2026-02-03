// Mock header to simulate CuTe for compilation test
#ifndef MOCK_CUTE_H
#define MOCK_CUTE_H

#include <iostream>
#include <type_traits>

namespace cute {
    // Simplified mock for layout functionality
    template<class T>
    struct Int {
        static constexpr int value = T::value;
    };
    
    template<int V>
    struct constant {
        static constexpr int value = V;
    };
    
    template<int V>
    using IntConstant = Int<constant<V>>;
    
    template<int M, int N>
    struct SimpleLayout {
        __host__ __device__
        constexpr int operator()(int i, int j) const {
            return i * N + j;  // Simple row-major mapping
        }
        
        struct Shape {
            int m, n;
            Shape(int _m, int _n) : m(_m), n(_n) {}
        };
        
        Shape shape() const { return Shape(M, N); }
    };
    
    template<int M, int N>
    auto make_layout(const std::pair<Int<M>, Int<N>>&, auto) -> SimpleLayout<M, N> {
        return SimpleLayout<M, N>{};
    }
    
    template<int M>
    constexpr auto Int_v = Int<constant<M>>{};
    
    template<class Layout>
    void print(const Layout& layout) {
        std::cout << "Layout with shape (" << layout.shape().m << "x" << layout.shape().n << ")" << std::endl;
        for(int i = 0; i < layout.shape().m; i++) {
            for(int j = 0; j < layout.shape().n; j++) {
                std::cout << layout(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    struct GenRowMajor {};
    struct GenColMajor {};
}

// Define Int constants for compatibility
template<int V>
struct IntVal {
    static constexpr int value = V;
};

#define Int<V> cute::Int<constant<V>>

#endif