// main.cu
#include <iostream>
#include <vector>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>

using namespace cute;

void run_gemm() {
    // 1. Define Problem Dimensions
    int M = 128;
    int N = 128;
    int K = 64;

    // 2. Define the GEMM Configuration (Modular-friendly CuTe style)
    // We'll use FP32 for simplicity on your 4060
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using LayoutA  = cutlass::layout::RowMajor;
    using LayoutB  = cutlass::layout::ColumnMajor;
    using LayoutC  = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC
    >;

    Gemm gemm_op;

    // 3. Allocate and Initialize Tensors
    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C({M, N});

    // Simple fill
    for (int i = 0; i < A.size(); ++i) A.host_data()[i] = 1.0f;
    for (int i = 0; i < B.size(); ++i) B.host_data()[i] = 2.0f;
    
    A.sync_device();
    B.sync_device();
    C.sync_device();

    // 4. Set up Arguments
    typename Gemm::Arguments args(
        {M, N, K}, 
        A.device_ref(), 
        B.device_ref(), 
        C.device_ref(), 
        C.device_ref(), 
        {1.0f, 0.0f} // alpha, beta
    );

    // 5. Run it
    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed!" << std::endl;
    } else {
        C.sync_host();
        std::cout << "Successfully ran GEMM! First element of C: " << C.host_data()[0] << std::endl;
    }
}

int main() {
    run_gemm();
    return 0;
}
