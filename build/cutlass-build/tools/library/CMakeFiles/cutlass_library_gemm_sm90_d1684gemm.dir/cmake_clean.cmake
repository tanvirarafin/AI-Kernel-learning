file(REMOVE_RECURSE
  "libcutlass_gemm_sm90_d1684gemm.pdb"
  "libcutlass_gemm_sm90_d1684gemm.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm90_d1684gemm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
