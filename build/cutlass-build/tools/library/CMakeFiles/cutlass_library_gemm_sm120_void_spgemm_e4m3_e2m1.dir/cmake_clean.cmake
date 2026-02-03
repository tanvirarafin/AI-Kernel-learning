file(REMOVE_RECURSE
  "libcutlass_gemm_sm120_void_spgemm_e4m3_e2m1.pdb"
  "libcutlass_gemm_sm120_void_spgemm_e4m3_e2m1.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm120_void_spgemm_e4m3_e2m1.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
