file(REMOVE_RECURSE
  "libcutlass_gemm_sm90_gemm.a"
  "libcutlass_gemm_sm90_gemm.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm90_gemm_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
