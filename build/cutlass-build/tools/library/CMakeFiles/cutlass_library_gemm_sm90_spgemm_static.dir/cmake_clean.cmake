file(REMOVE_RECURSE
  "libcutlass_gemm_sm90_spgemm.a"
  "libcutlass_gemm_sm90_spgemm.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm90_spgemm_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
