file(REMOVE_RECURSE
  "libcutlass_gemm_sm90_spgemm.pdb"
  "libcutlass_gemm_sm90_spgemm.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm90_spgemm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
