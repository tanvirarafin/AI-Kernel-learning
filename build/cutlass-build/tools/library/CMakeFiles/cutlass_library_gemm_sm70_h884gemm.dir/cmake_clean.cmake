file(REMOVE_RECURSE
  "libcutlass_gemm_sm70_h884gemm.pdb"
  "libcutlass_gemm_sm70_h884gemm.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm70_h884gemm.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
