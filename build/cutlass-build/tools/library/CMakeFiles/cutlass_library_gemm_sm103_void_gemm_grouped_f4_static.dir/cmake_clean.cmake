file(REMOVE_RECURSE
  "libcutlass_gemm_sm103_void_gemm_grouped_f4.a"
  "libcutlass_gemm_sm103_void_gemm_grouped_f4.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm103_void_gemm_grouped_f4_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
