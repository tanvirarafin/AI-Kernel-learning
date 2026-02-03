file(REMOVE_RECURSE
  "libcutlass_gemm_sm90_gemm_grouped_f32xe5m2_f32xe5m2.a"
  "libcutlass_gemm_sm90_gemm_grouped_f32xe5m2_f32xe5m2.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_gemm_sm90_gemm_grouped_f32xe5m2_f32xe5m2_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
