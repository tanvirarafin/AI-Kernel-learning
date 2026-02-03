file(REMOVE_RECURSE
  "libcutlass_conv2d_sm90_wgrad_f16nhwc_f16nhwc_f32_f32_f32kcsr.pdb"
  "libcutlass_conv2d_sm90_wgrad_f16nhwc_f16nhwc_f32_f32_f32kcsr.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv2d_sm90_wgrad_f16nhwc_f16nhwc_f32_f32_f32kcsr.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
