file(REMOVE_RECURSE
  "libcutlass_conv2d_sm90_fprop_bf16nhwc_bf16nhwc_f32_f32_f32nhwc.a"
  "libcutlass_conv2d_sm90_fprop_bf16nhwc_bf16nhwc_f32_f32_f32nhwc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv2d_sm90_fprop_bf16nhwc_bf16nhwc_f32_f32_f32nhwc_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
