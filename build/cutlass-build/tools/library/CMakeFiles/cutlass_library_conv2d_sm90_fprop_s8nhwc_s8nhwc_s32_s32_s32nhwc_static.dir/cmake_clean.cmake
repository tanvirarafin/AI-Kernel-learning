file(REMOVE_RECURSE
  "libcutlass_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc.a"
  "libcutlass_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
