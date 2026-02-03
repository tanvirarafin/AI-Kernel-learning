file(REMOVE_RECURSE
  "libcutlass_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc.pdb"
  "libcutlass_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv2d_sm90_fprop_s8nhwc_s8nhwc_s32_s32_s32nhwc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
