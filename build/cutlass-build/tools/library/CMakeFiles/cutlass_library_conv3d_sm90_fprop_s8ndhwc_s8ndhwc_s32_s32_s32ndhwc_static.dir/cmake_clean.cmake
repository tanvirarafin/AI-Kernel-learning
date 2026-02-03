file(REMOVE_RECURSE
  "libcutlass_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc.a"
  "libcutlass_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
