file(REMOVE_RECURSE
  "libcutlass_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc.pdb"
  "libcutlass_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv3d_sm90_fprop_s8ndhwc_s8ndhwc_s32_s32_s32ndhwc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
