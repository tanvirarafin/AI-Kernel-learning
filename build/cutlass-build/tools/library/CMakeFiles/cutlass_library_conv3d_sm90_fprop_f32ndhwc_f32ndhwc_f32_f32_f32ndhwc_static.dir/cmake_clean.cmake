file(REMOVE_RECURSE
  "libcutlass_conv3d_sm90_fprop_f32ndhwc_f32ndhwc_f32_f32_f32ndhwc.a"
  "libcutlass_conv3d_sm90_fprop_f32ndhwc_f32ndhwc_f32_f32_f32ndhwc.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv3d_sm90_fprop_f32ndhwc_f32ndhwc_f32_f32_f32ndhwc_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
