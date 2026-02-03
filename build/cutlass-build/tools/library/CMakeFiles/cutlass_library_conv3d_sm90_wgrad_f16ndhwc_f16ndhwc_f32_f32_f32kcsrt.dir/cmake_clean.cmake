file(REMOVE_RECURSE
  "libcutlass_conv3d_sm90_wgrad_f16ndhwc_f16ndhwc_f32_f32_f32kcsrt.pdb"
  "libcutlass_conv3d_sm90_wgrad_f16ndhwc_f16ndhwc_f32_f32_f32kcsrt.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/cutlass_library_conv3d_sm90_wgrad_f16ndhwc_f16ndhwc_f32_f32_f32kcsrt.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
