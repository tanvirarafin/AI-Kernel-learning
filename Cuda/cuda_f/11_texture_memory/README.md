# Texture Memory in CUDA

Master texture memory for spatial data access with interpolation.

## Concepts Covered
- Texture reference (legacy)
- Texture object (modern)
- Normalized coordinates
- Linear interpolation
- Address modes

## Levels

### Level 1: Texture Basics (`level1_texture_basics.cu`)
- **Goal**: Learn texture object creation and access
- **Missing**: cudaTextureObject_t creation, tex2D access
- **Concepts**: Texture memory, cached reads

### Level 2: 2D Texture Access (`level2_2d_texture.cu`)
- **Goal**: Access 2D data with texture memory
- **Missing**: cudaArray, cudaMemcpy2DToArray
- **Concepts**: 2D spatial locality, texture fetch

### Level 3: Linear Interpolation (`level3_interpolation.cu`)
- **Goal**: Use hardware interpolation
- **Missing**: Filter mode configuration
- **Concepts**: Bilinear interpolation, fractional coordinates

### Level 4: Address Modes (`level4_address_modes.cu`)
- **Goal**: Configure boundary handling
- **Missing**: Address mode configuration
- **Concepts**: Clamp, wrap, mirror modes

## Key Principles
1. **Texture Object**: Modern API (cudaTextureObject_t)
2. **cudaArray**: Optimized layout for texture access
3. **Interpolation**: Hardware bilinear filtering
4. **Address Modes**: Handle boundary conditions
