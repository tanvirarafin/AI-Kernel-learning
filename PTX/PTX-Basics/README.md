# PTX Basics Module

Welcome to the PTX (Parallel Thread Execution) Basics module! This module will introduce you to the fundamentals of PTX assembly language, which is NVIDIA's low-level parallel computing architecture.

## Learning Objectives
- Understand what PTX is and why it matters for GPU programming
- Learn the basic syntax and structure of PTX code
- Familiarize yourself with PTX data types and registers
- Write your first simple PTX programs
- Understand the relationship between CUDA C/C++ and PTX

## Table of Contents
1. [Introduction to PTX](#introduction-to-ptx)
2. [Basic Syntax and Structure](#basic-syntax-and-structure)
3. [Data Types and Registers](#data-types-and-registers)
4. [Simple PTX Programs](#simple-ptx-programs)
5. [CUDA to PTX Translation](#cuda-to-ptx-translation)
6. [Exercise: Hello World in PTX](#exercise-hello-world-in-ptx)

## Introduction to PTX

PTX (Parallel Thread Execution) is a virtual instruction set architecture (ISA) for NVIDIA GPUs. It serves as an intermediate representation between high-level languages like CUDA C/C++ and the actual GPU machine code. PTX provides a stable interface that allows CUDA programs to run on different GPU architectures.

### Why PTX Matters
- **Portability**: PTX code can run on different GPU generations
- **Optimization**: Understanding PTX allows for fine-grained optimization
- **Debugging**: PTX gives insight into what the compiler actually generates
- **Performance**: Knowledge of PTX enables writing more efficient CUDA code

## Basic Syntax and Structure

A typical PTX program consists of:
- `.version` directive: Specifies the PTX version
- `.target` directive: Defines the target GPU architecture
- `.address_size` directive: Sets address size (32 or 64 bit)
- Function declarations and definitions
- Variable declarations
- Instructions

Example skeleton:
```
.version 6.0
.target sm_50
.address_size 64

.entry main {
    // Function body
}
```

## Data Types and Registers

PTX supports various data types:
- Integer types: `.u8`, `.u16`, `.u32`, `.u64` (unsigned), `.s8`, `.s16`, `.s32`, `.s64` (signed)
- Floating-point types: `.f16`, `.f32`, `.f64`
- Predicate type: `.pred`

Registers are named with `%` prefix:
- `%r0`, `%r1`, ... for 32-bit registers
- `%rd0`, `%rd1`, ... for 64-bit registers
- `%f0`, `%f1`, ... for 32-bit floating-point registers
- `%d0`, `%d1`, ... for 64-bit floating-point registers
- `%p0`, `%p1`, ... for predicate registers

## Simple PTX Programs

Let's look at a simple addition example:
```
.version 6.0
.target sm_50
.address_size 64

.visible .entry add_simple(.param .u32 a, .param .u32 b, .param .u32* result) {
    .reg .u32 %a<2>;
    .reg .u32 %b<2>;
    .reg .u32 %result<2>;
    .reg .u32 %sum;
    
    ld.param.u32 %a1, [a];
    ld.param.u32 %b1, [b];
    add.u32 %sum, %a1, %b1;
    ld.param.u64 %result1, [result];
    st.u32 [%result1], %sum;
    ret;
}
```

## CUDA to PTX Translation

To generate PTX from CUDA code, use:
```bash
nvcc -ptx kernel.cu -o kernel.ptx
```

Or to see the PTX inline during compilation:
```bash
nvcc -Xptxas -v kernel.cu
```

## Exercise: Hello World in PTX

Try compiling and understanding the following simple PTX program that adds two numbers:

1. Create a file called `add.ptx` with the example above
2. Compile it to SASS (actual GPU assembly) with:
   ```bash
   nvcc -cubin add.ptx -o add.cubin
   ```
3. Disassemble it to see the actual GPU instructions:
   ```bash
   cuobjdump -sass add.cubin
   ```

## Next Steps

After completing this module, you should have a solid understanding of:
- PTX syntax and structure
- Basic PTX instructions
- How to generate PTX from CUDA code
- How to examine the resulting assembly

Proceed to the next module to learn about memory management in PTX.