
This directory provides examples of using the hiCUDA language and compiler.
In each example,

- "make" compiles the hiCUDA version of the application into a CUDA program.

- "make <app>_hicuda" compiles the generated CUDA program (NVCC required).

- "make <app>_hicuda_seq" compiles the hiCUDA version using a regular
  compiler, ignoring all directives.

- "make <app>_cuda" compiles the hand-written CUDA program (NVCC required).

- "make all" does everything above.

- "make clean" removes all files generated during compilation.

