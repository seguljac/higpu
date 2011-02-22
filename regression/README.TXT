
This directory holds regression tests of the hiCUDA compiler, divided into the
following categories:

[HTA] Allocation and data transfer for global and constant directives of
      statically and dynamically allocated variables (i.e. shape directive);
      optimized constant memory allocation.

[HTB] Translation of the loop partition directives.

[HTC] Allocation and data transfer for shared directives of statically
      and dynamically allocated variables; optimized shared memory allocation.
      
[HTD] Inter-procedural reaching directives analysis,
      kernel data access analysis, kernel access redirection, and
      kernel outlining.
      
[HTE] Kernel context propagation, kernel loop context propagation

[HTF] hiCUDA front-end and CUDA code generator

Simply type "./run-all.sh" to run all these tests.
