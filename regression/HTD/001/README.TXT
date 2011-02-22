
This program tests a function that is invoked both inside and outside a kernel
region.

Ideally, <compute> should not be cloned. Instead, it has __device__ and
__host__ flags.

