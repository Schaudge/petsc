#define cupmStream_t cudaStream_t

#include "ex15_device.inl"
#include "ex15.inl"

/*TEST

 build:
   requires: cuda

 testset:
   output_file: ./output/ExitSuccess.out
   args: -root_device_context_stream_type {{default_blocking global_nonblocking}}

   test:
     requires: cuda
     args: -default_device_type cuda -cycles 1000000
     suffix: cuda

TEST*/
