#define cupmStream_t cudaStream_t
#include "ex13.inl"

/*TEST

 build:
   requires: cuda

 testset:
   output_file: ./output/ExitSuccess.out
   args: -device_enable {{lazy eager}}                                                         \
         -root_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}

   test:
     requires: cuda
     args: -default_device_type cuda
     suffix: cuda

TEST*/
