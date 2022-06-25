#define cupmStream_t hipStream_t
#include "ex13.inl"

/*TEST

 build:
   requires: hip

 testset:
   output_file: ./output/ExitSuccess.out
   args: -device_enable {{lazy eager}}                                                         \
         -root_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}

   test:
     requires: hip
     args: -default_device_type hip
     suffix: hip

TEST*/
