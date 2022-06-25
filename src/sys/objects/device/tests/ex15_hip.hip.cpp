#define cupmStream_t hipStream_t

#include "ex15_device.inl"
#include "ex15.inl"

/*TEST

 build:
   requires: hip

 testset:
   output_file: ./output/ExitSuccess.out
   args: -root_device_context_stream_type {{default_blocking global_nonblocking}}

   test:
     requires: hip
     args: -default_device_type hip -cycles 1000000
     suffix: hip

TEST*/
