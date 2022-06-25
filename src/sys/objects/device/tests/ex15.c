#include <petscdevice.h>
#include <time.h>

static PetscErrorCode WasteSomeTime(PetscDeviceContext dctx, PetscInt ncycles, clock_t *global_now) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#include "ex15.inl"

/*TEST

  testset:
    requires: cxx
    suffix: cxx
    output_file: ./output/ExitSuccess.out
    args: -device_enable {{lazy eager}} -cycles 0 \
         -root_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}

    test:
      requires: !device
      suffix: host_no_device
    test:
      requires: device
      args: -default_device_type host
      suffix: host_with_device
    test:
      requires: sycl
      args: -default_device_type sycl
      suffix: sycl

  test:
    requires: !cxx
    suffix: no_cxx
    output_file: ./output/ExitSuccess.out
TEST*/
