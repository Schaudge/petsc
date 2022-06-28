static const char help[] = "Tests PetscManagedScalar memory pool coherence.\n\n";

#include <petscdevice.h>

int main(int argc, char *argv[]) {
  MPI_Comm           comm;
  PetscDeviceType    dtype;
  PetscDeviceContext dctxa, dctxb;
  PetscManagedScalar scal;
  PetscScalar       *ptra, *ptrb, *ptrc;
  PetscInt           n = 1000;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, NULL, "Test Options", NULL);
  PetscCall(PetscOptionsRangeInt("-n", "Size of managed scalars. Should be large enough to hide host to device memcopies", NULL, n, &n, NULL, 1, PETSC_MAX_INT));
  PetscOptionsEnd();

  // create the device contexts, each must be on a non-blocking stream-type to illustrate this
  // test
  PetscCall(PetscDeviceContextCreate(&dctxa));
  PetscCall(PetscDeviceContextSetStreamType(dctxa, PETSC_STREAM_DEFAULT_BLOCKING));
  // we want to leave one of the managed scalars "dangling" below, and hence won't be
  // synchronizing on the context before it is destroyed
  PetscCall(PetscDeviceContextSetOption(dctxa, PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS, PETSC_TRUE));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctxa_", dctxa));
  PetscCall(PetscDeviceContextGetDeviceType(dctxa, &dtype));

  PetscCall(PetscDeviceContextCreate(&dctxb));
  PetscCall(PetscDeviceContextSetStreamType(dctxb, PETSC_STREAM_DEFAULT_BLOCKING));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctxb_", dctxb));

  // allocate and fill the value buffers
  PetscCall(PetscDeviceMalloc(dctxa, PETSC_MEMTYPE_HOST, n, &ptra));
  PetscCall(PetscDeviceMalloc(dctxa, PETSC_MEMTYPE_HOST, n, &ptrb));
  PetscCall(PetscDeviceContextSynchronize(dctxa));

  for (PetscInt i = 0; i < n; ++i) {
    ptra[i] = (PetscScalar)1.0;
    ptrb[i] = (PetscScalar)2.0;
  }

  PetscCall(PetscManagedScalarCreateDefault(dctxa, n, &scal));
  // make sure we instantiate both the host and device pointers
  PetscCall(PetscManagedScalarEnsureOffload(dctxa, scal, dtype == PETSC_DEVICE_HOST ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_BOTH, PETSC_FALSE));
  // now we set the values to all 1's
  PetscCall(PetscManagedScalarSetValues(dctxa, scal, PETSC_MEMTYPE_HOST, ptra, n));
  // ensure a memcopy is taking place (if we have a device to memcpy to)
  PetscCall(PetscManagedScalarGetValues(dctxa, scal, dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_FALSE, &ptrc));
  PetscCall(PetscManagedScalarDestroy(dctxa, &scal));
  // if the mem pool is working correctly scal should get a completely fresh allocation from
  // the pool and share no data with its previous version
  PetscCall(PetscManagedScalarCreateDefault(dctxb, n, &scal));
  // now fill the array with 2's
  PetscCall(PetscManagedScalarSetValues(dctxb, scal, PETSC_MEMTYPE_HOST, ptrb, n));
  PetscCall(PetscManagedScalarGetValues(dctxb, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &ptrc));

  for (PetscInt i = 0; i < n; ++i)
    PetscCheck(ptrc[i] == ptrb[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "actual[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g, memory pool likely corrupted", i, (double)PetscRealPart(ptrc[i]), i, (double)PetscRealPart(ptrb[i]));

  PetscCall(PetscManagedScalarDestroy(dctxb, &scal));

  PetscCall(PetscDeviceFree(dctxb, ptra));
  PetscCall(PetscDeviceFree(dctxb, ptrb));

  PetscCall(PetscDeviceContextDestroy(&dctxa));
  PetscCall(PetscDeviceContextDestroy(&dctxb));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    suffix: cxx
    output_file: ./output/ExitSuccess.out
    args: -dctxa_device_context_stream_type {{default_blocking global_nonblocking}} \
          -dctxb_device_context_stream_type {{default_blocking global_nonblocking}}

    test:
      requires: !device
      suffix: host_no_device
    test:
      requires: device
      args: -default_device_type host
      suffix: host_with_device
    test:
      requires: cuda
      args: -default_device_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -default_device_type hip
      suffix: hip
    test:
      requires: sycl
      args: -default_device_type sycl
      suffix: sycl

  test:
    requires: !cxx
    suffix: no_cxx
    output_file: ./output/ExitSuccess.out
TEST*/
