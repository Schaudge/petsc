static const char help[] = "Tests PetscManagedIntEqual() and PetscManagedIntKnownAndEqual()\n\n";

#include <petscdevice.h>

static PetscErrorCode CheckEqual(PetscManagedInt mint, PetscInt value, PetscBool expected_known, PetscBool expected_equal) {
  const PetscBool expected_known_and_equal = (PetscBool)(expected_known && expected_equal);
  PetscBool       known, equal, known_and_equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntEqual(mint, value, &known, &equal));
  PetscCheck(known == expected_known, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscManagedIntEqual(): actual known %s != expected known %s for value %" PetscInt_FMT, PetscBools[known], PetscBools[expected_known], value);
  if (expected_known) PetscCheck(equal == expected_equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscManagedIntEqual(): actual equal %s != expected equal %s for value %" PetscInt_FMT, PetscBools[equal], PetscBools[expected_equal], value);
  known_and_equal = PetscManagedIntKnownAndEqual(mint, value);
  PetscCheck(known_and_equal == expected_known_and_equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in PetscManagedIntKnownAndEqual(): actual known_and_equal %s != expected known_and_equal %s for value %" PetscInt_FMT, PetscBools[known_and_equal], PetscBools[expected_known_and_equal], value);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckHost(PetscDeviceContext dctx, PetscInt n) {
  PetscInt       *host, *host_values;
  PetscDeviceType dtype;
  PetscManagedInt mint;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntCreate(dctx, NULL, NULL, 1, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED, &mint));
  PetscCall(CheckEqual(mint, 12345, PETSC_TRUE, PETSC_FALSE));
  PetscCall(PetscManagedIntDestroy(dctx, &mint));

  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n, &host));
  for (PetscInt i = 0; i < n; ++i) host[i] = 100;
  PetscCall(PetscManageHostInt(dctx, host, n, &mint));

  // only have host values, so answer should be known both times
  PetscCall(CheckEqual(mint, 100, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 105, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscMalloc1(n, &host_values));
  for (PetscInt i = 0; i < n; ++i) host_values[i] = 15;
  PetscCall(PetscManagedIntSetValues(dctx, mint, PETSC_MEMTYPE_HOST, host_values, n));
  PetscCall(PetscFree(host_values));

  // only have host values, so answer should be known both times
  PetscCall(CheckEqual(mint, 15, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 45, PETSC_TRUE, PETSC_FALSE));

  // managehostint() just borrrows the pointer, so manipulating it directly achieves the same
  // result as setvalues()
  for (PetscInt i = 0; i < n; ++i) host[i] = 12345;

  // only have host values, so answer should be known both times
  PetscCall(CheckEqual(mint, 12345, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 123456, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscManagedIntGetValues(dctx, mint, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &host_values));
  for (PetscInt i = 0; i < n; ++i) host_values[i] = -1;

  // only have host values, so answer should be known both times
  PetscCall(CheckEqual(mint, -1, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 16, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dtype != PETSC_DEVICE_HOST) {
    PetscInt *device_values;

    PetscCall(PetscManagedIntGetValues(dctx, mint, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &device_values));
    // should be unchanged
    PetscCall(CheckEqual(mint, -1, PETSC_TRUE, PETSC_TRUE));
    PetscCall(CheckEqual(mint, -123, PETSC_TRUE, PETSC_FALSE));

    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, n, &device_values));
    PetscCall(PetscManagedIntSetValues(dctx, mint, PETSC_MEMTYPE_DEVICE, device_values, n));
    PetscCall(PetscDeviceFree(dctx, device_values));

    // should no longer be known
    PetscCall(CheckEqual(mint, 0, PETSC_FALSE, PETSC_TRUE));
    PetscCall(CheckEqual(mint, 42, PETSC_FALSE, PETSC_FALSE));

    PetscCall(PetscManagedIntEnsureOffload(dctx, mint, PETSC_OFFLOAD_CPU, PETSC_FALSE));
    // should still not be known (didn't synchronize)
    PetscCall(CheckEqual(mint, 0, PETSC_FALSE, PETSC_TRUE));
    PetscCall(CheckEqual(mint, 50, PETSC_FALSE, PETSC_FALSE));

    PetscCall(PetscManagedIntEnsureOffload(dctx, mint, PETSC_OFFLOAD_CPU, PETSC_TRUE));
    // should know again
    PetscCall(CheckEqual(mint, 0, PETSC_TRUE, PETSC_TRUE));
    PetscCall(CheckEqual(mint, 50, PETSC_TRUE, PETSC_FALSE));
  }
  PetscCall(PetscManagedIntDestroy(dctx, &mint));
  PetscCall(PetscDeviceFree(dctx, host));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckDevice(PetscDeviceContext dctx, PetscInt n) {
  PetscDeviceType dtype;
  PetscManagedInt mint;
  PetscInt       *device, *device_values, *host_values;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntCreate(dctx, NULL, NULL, 1, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED, &mint));
  PetscCall(CheckEqual(mint, 12345, PETSC_TRUE, PETSC_FALSE));
  PetscCall(PetscManagedIntDestroy(dctx, &mint));

  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dtype == PETSC_DEVICE_HOST) PetscFunctionReturn(0);

  PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, n, &device));
  PetscCall(PetscManageDeviceInt(dctx, device, n, &mint));
  // only have device values, so answer should not be known
  PetscCall(CheckEqual(mint, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 1, PETSC_FALSE, PETSC_FALSE));

  PetscCall(PetscManagedIntGetValues(dctx, mint, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &host_values));
  // now have host values, answer should be known
  PetscCall(CheckEqual(mint, 0, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, 1, PETSC_TRUE, PETSC_FALSE));

  for (PetscInt i = 0; i < n; ++i) host_values[i] = 25;
  // answer should still be known
  PetscCall(CheckEqual(mint, 25, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscManagedIntGetValues(dctx, mint, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_FALSE, &device_values));
  // back down to only device, answer should not be known
  PetscCall(CheckEqual(mint, 25, PETSC_FALSE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_FALSE, PETSC_FALSE));

  PetscCall(PetscManagedIntEnsureOffload(dctx, mint, PETSC_OFFLOAD_BOTH, PETSC_FALSE));
  // answer should still not be not be known (didn't synchronize)
  PetscCall(CheckEqual(mint, 25, PETSC_FALSE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_FALSE, PETSC_FALSE));

  PetscCall(PetscManagedIntEnsureOffload(dctx, mint, PETSC_OFFLOAD_BOTH, PETSC_TRUE));
  // answer should now be known again
  PetscCall(CheckEqual(mint, 25, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscManagedIntGetValues(dctx, mint, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &device_values));
  // device only reads, answer should stil be known
  PetscCall(CheckEqual(mint, 25, PETSC_TRUE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_TRUE, PETSC_FALSE));

  PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, n, &device_values));
  PetscCall(PetscManagedIntSetValues(dctx, mint, PETSC_MEMTYPE_DEVICE, device_values, n));
  PetscCall(PetscDeviceFree(dctx, device_values));
  // device has written, answer should no longer be known
  PetscCall(CheckEqual(mint, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(CheckEqual(mint, -123, PETSC_FALSE, PETSC_FALSE));

  PetscCall(PetscManagedIntDestroy(dctx, &mint));
  PetscCall(PetscDeviceFree(dctx, device));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  MPI_Comm           comm;
  PetscDeviceContext dctx;
  PetscInt           n = 1;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, NULL, "Test Options", NULL);
  PetscCall(PetscOptionsInt("-n", "Size of managed ints", NULL, n, &n, NULL));
  PetscOptionsEnd();

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(CheckHost(dctx, n));
  PetscCall(CheckDevice(dctx, n));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: cxx
    suffix: cxx
    output_file: ./output/ExitSuccess.out
    args: -device_enable {{lazy eager}} -n {{1 5}} \
         -root_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}

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
    args: -n {{1 5}}
TEST*/
