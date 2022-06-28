static const char help[] = "Tests PetscDeviceContext auto dependency-graph generation\n\n";

#include "petscdevicetestcommon.h"

static PetscErrorCode mark(PetscDeviceContext dctx, PetscObject obj, PetscMemoryAccessMode mode) {
  PetscObjectId id;

  PetscFunctionBegin;
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscObjectGetId(obj, &id));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, id, mode));
  PetscFunctionReturn(0);
}

static PetscErrorCode dot(PetscObject x, PetscObject y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ));
  PetscCall(mark(dctx, y, PETSC_MEMORY_ACCESS_READ));
  PetscFunctionReturn(0);
}

static PetscErrorCode axpy(PetscObject x, PetscObject y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ_WRITE));
  PetscCall(mark(dctx, y, PETSC_MEMORY_ACCESS_READ));
  PetscFunctionReturn(0);
}

static PetscErrorCode scale(PetscObject x, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ_WRITE));
  PetscFunctionReturn(0);
}

static PetscErrorCode set(PetscObject x, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_WRITE));
  PetscFunctionReturn(0);
}

static PetscErrorCode sync(MPI_Comm comm, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscPrintf(comm, "==== sync ====\n"));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  MPI_Comm           comm;
  PetscObject        x, y;
  PetscContainer     x_c, y_c;
  PetscDeviceContext dctx_0, dctx_1, dctx_2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscDeviceContextCreate(&dctx_0));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_0_", dctx_0));
  PetscCall(PetscDeviceContextDuplicate(dctx_0, &dctx_1));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_1_", dctx_1));
  PetscCall(PetscDeviceContextDuplicate(dctx_0, &dctx_2));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_2_", dctx_2));

  // user PetscContainer as proxy for a generic PetscObject
  PetscCall(PetscContainerCreate(comm, &x_c));
  x = (PetscObject)x_c;
  PetscCall(PetscContainerCreate(comm, &y_c));
  y = (PetscObject)y_c;

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  PetscCall(sync(comm, dctx_0));
  /*
    assert sorted(tuple(call_map.keys())) == sorted((x,y))
    assert call_map[x] == [[ctx_1,Access.READ]]
    assert call_map[y] == [[ctx_1,Access.READ]]
  */
  PetscCall(sync(comm, dctx_1));
  /*
    assert call_map == {}
  */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  PetscCall(sync(comm, dctx_1));
  /*
    assert sorted(tuple(call_map.keys())) == sorted((x,y))
    assert call_map[x] == [[ctx_0,Access.READ]]
    assert call_map[y] == [[ctx_0,Access.READ]]
  */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(axpy(x, y, dctx_0));
  PetscCall(axpy(x, y, dctx_1));
  PetscCall(sync(comm, dctx_1));
  /*
    assert call_map == {}
  */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  PetscCall(sync(comm, dctx_2));
  /*
    assert sorted(tuple(call_map.keys())) == sorted((x,y))
    assert call_map[x] == [[ctx_0,Access.READ],[ctx_1,Access.READ]]
    assert call_map[y] == [[ctx_0,Access.READ],[ctx_1,Access.READ]]
  */

  PetscCall(set(x, dctx_1));
  PetscCall(set(x, dctx_2));
  PetscCall(set(x, dctx_0));
  PetscCall(scale(x, dctx_2));
  PetscCall(sync(comm, dctx_2));
  /*
    assert call_map == {}
  */

  PetscCall(sync(comm, dctx_2));
  /*
    assert call_map == {}
  */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  PetscCall(PetscDeviceContextWaitForContext(dctx_2, dctx_1));
  PetscCall(sync(comm, dctx_2));
  /*
    assert sorted(tuple(call_map.keys())) == sorted((x,y))
    assert call_map[x] == [[ctx_0,Access.READ]]
    assert call_map[y] == [[ctx_0,Access.READ]]
  */

  PetscCall(sync(comm, dctx_0));
  /*
    assert call_map == {}
  */

  PetscCall(PetscContainerDestroy(&x_c));
  PetscCall(PetscContainerDestroy(&y_c));

  PetscCall(PetscDeviceContextDestroy(&dctx_0));
  PetscCall(PetscDeviceContextDestroy(&dctx_1));
  PetscCall(PetscDeviceContextDestroy(&dctx_2));

  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: defined(PETSC_USE_INFO) defined(PETSC_USE_DEBUG)
    suffix: info_debug
    output_file: ./output/ex16_info_debug.out
    args: -info
    filter: grep -E -ve "Petsc(_)?(Inner)?(Outer)?Comm" -ve "Petsc(Device)?Initialize" \
    -ve Petsc_Counter -ve PetscDetermineInitialFPTrap -ve PetscDeviceContextSetUp \
    -ve PetscGetHostName -ve "Configured device"

    test:
      requires: !device cxx
      suffix: host_no_device
    test:
      requires: device cxx
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

  testset:
    requires: defined(PETSC_USE_INFO) !defined(PETSC_USE_DEBUG)
    suffix: info_no_debug
    output_file: ./output/ex16_info_no_debug.out
    args: -info
    filter: grep -E -ve "Petsc(_)?(Inner)?(Outer)?Comm" -ve "Petsc(Device)?Initialize" \
    -ve Petsc_Counter -ve PetscDetermineInitialFPTrap -ve PetscDeviceContextSetUp \
    -ve PetscGetHostName -ve "Configured device"

    test:
      requires: !device cxx
      suffix: host_no_device
    test:
      requires: device cxx
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
    suffix: no_info
TEST*/
