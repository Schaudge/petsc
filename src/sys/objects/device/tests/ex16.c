static const char help[] = "Tests PetscDeviceContext auto dependency-graph generation\n\n";

#include "petscdevicetestcommon.h"

#define PetscVerboseFunctionBegin \
  PetscFunctionBegin; \
  do { \
    static PetscInt i = 0; \
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "<- %s() %" PetscInt_FMT " begin ->\n", PETSC_FUNCTION_NAME, i))

#define PetscVerboseFunctionReturn(...) \
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "<- %s() %" PetscInt_FMT " end   ->\n", PETSC_FUNCTION_NAME, i)); \
  ++i; \
  } \
  while (0) \
    ; \
  PetscFunctionReturn(__VA_ARGS__)

static PetscErrorCode mark(PetscDeviceContext dctx, PetscObject obj, PetscMemoryAccessMode mode) {
  const char   *name;
  PetscObjectId id;

  PetscFunctionBegin;
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscObjectGetId(obj, &id));
  PetscCall(PetscObjectGetName(obj, &name));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, id, mode, name));
  PetscFunctionReturn(0);
}

static PetscErrorCode norm(PetscObject x, PetscDeviceContext dctx) {
  PetscVerboseFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ));
  PetscVerboseFunctionReturn(0);
}

static PetscErrorCode scale(PetscObject x, PetscDeviceContext dctx) {
  PetscVerboseFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ_WRITE));
  PetscVerboseFunctionReturn(0);
}

static PetscErrorCode set(PetscObject x, PetscDeviceContext dctx) {
  PetscVerboseFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_WRITE));
  PetscVerboseFunctionReturn(0);
}

static PetscErrorCode dot(PetscObject x, PetscObject y, PetscDeviceContext dctx) {
  PetscVerboseFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ));
  PetscCall(mark(dctx, y, PETSC_MEMORY_ACCESS_READ));
  PetscVerboseFunctionReturn(0);
}

static PetscErrorCode axpy(PetscObject x, PetscObject y, PetscDeviceContext dctx) {
  PetscVerboseFunctionBegin;
  PetscCall(mark(dctx, x, PETSC_MEMORY_ACCESS_READ_WRITE));
  PetscCall(mark(dctx, y, PETSC_MEMORY_ACCESS_READ));
  PetscVerboseFunctionReturn(0);
}

static PetscErrorCode sync(MPI_Comm comm, PetscDeviceContext dctx) {
  static PetscInt i = 0;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm, "==== sync %" PetscInt_FMT " begin ====\n", i));
  PetscCall(AssertDeviceContextExists(dctx));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscPrintf(comm, "==== sync %" PetscInt_FMT " end   ====\n", i));
  ++i;
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintID(MPI_Comm comm, PetscObjectId id, const char prefix[], const char name[]) {
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm, "Object '%s' is %s %" PetscInt64_FMT "\n", name, prefix, id));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextPrintID(MPI_Comm comm, PetscDeviceContext dctx, const char name[]) {
  PetscFunctionBegin;
  PetscCall(PrintID(comm, dctx->id, "dctx", name));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePetscObject(MPI_Comm comm, const char name[], PetscObject *obj) {
  PetscObjectId  id;
  PetscContainer container;

  PetscFunctionBegin;
  // use PetscContainer as a proxy for PetscObject (since you cannot create them directly)
  PetscCall(PetscContainerCreate(comm, &container));
  PetscCall(PetscObjectGetId((PetscObject)container, &id));
  PetscCall(PetscObjectSetName((PetscObject)container, name));
  PetscCall(PrintID(comm, id, "object", name));
  *obj = (PetscObject)container;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  MPI_Comm           comm;
  PetscObject        x, y;
  PetscDeviceContext dctx_0, dctx_1, dctx_2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscDeviceContextCreate(&dctx_0));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_0_", dctx_0));
  PetscCall(PetscDeviceContextDuplicate(dctx_0, &dctx_1));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_1_", dctx_1));
  PetscCall(PetscDeviceContextDuplicate(dctx_0, &dctx_2));
  PetscCall(PetscDeviceContextSetFromOptions(comm, "dctx_2_", dctx_2));

  PetscCall(PetscDeviceContextPrintID(comm, dctx_0, "dctx_0"));
  PetscCall(PetscDeviceContextPrintID(comm, dctx_1, "dctx_1"));
  PetscCall(PetscDeviceContextPrintID(comm, dctx_2, "dctx_2"));
  PetscCall(CreatePetscObject(comm, "x", &x));
  PetscCall(CreatePetscObject(comm, "y", &y));

  PetscCall(PetscPrintf(comm, "=== BEGIN ===\n"));
  PetscCall(dot(x, y, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]],
      y : [[ctx_0,Access.READ]]
    }
  */
  PetscCall(dot(x, y, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]]
    }
   */
  PetscCall(sync(comm, dctx_0));
  /*
    call_map = {
      x : [[ctx_1,Access.READ]],
      y : [[ctx_1,Access.READ]]
    }
  */
  PetscCall(sync(comm, dctx_1));
  /*
    call_map = {}
  */

  /* ================================================================================ */

  PetscCall(norm(x, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]]
    }
  */
  PetscCall(norm(x, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]]
    }
  */
  PetscCall(set(x, dctx_2));
  /*
    call_map = {
      x : [[ctx_2,Access.WRITE]]
    }
  */
  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {}
  */

  /* ================================================================================ */

  PetscCall(dot(x, y, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]],
      y : [[ctx_0,Access.READ]],
    }
  */
  PetscCall(dot(x, y, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */

  PetscCall(sync(comm, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]],
      y : [[ctx_0,Access.READ]],
    }
  */

  PetscCall(dot(x, y, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]],
      y : [[ctx_0,Access.READ]],
    }
  */
  PetscCall(axpy(x, y, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.READ_WRITE]],
      y : [[ctx_0,Access.READ]],
    }
  */
  PetscCall(axpy(x, y, dctx_1));
  /*
    call_map = {
      x : [[ctx_1,Access.READ_WRITE]],
      y : [[ctx_0,Access.READ],[ctx_1.READ]],
    }
  */
  PetscCall(sync(comm, dctx_1));
  /*
    assert call_map == {}

    dctx_1 waited for dctx_0 during the first axpy(), so synchronizing here has the effect of
    implicitly synchronizing the read on y
  */

  /* ================================================================================ */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }

    dctx_2 never participated prior, so no effect
  */

  PetscCall(set(x, dctx_1));
  /*
    call_map = {
      x : [[ctx_1,Access.WRITE]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(set(x, dctx_2));
  /*
    call_map = {
      x : [[ctx_2,Access.WRITE]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(set(x, dctx_0));
  /*
    call_map = {
      x : [[ctx_0,Access.WRITE]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(scale(x, dctx_2));
  /*
    call_map = {
      x : [[ctx_2,Access.WRITE]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {}

    dctx_2 has transitively serialized with everyone, so syncing on it clears the map
  */

  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {}

    no effect
  */

  PetscCall(dot(x, y, dctx_0));
  PetscCall(dot(x, y, dctx_1));
  /*
    call_map = {
      x : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
      y : [[ctx_0,Access.READ],[ctx_1,Access.READ]],
    }
  */
  PetscCall(PetscDeviceContextWaitForContext(dctx_2, dctx_1));
  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {
      x : [[ctx_0,Access.READ]],
      y : [[ctx_0,Access.READ]]
    }

    dctx_2 is serialized with dctx_1 (so should remove its dependencies) but not with dctx_0
  */

  PetscCall(PetscDeviceContextWaitForContext(dctx_1, dctx_0));
  PetscCall(PetscDeviceContextWaitForContext(dctx_2, dctx_1));
  PetscCall(sync(comm, dctx_2));
  /*
    call_map = {}

    dctx_2 now transitively serialized with dctx_0
  */

  PetscCall(PetscObjectDestroy(&x));
  PetscCall(PetscObjectDestroy(&y));

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
