static const char help[] = "Tests PetscManagedTypeGetSubRange().\n\n";

#include <petscdevice.h>

static PetscErrorCode TestChildSubRange(PetscDeviceContext dctx, PetscManagedReal child, const PetscReal *parent_values, const PetscReal *reference, PetscInt parent_begin, PetscInt global_begin) {
  PetscReal *child_values;
  PetscInt   n;

  PetscFunctionBegin;
  PetscCall(PetscManagedRealGetArray(dctx, child, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &child_values));
  PetscCall(PetscManagedRealGetSize(child, &n));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt global_idx = global_begin + i;
    const double   val        = (double)child_values[i];
    const double   pval       = (double)parent_values[parent_begin + i];
    const double   ref        = (double)reference[global_idx];

    // check parent is coherent
    PetscCheck(pval == ref, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "parent_values[(local %" PetscInt_FMT ", global %" PetscInt_FMT ")] %g != reference[(global %" PetscInt_FMT ")] %g", parent_begin + i, global_idx, pval, global_idx, ref);
    // check child is coherent
    PetscCheck(val == ref, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "child_values[(local %" PetscInt_FMT ", global %" PetscInt_FMT ")] %g != reference[(global %" PetscInt_FMT ")] %g", i, global_idx, val, global_idx, ref);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestNestedGetSubRange(PetscDeviceContext dctx, PetscManagedReal parent, PetscInt global_begin, const PetscReal *reference) {
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(PetscManagedRealGetSize(parent, &n));
  if (n) {
    PetscManagedReal sub;
    const PetscInt   nchild = n / 2, parent_begin = n / 4, adjusted_global_begin = global_begin + parent_begin;
    PetscReal       *parent_values;

    PetscCall(PetscManagedRealGetArray(dctx, parent, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &parent_values));
    PetscCall(PetscManagedRealGetSubRange(dctx, parent, parent_begin, nchild, &sub));
    {
      PetscManagedReal copy;

      PetscCall(PetscManagedRealCreateDefault(dctx, nchild, &copy));
      PetscCall(PetscManagedRealCopy(dctx, copy, sub));
      PetscCall(PetscManagedRealDestroy(dctx, &copy));
    }

    // check
    PetscCall(TestChildSubRange(dctx, sub, parent_values, reference, parent_begin, adjusted_global_begin));
    // recurse
    PetscCall(TestNestedGetSubRange(dctx, sub, adjusted_global_begin, reference));
    PetscCall(PetscManagedRealRestoreSubRange(dctx, parent, &sub));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  PetscDeviceContext dctx;
  PetscInt           n = 30;
  PetscReal         *arr, *backup_arr;
  PetscManagedReal   rl;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // allocate separate because this test also implicitly tests PETSC_OWN_POINTER for managed
  // real
  PetscCall(PetscMalloc1(n, &arr));
  PetscCall(PetscMalloc1(n, &backup_arr));
  for (PetscInt i = 0; i < n; ++i) arr[i] = backup_arr[i] = (PetscReal)i;

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscManagedRealCreate(dctx, arr, NULL, n, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_CPU, &rl));

  // iteratively stride over the array getting ever larger chunks
  for (PetscInt size = 1; size < n; ++size) {
    for (PetscInt i = 0; i < n; ++i) {
      const PetscInt   adjusted_size = i + size > n ? n - i : size;
      PetscManagedReal sub_rl;
      PetscReal       *sub_v;

      PetscCall(PetscManagedRealGetSubRange(dctx, rl, i, adjusted_size, &sub_rl));
      PetscCall(PetscManagedRealGetArray(dctx, sub_rl, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &sub_v));
      for (PetscInt j = 0; j < adjusted_size; ++j)
        PetscCheck(arr[i + j] == sub_v[j], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "child_values[(local %" PetscInt_FMT ", global %" PetscInt_FMT ")] %g != reference[(global %" PetscInt_FMT ")] %g", j, j + i, (double)sub_v[j], i + j, (double)(arr[i + j]));
      PetscCall(PetscManagedRealRestoreSubRange(dctx, rl, &sub_rl));
    }
  }
  // recursively cut the range in half each time
  PetscCall(TestNestedGetSubRange(dctx, rl, 0, backup_arr));

  PetscCall(PetscManagedRealDestroy(dctx, &rl));
  PetscCall(PetscFree(backup_arr));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 testset:
   requires: cxx
   suffix: cxx
   output_file: ./output/ExitSuccess.out
   args: -device_enable {{lazy eager}}                                                         \
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
   output_file: ./output/ExitSuccess.out
   suffix: no_cxx

TEST*/
