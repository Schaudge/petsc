static char help[] = "Basic vector routines asynchronously\n\n";

#include <petscvec.h>
#include <petscdevice.h>

static PetscErrorCode PrintNorm(PetscDeviceContext dctx, PetscManagedReal norm, Vec v, PetscReal scaling, const char funcname[]) {
  PetscInt   n;
  PetscReal  norm_v;
  PetscReal *norm_ptr;
  MPI_Comm   comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCall(VecGetSize(v, &n));
  PetscCall(VecNormAsync(v, NORM_2, norm, dctx));
  PetscCall(PetscManagedRealGetArray(dctx, norm, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &norm_ptr));
  norm_v = (*norm_ptr) - (scaling * PetscSqrtReal((PetscReal)n));
  if (norm_v > -PETSC_SMALL && norm_v < PETSC_SMALL) norm_v = 0.0;
  PetscCall(PetscPrintf(comm, "%s %g\n", funcname, (double)norm_v));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  Vec              x, y, w; /* vectors */
  Vec             *z;       /* array of vectors */
  PetscManagedReal norm, v1, v2, maxval;
  PetscReal        v;

  PetscInt           n = 20;
  PetscManagedInt    maxind;
  PetscManagedScalar one, two, three, dots, dot;
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /*
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(), the vector format
     (currently parallel, shared, or sequential) is determined at runtime.  Also, the
     parallel partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate(), VecSetSizes() and VecSetFromOptions() the option -vec_type mpi or
     -vec_type shared causes the particular type of vector to be formed.

  */
  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD, NULL, dctx));

  PetscCall(VecCreateAsync(PETSC_COMM_WORLD, dctx, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
  PetscDeviceContext *subs;
  PetscCall(PetscDeviceContextFork(dctx, 3, &subs));
  PetscCall(VecDuplicateAsync(x, &y, dctx));
  PetscCall(VecDuplicateAsync(x, &w, subs[0]));

  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
  PetscCall(VecDuplicateVecs(x, 3, &z));
  /*
     Set the vectors to entries to a constant value.
  */

  PetscCall(PetscManagedScalarCreateDefault(subs[0], 1, &one));
  PetscCall(PetscManagedScalarCreateDefault(subs[1], 1, &two));
  PetscCall(PetscManagedScalarCreateDefault(subs[2], 1, &three));

  PetscCall(PetscManagedScalarSetValues(subs[0], one, PETSC_MEMTYPE_HOST, (PetscScalar[]){1.0}, 1));
  PetscCall(PetscManagedScalarSetValues(subs[1], two, PETSC_MEMTYPE_HOST, (PetscScalar[]){2.0}, 1));
  PetscCall(PetscManagedScalarSetValues(subs[2], three, PETSC_MEMTYPE_HOST, (PetscScalar[]){3.0}, 1));

  PetscCall(VecSetAsync(x, one, subs[0]));
  PetscCall(VecSetAsync(y, two, subs[1]));
  PetscCall(VecSetAsync(z[0], one, subs[0]));
  PetscCall(VecSetAsync(z[1], two, subs[1]));
  PetscCall(VecSetAsync(z[2], three, subs[2]));

  /*
     Demonstrate various basic vector routines.
  */
  PetscCall(PetscManagedScalarCreateDefault(subs[1], 1, &dot));
  // make subs[1] (owner of dot) wait for subs[0] (implicit owner of x)
  PetscCall(PetscDeviceContextWaitForContext(subs[1], subs[0]));
  PetscCall(VecDotAsync(x, y, dot, subs[1]));
  PetscCall(PetscManagedScalarDestroy(subs[1], &dot));

  PetscManagedInt ndots;
  PetscInt        ndots_v = 3;
  PetscCall(PetscManageHostInt(dctx, &ndots_v, 1, &ndots));
  PetscCall(PetscManagedScalarCreateDefault(dctx, ndots_v, &dots));

  PetscCall(PetscDeviceContextJoin(dctx, 3, PETSC_DEVICE_CONTEXT_JOIN_DESTROY, &subs));
  PetscCall(VecMDotAsync(x, ndots, z, dots, dctx));

  /*
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vector length %" PetscInt_FMT "\n", n));
  PetscCall(PetscManagedIntCreateDefault(dctx, 1, &maxind));
  PetscCall(PetscManagedRealCreateDefault(dctx, 1, &maxval));
  PetscCall(VecMaxAsync(x, maxind, maxval, dctx));
  PetscInt  *maxind_v;
  PetscReal *maxval_v;
  PetscCall(PetscManagedIntGetArray(dctx, maxind, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &maxind_v));
  PetscCall(PetscManagedRealGetArray(dctx, maxval, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &maxval_v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecMax %g, VecInd %" PetscInt_FMT "\n", (double)*maxval_v, *maxind_v));

  PetscCall(VecMinAsync(x, maxind, maxval, dctx));
  PetscCall(PetscManagedIntGetArray(dctx, maxind, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &maxind_v));
  PetscCall(PetscManagedRealGetArray(dctx, maxval, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &maxval_v));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecMin %g, VecInd %" PetscInt_FMT "\n", (double)*maxval_v, *maxind_v));
  PetscCall(PetscManagedIntDestroy(dctx, &maxind));
  PetscCall(PetscManagedRealDestroy(dctx, &maxval));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "All other values should be near zero\n"));
  PetscCall(PetscManagedRealCreateDefault(dctx, 1, &norm));
  PetscCall(VecScaleAsync(x, two, dctx));
  PetscCall(PrintNorm(dctx, norm, x, 2.0, "VecScale"));

  PetscCall(VecCopyAsync(x, w, dctx));
  PetscCall(PrintNorm(dctx, norm, x, 2.0, "VecCopy"));

  PetscCall(VecAXPYAsync(y, three, x, dctx));
  PetscCall(PrintNorm(dctx, norm, x, 8.0, "VecAXPY"));

  PetscCall(VecAYPXAsync(y, two, x, dctx));
  PetscCall(PrintNorm(dctx, norm, x, 18.0, "VecAYPX"));

  PetscCall(VecSwapAsync(x, y, dctx));
  PetscCall(PrintNorm(dctx, norm, y, 2.0, "VecSwap"));
  PetscCall(PrintNorm(dctx, norm, x, 18.0, "VecSwap"));

  PetscCall(VecWAXPYAsync(w, two, x, y, dctx));
  PetscCall(PrintNorm(dctx, norm, w, 38.0, "VecWAXPY"));

  PetscCall(VecPointwiseMultAsync(w, y, x, dctx));
  PetscCall(PrintNorm(dctx, norm, w, 36.0, "VecPointwiseMult"));

  PetscCall(VecPointwiseDivideAsync(w, x, y, dctx));
  PetscCall(PrintNorm(dctx, norm, w, 9.0, "VecPointwiseDivide"));

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;

  PetscCall(PetscManagedScalarCopy(dctx, dots, one));
  PetscCall(PetscManagedScalarShiftPointer(dots, 1));
  PetscCall(PetscManagedScalarCopy(dctx, dots, three));
  PetscCall(PetscManagedScalarShiftPointer(dots, 1));
  PetscCall(PetscManagedScalarCopy(dctx, dots, two));
  PetscCall(PetscManagedScalarResetShift(dots));

  PetscCall(VecSetAsync(x, one, dctx));
  PetscCall(VecMAXPYAsync(x, ndots, dots, z, dctx));

  PetscReal v_3[3];
  for (PetscInt i = 0; i < 3; ++i) { PetscCall(VecNormAsync(z[i], NORM_2, norm, dctx)); }
  PetscCall(PetscManagedRealGetArray(dctx, norm, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &norm_v));
  v = (*norm_v) - PetscSqrtReal((PetscReal)n);
  if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(VecNormAsync(z[1], NORM_2, norm, dctx));
  v = (*norm_v) - 2.0 * PetscSqrtReal((PetscReal)n);
  if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(VecNormAsync(z[2], NORM_2, norm, dctx));
  v = (*norm_v) - 3.0 * PetscSqrtReal((PetscReal)n);
  if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecMAXPY %g %g %g \n", (double)v, (double)v1, (double)v2));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(PetscManagedIntDestroy(dctx, &ndots));
  PetscCall(PetscManagedScalarDestroy(dctx, &dots));
  PetscCall(PetscManagedScalarDestroy(dctx, &one));
  PetscCall(PetscManagedScalarDestroy(dctx, &two));
  PetscCall(PetscManagedScalarDestroy(dctx, &three));
  PetscCall(PetscManagedRealDestroy(dctx, &norm));
  PetscCall(PetscDeviceContextDestroy(&dctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroyVecs(3, &z));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: todo broken

  testset:
    output_file: output/ex1_1.out
    # This is a test where the exact numbers are critical
    diff_args: -j

    test:

    test:
        suffix: cuda
        args: -vec_type cuda
        requires: cuda

    test:
        suffix: kokkos
        args: -vec_type kokkos
        requires: kokkos_kernels

    test:
        suffix: hip
        args: -vec_type hip
        requires: hip

    test:
        suffix: 2
        nsize: 2

    test:
        suffix: 2_cuda
        nsize: 2
        args: -vec_type cuda
        requires: cuda

    test:
        suffix: 2_kokkos
        nsize: 2
        args: -vec_type kokkos
        requires: kokkos_kernels

    test:
        suffix: 2_hip
        nsize: 2
        args: -vec_type hip
        requires: hip

TEST*/
