
/*
     Code for some of the parallel vector primitives.
*/
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>
#include <petsc/private/deviceimpl.h>

PetscErrorCode VecMDot_MPI(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Standard(xin, nv, y, z, dctx, VecMDot_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Standard(xin, nv, y, z, dctx, VecMTDot_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(VecNorm_MPI_Standard(xin, type, z, dctx, VecNorm_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx) {
  const MPI_Op ops[] = {MPIU_MAXLOC, MPIU_MAX};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Standard(xin, idx, z, dctx, VecMax_Seq, ops));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx) {
  const MPI_Op ops[] = {MPIU_MINLOC, MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Standard(xin, idx, z, dctx, VecMin_Seq, ops));
  PetscFunctionReturn(0);
}
