#pragma once

#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF, MPI_Datatype, PetscMemType, void *, PetscMemType, const void *, void *, MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFReducePrepareMPIBuffers_Gatherv(PetscSF, PetscSFLink, MPI_Op, PetscMemType *, void **, PetscMemType *, const void **);
PETSC_INTERN PetscErrorCode PetscSFAllreduceBegin_Gatherv(PetscSF, MPI_Datatype, PetscMemType, const void *, PetscMemType, void *, MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFAllreduceEnd_Gatherv(PetscSF, MPI_Datatype, const void *, void *, MPI_Op);
