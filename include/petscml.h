#pragma once

#include <petscregressor.h>
#include <petsctao.h>

/* SUBMANSEC = ML */

/*S
     ML - Abstract PETSc ML Object

   Level: beginner
S*/
typedef struct _p_ML *ML;

PETSC_EXTERN PetscErrorCode MLCreate(MPI_Comm, ML *);
PETSC_EXTERN PetscErrorCode MLDestroy(ML *);
