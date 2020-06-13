#ifndef PETSCSINDYPDE_H
#define PETSCSINDYPDE_H

#include <petscvec.h>
#include <petscdmda.h>

PETSC_EXTERN PetscErrorCode GetData(PetscInt*, Vec**, Vec**, PetscReal**, DM*);

#endif
