#ifndef ROSENBROCK1_H_
#define ROSENBROCK1_H_

#include <petscmat.h>

PETSC_EXTERN PetscErrorCode Rosenbrock1ObjAndGradCUDA(Vec , Vec , PetscReal *, PetscReal , PetscInt);

#endif 
