#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  void *data;
  PetscErrorCode (*applyproximalmap)(DM, DM, PetscReal, Vec, Vec, PetscBool);
} DMTao_Shell;
