/*
 Context for limited memory variable metric method for unconstrained
 optimization.
*/

#pragma once

#include <petscdevice.h>
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat M;

  Vec X;
  Vec G;
  Vec D;
  Vec W;

  Vec Xold;
  Vec Gold;

  PetscInt bfgs;
  PetscInt grad;
  Mat      H0;

  PetscBool recycle;

  PetscDeviceContext dctx;
} TAO_LMVM;
