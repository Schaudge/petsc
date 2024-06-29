#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {

  DM smoothterm, reg, g_prox, h_prox;

  Mat g_lmap;
  Vec workvec, workvec2, grad_old, x_old;

  PetscReal step_old;
  PetscReal gnorm_norm;
  PetscReal g_lmap_norm;
  PetscReal lip;

  PetscBool use_accel, use_adapt, approx_lip, lip_set;
} TAO_CV;
