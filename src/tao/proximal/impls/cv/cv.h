#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {

  DM smoothterm, reg, g_prox, h_prox;

  Mat g_lmap;
  Vec workvec, workvec2, grad_old, x_old, v;
  Vec dualvec, dualvec_old, dualvec_work, Ax, Ax_old, ATy;
  Vec dualvec_test, ATy_test;

  PetscReal step_old;
  PetscReal gnorm_norm;
  PetscReal g_lmap_norm;
  PetscReal f_scale;
  PetscReal g_scale;
  PetscReal h_scale;
  PetscReal lip;
  PetscReal rho;
  PetscReal sigma;
  PetscReal eta; // lmap_norm estimate
  PetscReal nu; // usually 1.1 <= eta <= 1.5
  PetscReal pd_ratio; //t variable
  PetscReal R; //scale factor for estimating linear map norm. Must be <= 1.
  PetscReal r; //backtracking parameter > 1

  PetscBool use_accel, use_adapt, approx_lip, lip_set, lmap_norm_set, approx_lmap_norm;
} TAO_CV;
