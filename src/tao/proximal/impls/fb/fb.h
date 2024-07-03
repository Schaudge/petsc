#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  DM reg; /*dm1 term for DMTaoApplyProximalMap */
  DM smoothterm;

  DM proxterm;

  Mat lmap;

  PetscReal lmap_norm;

  Vec s_vec_lv;
  Vec workvec, workvec2, workvec3, dualvec, dualwork, dualwork2, x_old, grad_old;

  PetscReal eta; /* backtracking constant */
  PetscReal t_fista, t_fista_old, fista_beta;
  PetscReal lip; /* L_f */
  PetscReal mu_fg; /* mu_f + mu_g */

  PetscReal step_old;
  PetscReal gnorm_norm; /* Normalizer for relative error. max(|gradf|, |xdiff/step|) + eps */
  PetscReal f_scale;
  PetscReal prox_scale;

  PetscBool use_accel;
  PetscBool use_adapt;

  PetscReal bb_param;
  PetscBool approx_lip;
  PetscBool lip_set;
  PetscBool mu_set;
} TAO_FB;
