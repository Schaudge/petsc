#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  TaoFBType type;

  DM reg; //dm1 term for DMTaoApplyProximalMap
  DM smoothterm;

  DM proxterm;

  Mat lmap;

  Vec s_vec_lv;
  Vec workvec, workvec2, dualvec, x_old, grad_old;

  PetscReal eta; //backtracking constant
  PetscReal t_fista, t_fista_old;
  PetscReal lip; //Lipschitz constant of f(x)
  PetscReal mu_f; //strong convexity constant of f(x) (if exists)

  PetscReal tau_lv, sigma_lv, rho_lv; //tau, sigma for LV. sigma=1/(tau |L|^2)

  PetscReal gnorm_norm; //Normalizer for relative error.
                        //max(|gradf|, |xdiff/step|) + eps

  PetscReal bb_param;
  PetscBool approx_lip; //approximate L in the beginning
  PetscBool lip_set;
  PetscBool mu_set;

} TAO_FB;
