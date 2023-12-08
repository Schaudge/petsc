/*
   Context for a Newton trust region method for solving a system
   of nonlinear equations
 */

#pragma once
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscReal delta;  /* trust region parameter */
  PetscReal delta0; /* initial radius for trust region */
  PetscReal deltaM; /* maximum radius for trust region */
  PetscReal kmdc;   /* sufficient decrease parameter */

  /*
    Given rho = (fk - fkp1) / (m(0) - m(pk))

    The radius is modified as:
      rho < eta2 -> delta *= t1
      rho > eta3 -> delta *= t2
      delta = min(delta,deltaM)

    The step is accepted if rho > eta1
  */
  PetscReal eta1;
  PetscReal eta2;
  PetscReal eta3;
  PetscReal t1;
  PetscReal t2;

  /* Use quasi-Newton models for J and (possibly different) Jp */
  SNESNewtonTRQNType qn;
  Mat                qnB;
  Mat                qnB_pre;

  /* The type of norm for the trust region */
  NormType norm;

  SNESNewtonTRFallbackType fallback; /* enum to distinguish fallback in case Newton step is outside of the trust region */

  PetscErrorCode (*precheck)(SNES, Vec, Vec, PetscBool *, void *);
  void *precheckctx;
  PetscErrorCode (*postcheck)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  void *postcheckctx;
} SNES_NEWTONTR;
