/*
    Context for proximal (unconstrained minimization)
 */

#ifndef __TAO_PROX_H
#define __TAO_PROX_H

#include <petsc/private/taoimpl.h>

typedef struct _TaoPROXOps *TaoPROXOps;

struct _TaoPROXOps {
  PetscErrorCode (*orig_obj)(Tao, Vec, PetscReal *, void *);
  PetscErrorCode (*orig_objgrad)(Tao, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*orig_grad)(Tao, Vec, Vec, void *);
  PetscErrorCode (*orig_hess)(Tao, Vec, Mat, Mat, void *);
};

typedef struct {
  PetscReal lb;
  PetscReal ub;
} TAO_PROX_L1;

typedef struct {
  Vec a;        
} TAO_PROX_AFFINE;

typedef struct {
  PETSCHEADER(struct _TaoPROXOps);        
  Tao subsolver;

  TAO_PROX_L1     *L1;
  TAO_PROX_AFFINE *affine;

  Mat vm; /* Variable Metric matrix */	
  Mat H_orig, H_pre_orig;

  Vec G_old, X_old, workvec1;
  Vec y; /* Input y vector for prox(y) */

  PetscReal eta;       /*  Restart tolerance */
  PetscReal stepsize, stepsize_old;     /*  Step size */

  PetscInt step_type;	  

  TaoPROXStrategy strategy;
  TaoPROXType type;

  TaoMetricType metric_type;

  void  *orig_objP;
  void  *orig_objgradP;
  void  *orig_gradP;
  void  *orig_hessP;
} TAO_PROX;

#endif /* ifndef __TAO_PROX_H */
