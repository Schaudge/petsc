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
  PETSCHEADER(struct _TaoPROXOps);        
  Tao subsolver;
  Mat vm; /* Variable Metric matrix */	

  Vec G_old;
  Vec X_old;
  Vec W; /*  work vector */

  PetscReal eta;       /*  Restart tolerance */
  PetscReal stepsize;     /*  Step size */

  PetscInt step_type;	  

  TaoPROXType strategy;

  void  *orig_objP;
  void  *orig_objgradP;
  void  *orig_gradP;
  void  *orig_hessP;
} TAO_PROX;

#endif /* ifndef __TAO_PROX_H */
