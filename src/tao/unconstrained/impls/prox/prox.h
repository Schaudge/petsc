/*
    Context for proximal (unconstrained minimization)
 */

#ifndef __TAO_PROX_H
#define __TAO_PROX_H

#include <petsc/private/taoimpl.h>

typedef struct {
  Mat VM; /* Variable Metric matrix */	

  Vec G_old;
  Vec X_old;
  Vec W; /*  work vector */

  PetscReal eta;       /*  Restart tolerance */
  PetscReal gamma      /*  Step size */

  PetscInt step_type;	  
} TAO_PROX;

#define PROX_DEFAULT  0
#define PROX_ADAPTIVE 1
#define PROX_VM       2

#endif /* ifndef __TAO_PROX_H */
