/*****************************************************************************/
/* Context for using multipreconditioned conjugate gradient method           */
/*****************************************************************************/

#if !defined(__CG_MP)
#define __CG_MP

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscScalar eigtrunc;       /*Threshold for eigenvalue-based pseudo inversion*/
} KSP_MPCG;

#endif

