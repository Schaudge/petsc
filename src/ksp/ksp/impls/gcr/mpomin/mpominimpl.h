/*****************************************************************************/
/* Context for using multipreconditioned orthomin method           */
/*****************************************************************************/

#if !defined(__OMIN_MP)
#define __OMIN_MP

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscScalar eigtrunc;       /*Threshold for eigenvalue-based pseudo inversion*/
} KSP_MPOMIN;

#endif

