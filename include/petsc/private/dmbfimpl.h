#if !defined(_DMBFIMPL_H)
#define _DMBFIMPL_H

#include <petscdmbf.h>

PETSC_EXTERN PetscErrorCode DMBFGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMBFGetGhost(DM,void*);

#endif /* defined(_DMBFIMPL_H) */
