#if !defined(PETSCDMP4ESTFV_H)
#define PETSCDMP4ESTFV_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMP4estFVGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMP4estFVGetGhost(DM,void*);

#endif /* defined(PETSCDMP4ESTFV_H) */
