#if !defined(PETSCHYBRID_H)
#define PETSCHYBRID_H

#include <petscvec.h>

#include <libaxb.h>

PETSC_EXTERN PetscErrorCode VecHybridGetArray(Vec v, struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecHybridRestoreArray(Vec v, struct axbVec_s **a);

PETSC_EXTERN PetscErrorCode VecHybridGetArrayRead(Vec v, const struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecHybridRestoreArrayRead(Vec v, const struct axbVec_s **a);

PETSC_EXTERN PetscErrorCode VecHybridGetArrayWrite(Vec v, struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecHybridRestoreArrayWrite(Vec v, struct axbVec_s **a);


#endif
