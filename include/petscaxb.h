#if !defined(PETSCAXB_H)
#define PETSCAXB_H

#include <petscvec.h>

#include <libaxb.h>

PETSC_EXTERN PetscErrorCode VecAXBGetArray(Vec v, struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecAXBRestoreArray(Vec v, struct axbVec_s **a);

PETSC_EXTERN PetscErrorCode VecAXBGetArrayRead(Vec v, const struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecAXBRestoreArrayRead(Vec v, const struct axbVec_s **a);

PETSC_EXTERN PetscErrorCode VecAXBGetArrayWrite(Vec v, struct axbVec_s **a);
PETSC_EXTERN PetscErrorCode VecAXBRestoreArrayWrite(Vec v, struct axbVec_s **a);


#endif
