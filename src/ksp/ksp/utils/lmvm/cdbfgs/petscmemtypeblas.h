
#include <petscdevicetypes.h>
#include <petscblaslapack.h>

PETSC_EXTERN PetscErrorCode PetscMemTypeGEMV(PetscMemType, const char *, PetscInt, PetscInt, PetscScalar, const PetscScalar *, PetscInt, const PetscScalar *, PetscInt, PetscScalar, PetscScalar *, PetscInt);

PETSC_EXTERN PetscErrorCode PetscMemTypeGEMM(PetscMemType, const char *, const char *, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar *, PetscInt, const PetscScalar *, PetscInt, PetscScalar, PetscScalar *, PetscInt);

PETSC_EXTERN PetscErrorCode PetscMemTypeTRSM(PetscMemType, const char *, const char *, const char *, const char *, PetscInt, PetscInt, PetscScalar, const PetscScalar *, PetscInt, PetscScalar *, PetscInt);

PETSC_EXTERN PetscErrorCode PetscMemTypeTRSV(PetscMemType, const char *, const char *, const char *, PetscInt, const PetscScalar *, PetscInt, PetscScalar *, PetscInt);

PETSC_EXTERN PetscErrorCode PetscMemTypeAXPY(PetscMemType, PetscInt, PetscScalar, const PetscScalar *, PetscInt, PetscScalar *, PetscInt);

