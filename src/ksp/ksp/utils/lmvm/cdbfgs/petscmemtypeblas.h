
#include <petscdevicetypes.h>
#include <petscblaslapack.h>

PETSC_EXTERN PetscErrorCode PetscMemtypeGEMM(PetscMemType, const char *, const char *, PetscBLASInt, PetscBLASInt, PetscBLASInt, PetscScalar, const PetscScalar *, PetscBLASInt, const PetscScalar *, PetscBLASInt, PetscScalar, PetscScalar *, PetscBLASInt);

PETSC_EXTERN PetscErrorCode PetscMemtypeTRSM(PetscMemType, const char *, const char *, const char *, const char *, PetscInt, PetscInt, PetscScalar, const PetscScalar *, PetscInt, PetscScalar *, PetscInt);
