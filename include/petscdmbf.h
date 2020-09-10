#if !defined(PETSCDMBF_H)
#define PETSCDMBF_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMBFSetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetBlockSize(DM,PetscInt*);

PETSC_EXTERN PetscErrorCode DMBFCoarsenInPlace(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFRefineInPlace(DM,PetscInt);

typedef struct _p_DM_BF_CellData {
  PetscInt  index_local;
  PetscInt  level;
  PetscReal corner[3];
  PetscReal length[3];
} DM_BF_CellData;
PETSC_EXTERN PetscErrorCode DMBFIterateOverCells(DM,PetscErrorCode(*)(DM_BF_CellData*,void*),void*);

PETSC_EXTERN PetscErrorCode DMBFGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMBFGetGhost(DM,void*);

#endif /* defined(PETSCDMBF_H) */
