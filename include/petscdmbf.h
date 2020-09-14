#if !defined(PETSCDMBF_H)
#define PETSCDMBF_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMBFSetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetBlockSize(DM,PetscInt*);

PETSC_EXTERN PetscErrorCode DMBFCoarsenInPlace(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFRefineInPlace(DM,PetscInt);

typedef struct _p_DM_BF_CellData {
  /* cell indices local to this processor rank and global across all ranks*/
  PetscInt    index_local, index_global;
  /* cell refinement level, corner coordinates, and side lengths */
  PetscInt    level;
  PetscReal   corner[3], length[3];
  /* data of vectors corresponding to this cell */
  PetscScalar *vecdata_in, *vecdata_out;
} DM_BF_CellData;

PETSC_EXTERN PetscErrorCode DMBFIterateOverCellsVectors(DM,PetscErrorCode(*)(DM_BF_CellData*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverCells(DM,PetscErrorCode(*)(DM_BF_CellData*,void*),void*);

typedef struct _p_DM_BF_FaceData {
  PetscInt       nCellsL, nCellsR;
  DM_BF_CellData *cellDataL[4], *cellDataR[4];
} DM_BF_FaceData;

PETSC_EXTERN PetscErrorCode DMBFIterateOverFacesVectors(DM,PetscErrorCode(*)(DM_BF_FaceData*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverFaces(DM,PetscErrorCode(*)(DM_BF_FaceData*,void*),void*);

PETSC_EXTERN PetscErrorCode DMBFGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMBFGetGhost(DM,void*);

#endif /* defined(PETSCDMBF_H) */
