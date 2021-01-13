#if !defined(PETSCDMBF_H)
#define PETSCDMBF_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMBFSetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFSetCellDataSize(DM,PetscInt*,PetscInt,PetscInt*,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFGetCellDataSize(DM,PetscInt**,PetscInt*,PetscInt**,PetscInt*);

PETSC_EXTERN PetscErrorCode DMBFGetInfo(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetLocalSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetGlobalSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFGetGhostSize(DM,PetscInt*);

typedef struct _p_DM_BF_Cell {
  /* corner coordinates */
  PetscReal         corner[8*3], volume, sidelength[3], dummy1[4];
  /* cell indices local to this processor rank and global across all ranks */
  PetscInt          indexLocal, indexGlobal;
  /* cell refinement level */
  PetscInt          level, dummy2;
  /* view of vector entries corresponding to this cell (not owned) */
  const PetscScalar **vecViewRead;
  PetscScalar       **vecViewReadWrite;
  /* data corresponding to this cell (owned) */
  const PetscScalar *dataRead;
  PetscScalar       *dataReadWrite;
} DM_BF_Cell;

//#define DMBFCellIsGhost(cell) (-1 == (cell)->indexGlobal) //TODO implement version with checking global indices

PETSC_EXTERN PetscErrorCode DMBFIterateOverCellsVectors(DM,PetscErrorCode(*)(DM,DM_BF_Cell*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverCells(DM,PetscErrorCode(*)(DM,DM_BF_Cell*,void*),void*);

typedef struct _p_DM_BF_Face {
  PetscInt    nCellsL, nCellsR;
  DM_BF_Cell  *cellL[4], *cellR[4];
} DM_BF_Face;

PETSC_EXTERN PetscErrorCode DMBFIterateOverFacesVectors(DM,PetscErrorCode(*)(DM,DM_BF_Face*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PETSC_EXTERN PetscErrorCode DMBFIterateOverFaces(DM,PetscErrorCode(*)(DM,DM_BF_Face*,void*),void*);

PETSC_EXTERN PetscErrorCode DMBFSetCellData(DM,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode DMBFGetCellData(DM,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode DMBFCommunicateGhostCells(DM);

PETSC_EXTERN PetscErrorCode DMBFGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMBFGetGhost(DM,void*);

PETSC_EXTERN PetscErrorCode DMBFVTKWriteAll(PetscObject,PetscViewer);

#endif /* defined(PETSCDMBF_H) */
