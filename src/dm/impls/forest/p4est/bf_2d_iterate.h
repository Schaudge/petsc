#if !defined(PETSCDMBF_2D_ITERATE_H)
#define PETSCDMBF_2D_ITERATE_H

#include <petscdmbf.h> /*I "petscdmbf.h" I*/

PetscErrorCode DMBF_2D_IterateSetUpCells(DM,DM_BF_Cell*,size_t,size_t,size_t);
PetscErrorCode DMBF_2D_IterateSetCellData(DM,DM_BF_Cell*,size_t,size_t,size_t,const PetscInt*,PetscInt,const PetscInt*,PetscInt,Vec*,Vec*);
PetscErrorCode DMBF_2D_IterateGetCellData(DM,DM_BF_Cell*,size_t,size_t,size_t,const PetscInt*,PetscInt,const PetscInt*,PetscInt,Vec*,Vec*);

PetscErrorCode DMBF_2D_IterateGhostExchange(DM,DM_BF_Cell*,size_t);

PetscErrorCode DMBF_2D_IterateOverCellsVectors(DM,DM_BF_Cell*,size_t,PetscErrorCode(*)(DM,DM_BF_Cell*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PetscErrorCode DMBF_2D_IterateOverFaces(DM,DM_BF_Cell*,size_t,PetscErrorCode(*)(DM,DM_BF_Face*,void*),void*);

#endif /* defined(PETSCDMBF_2D_ITERATE_H) */
