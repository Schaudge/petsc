#if !defined(PETSCDMBF_2D_ITERATE_H)
#define PETSCDMBF_2D_ITERATE_H

#include <petscdmbf.h> /*I "petscdmbf.h" I*/

PetscErrorCode DMBF_2D_IterateOverCellsVectors(DM,char*,size_t,PetscErrorCode(*)(DM,DM_BF_Cell*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PetscErrorCode DMBF_2D_IterateOverFaces(DM,char*,size_t,PetscErrorCode(*)(DM,DM_BF_Face*,void*),void*);

#endif /* defined(PETSCDMBF_2D_ITERATE_H) */
