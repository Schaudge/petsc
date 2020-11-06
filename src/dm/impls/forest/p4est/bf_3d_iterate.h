#if !defined(PETSCDMBF_3D_ITERATE_H)
#define PETSCDMBF_3D_ITERATE_H

#include <petscdmbf.h> /*I "petscdmbf.h" I*/

PetscErrorCode DMBF_3D_IterateOverCellsVectors(DM,char*,size_t,PetscErrorCode(*)(DM,DM_BF_Cell*,void*),void*,Vec*,PetscInt,Vec*,PetscInt);
PetscErrorCode DMBF_3D_IterateOverFaces(DM,char*,size_t,PetscErrorCode(*)(DM,DM_BF_Face*,void*),void*);

#endif /* defined(PETSCDMBF_3D_ITERATE_H) */
