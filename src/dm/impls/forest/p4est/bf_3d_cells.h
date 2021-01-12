#if !defined(PETSCDMBF_CELLS_3D_H)
#define PETSCDMBF_CELLS_3D_H

#include "bf_3d_topology.h"

typedef struct _p_DM_BF_3D_Cells DM_BF_3D_Cells;

PetscErrorCode DMBF_3D_CellsCreate(DM,DM_BF_3D_Topology*,DM_BF_3D_Cells**);
PetscErrorCode DMBF_3D_CellsDestroy(DM,DM_BF_3D_Cells*);
PetscErrorCode DMBF_3D_CellsClone(DM_BF_3D_Cells*,DM_BF_3D_Cells**,DM);

PetscErrorCode DMBF_3D_GetSizes(DM,DM_BF_3D_Cells*,PetscInt*,PetscInt*,PetscInt*);
PetscErrorCode DMBF_3D_GetLocalToGlobalIndices(DM,DM_BF_3D_Cells*,PetscInt*,PetscInt*toIdx);

PetscErrorCode DMBF_3D_CellsGetP4est(DM_BF_3D_Cells*,void*);
PetscErrorCode DMBF_3D_CellsGetGhost(DM_BF_3D_Cells*,void*);

#endif /* defined(PETSCDMBF_CELLS_3D_H) */
