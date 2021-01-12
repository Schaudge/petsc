#include "bf_2d_cells.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_P4estCreate             DMBF_2D_P4estCreate
#define DMBF_XD_P4estDestroy            DMBF_2D_P4estDestroy
#define DMBF_XD_GhostCreate             DMBF_2D_GhostCreate
#define DMBF_XD_GhostDestroy            DMBF_2D_GhostDestroy

#define DM_BF_XD_Topology               DM_BF_2D_Topology
#define DM_BF_XD_Cells                  DM_BF_2D_Cells
#define _p_DM_BF_XD_Cells               _p_DM_BF_2D_Cells

#define DMBF_XD_TopologyGetConnectivity DMBF_2D_TopologyGetConnectivity
#define DMBF_XD_CellsCreate             DMBF_2D_CellsCreate
#define DMBF_XD_CellsDestroy            DMBF_2D_CellsDestroy
#define DMBF_XD_CellsClone              DMBF_2D_CellsClone
#define DMBF_XD_GetSizes                DMBF_2D_GetSizes
#define DMBF_XD_GetLocalToGlobalIndices DMBF_2D_GetLocalToGlobalIndices
#define DMBF_XD_CellsGetP4est           DMBF_2D_CellsGetP4est
#define DMBF_XD_CellsGetGhost           DMBF_2D_CellsGetGhost

/* include generic functions */
#include "bf_xd_cells.c"

#endif /* defined(PETSC_HAVE_P4EST) */
