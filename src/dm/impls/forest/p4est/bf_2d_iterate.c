#include "bf_2d_iterate.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_IterateSetUpCells       DMBF_2D_IterateSetUpCells
#define DMBF_XD_IterateSetUpP4estCells  DMBF_2D_IterateSetUpP4estCells
#define DMBF_XD_IterateCopyP4estCells   DMBF_2D_IterateCopyP4estCells
#define DMBF_XD_IterateSetCellData      DMBF_2D_IterateSetCellData
#define DMBF_XD_IterateGetCellData      DMBF_2D_IterateGetCellData

#define DMBF_XD_IterateGhostExchange    DMBF_2D_IterateGhostExchange

#define DMBF_XD_IterateOverCellsVectors DMBF_2D_IterateOverCellsVectors
#define DMBF_XD_IterateOverFaces        DMBF_2D_IterateOverFaces

/* include generic functions */
#include "bf_xd_iterate.c"

#endif /* defined(PETSC_HAVE_P4EST) */
