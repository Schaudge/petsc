#include "bf_3d_iterate.h"

#if defined(PETSC_HAVE_P4EST)
//TODO activate p8est and fix bugs
//#include <p4est_to_p8est.h> /* convert to p8est for 3D domains */

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_IterateSetUpCells       DMBF_3D_IterateSetUpCells
#define DMBF_XD_IterateSetCellData      DMBF_3D_IterateSetCellData
#define DMBF_XD_IterateGetCellData      DMBF_3D_IterateGetCellData

#define DMBF_XD_IterateOverCellsVectors DMBF_3D_IterateOverCellsVectors
#define DMBF_XD_IterateOverFaces        DMBF_3D_IterateOverFaces

/* include generic functions */
#include "bf_xd_iterate.c"

#endif /* defined(PETSC_HAVE_P4EST) */

