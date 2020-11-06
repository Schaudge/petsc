#include "bf_3d_iterate.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_IterateOverCellsVectors   DMBF_3D_IterateOverCellsVectors

/* include generic functions */
#include "bf_xd_iterate.c"

#endif /* defined(PETSC_HAVE_P4EST) */

