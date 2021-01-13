#if !defined(PETSCDMBF_3D_AMR_H)
#define PETSCDMBF_3D_AMR_H

#if defined(PETSCDMBF_2D_AMR_H)
#error "The include files bf_2d_amr.h and bf_3d_amr.h cannot be combined"
#endif

#include "bf_xd.h"

PetscErrorCode DMBF_3D_AmrCoarsenUniformly(p4est_t*,PetscInt);
PetscErrorCode DMBF_3D_AmrRefineUniformly(p4est_t*,PetscInt);

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_AmrCoarsenUniformly DMBF_3D_AmrCoarsenUniformly
#define DMBF_XD_AmrRefineUniformly  DMBF_3D_AmrRefineUniformly

#endif /* defined(PETSCDMBF_3D_AMR_H) */
