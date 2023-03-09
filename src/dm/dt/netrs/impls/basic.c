#include <petsc/private/netrsimpl.h> /*I "petscnetrp.h"  I*/

static PetscErrorCode NetRSSetFromOptions_Blank(PetscOptionItems *PetscOptionsObject, NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRSView_Blank(NetRS rs, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* fill these */
static PetscErrorCode NetRSSetup_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode NetRSReset_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRSSetupVecSpace_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRSResetVecSpace_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NetRSSolve_Blank(NetRS rs, Vec U, Vec Flux)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRSCreate_Blank(NetRS rs)
{
  PetscFunctionBegin;
  rs->data                = NULL;
  rs->ops->setfromoptions = NetRSSetFromOptions_Blank;
  rs->ops->view           = NetRSView_Blank;
  rs->ops->reset          = NetRSReset_Blank;
  rs->ops->setupvecspace  = NetRSSetupVecSpace_Blank;
  rs->ops->setup          = NetRSSetup_Blank;
  rs->ops->setupvecspace  = NetRSSetupVecSpace_Blank;
  rs->ops->resetvecspace  = NetRSResetVecSpace_Blank;
  rs->ops->solve          = NetRSSolve_Blank;
  PetscFunctionReturn(PETSC_SUCCESS);
}
