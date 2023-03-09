#include <petsc/private/localnetrpimpl.h> /*I "petscnetrp.h"  I*/

static PetscErrorCode NRPSetFromOptions_Blank(PetscOptionItems *PetscOptionsObject, NetRP rp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NRPView_Blank(NetRP rp, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRPCreate_Blank(NetRP rp)
{
  PetscFunctionBegin;
  rp->data                = NULL;
  rp->ops->setfromoptions = NRPSetFromOptions_Blank;
  rp->ops->view           = NRPView_Blank;
  PetscFunctionReturn(PETSC_SUCCESS);
}
