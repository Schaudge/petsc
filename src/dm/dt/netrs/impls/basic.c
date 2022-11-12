#include <petsc/private/netrsimpl.h>       /*I "petscnetrp.h"  I*/

static PetscErrorCode NetRSSetFromOptions_Blank(PetscOptionItems *PetscOptionsObject,NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRSView_Blank(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* fill these */
static PetscErrorCode NetRSSetup_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
static PetscErrorCode NetRSReset_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRSSetupVecSpace_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRSResetVecSpace_Blank(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRSSolve_Blank(NetRS rs,Vec U,Vec Flux)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRSCreate_Blank(NetRS rs)
{
  PetscFunctionBegin;
  rs->data = NULL;
  rs->ops->setfromoptions  = NetRSSetFromOptions_Blank;
  rs->ops->view            = NetRSView_Blank;
  rs->ops->reset           = NetRSReset_Blank; 
  rs->ops->setupvecspace   = NetRSSetupVecSpace_Blank; 
  rs->ops->setup           = NetRSSetup_Blank; 
  rs->ops->setupvecspace = NetRSSetupVecSpace_Blank; 
  rs->ops->resetvecspace = NetRSResetVecSpace_Blank; 
  rs->ops->solve         = NetRSSolve_Blank; 
  PetscFunctionReturn(0);
}
