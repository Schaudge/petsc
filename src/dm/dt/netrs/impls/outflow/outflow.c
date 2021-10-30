#include <petsc/private/netrsimpl.h>        /*I "petscnetrs.h"  I*/
#include <petscmat.h>
#include <petscvec.h>
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */

/*
    Heuristic Outflow Boundary Condtion that I use. Should be removed as it has no 
    serious justification for its existance other than to get a code to work. 

    Replace with a boundary condition class ?
*/

static PetscErrorCode NRSEvaluate_Outflow(NetRS rs, const PetscReal *u, const PetscBool *dir,PetscReal *flux) 
{
  void           *ctx;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *fluxrs;

  PetscFunctionBeginUser;
  if(rs->numedges != 1) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The Outflow NETRS requires exactly one edge");}
  ierr = RiemannSolverEvaluate(rs->rs,u,u,&fluxrs,NULL);CHKERRQ(ierr);
  for (i=0; i<rs->numfields; i++) {
    flux[i] = fluxrs[i]; 
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetUp_Outflow(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSReset_Outflow(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSDestroy_Outflow(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetFromOptions_Outflow(PetscOptionItems *PetscOptionsObject,NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSView_Outflow(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NRSCreate_Outflow(NetRS rs)
{
  PetscFunctionBegin;
  rs->ops->setup           = NRSSetUp_Outflow;
  rs->ops->reset           = NRSReset_Outflow;
  rs->ops->destroy         = NRSDestroy_Outflow;
  rs->ops->setfromoptions  = NRSSetFromOptions_Outflow;
  rs->ops->view            = NRSView_Outflow;
  rs->ops->evaluate        = NRSEvaluate_Outflow;
  PetscFunctionReturn(0);
}

