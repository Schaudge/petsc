#include <petsc/private/netrsimpl.h>        /*I "petscnetrs.h"  I*/
#include <petscmat.h>
#include <petscvec.h>
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */

/*
   Allows for a standard riemann solver as a network riemann solver in the 
   case when there are only two edges attached to vertex, where you can think of 
   the netrs as just standard rs. 
*/

static PetscErrorCode NRSEvaluate_RS(NetRS rs, const PetscReal *u, const EdgeDirection *dir,PetscReal *flux,PetscReal *error) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *fluxrs;

  PetscFunctionBeginUser;
  if(rs->numedges != 2) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The Riemann Solver NETRS requires exactly two edges");}
  if (dir[0] == EDGEIN && dir[1] == EDGEOUT ) {
    ierr = RiemannSolverEvaluate(rs->rs,u,u+rs->numfields,&fluxrs,NULL);CHKERRQ(ierr);
      /* adjust the computed flux to align with mesh discretization given by dir */
    for(i=0; i<rs->numfields; i++) {
      flux[i] = fluxrs[i];
    }
    for(i=0; i<rs->numfields; i++) {
      flux[i+rs->numfields] = fluxrs[i];
    }
  } else  if (dir[0] == EDGEOUT && dir[1] == EDGEIN ) {
    ierr = RiemannSolverEvaluate(rs->rs,u+rs->numfields,u,&fluxrs,NULL);CHKERRQ(ierr);
      /* adjust the computed flux to align with mesh discretization given by dir */
    for(i=0; i<rs->numfields; i++) {
      flux[i] = fluxrs[i];
    }
    for(i=0; i<rs->numfields; i++) {
      flux[i+rs->numfields] = fluxrs[i];
    }
  } else  if (dir[0] == EDGEIN && dir[1] == EDGEIN ) {
    ierr = RiemannSolverEvaluate(rs->rs,u,u+rs->numfields,&fluxrs,NULL);CHKERRQ(ierr);
      /* adjust the computed flux to align with mesh discretization given by dir */
    for(i=0; i<rs->numfields; i++) {
      flux[i] = fluxrs[i];
    }
    ierr = RiemannSolverEvaluate(rs->rs,u+rs->numfields,u,&fluxrs,NULL);CHKERRQ(ierr);
    for(i=0; i<rs->numfields; i++) {
      flux[i+rs->numfields] = fluxrs[i];
    }
  } else  if (dir[0] == EDGEOUT && dir[1] == EDGEOUT ) {
     ierr = RiemannSolverEvaluate(rs->rs,u+rs->numfields,u,&fluxrs,NULL);CHKERRQ(ierr);
      /* adjust the computed flux to align with mesh discretization given by dir */
    for(i=0; i<rs->numfields; i++) {
      flux[i] = fluxrs[i];
    }
    ierr = RiemannSolverEvaluate(rs->rs,u,u+rs->numfields,&fluxrs,NULL);CHKERRQ(ierr);
    for(i=0; i<rs->numfields; i++) {
      flux[i+rs->numfields] = fluxrs[i];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetUp_RS(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSReset_RS(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSDestroy_RS(NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetFromOptions_RS(PetscOptionItems *PetscOptionsObject,NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSView_RS(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NRSCreate_RS(NetRS rs)
{
  PetscFunctionBegin;
  rs->ops->setup           = NRSSetUp_RS;
  rs->ops->reset           = NRSReset_RS;
  rs->ops->destroy         = NRSDestroy_RS;
  rs->ops->setfromoptions  = NRSSetFromOptions_RS;
  rs->ops->view            = NRSView_RS;
  rs->ops->evaluate        = NRSEvaluate_RS;
  PetscFunctionReturn(0);
}

