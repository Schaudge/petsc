#include <petsc/private/riemannsolverimpl.h>        /*I "petscriemannsolver.h"  I*/

/*
    Implementation of the (local) lax friedrich riemann solver. 

    TODO: Add references and details to lax friedrich riemann solver here 
*/

typedef struct {
  PetscReal *flux_eval; /* work array for flux evaluations*/
} RS_Lax;

/* The actual lax friedrich riemann solver code */
PETSC_INTERN PetscErrorCode RiemannSolverEvaluate_Lax(RiemannSolver rs, const PetscReal *uL, const PetscReal *uR) 
{
  RS_Lax         *lax = (RS_Lax*)rs->data;
  void           *ctx;
  PetscErrorCode ierr;
  PetscReal      *flux_eval = lax->flux_eval;
  PetscInt       i;
 
  PetscFunctionBeginUser;
  ierr = RiemannSolverGetApplicationContext(rs,&ctx);CHKERRQ(ierr); 
  /* Compute the maximum wave speed for the riemann problem */
  ierr = RiemannSolverComputeMaxSpeed(rs,uL,uR,&rs->maxspeed);CHKERRQ(ierr);
  ierr = PetscArrayzero(rs->flux_wrk,rs->numfields);CHKERRQ(ierr);
  /* left portion */ 
  rs->fluxfun(ctx,uL,flux_eval);
  for(i=0;i<rs->numfields; i++) {
      rs->flux_wrk[i] += 0.5*(flux_eval[i] + rs->maxspeed*uL[i]);
  }
  /* right portion */ 
  rs->fluxfun(ctx,uR,flux_eval);
  for(i=0;i<rs->numfields; i++) {
      rs->flux_wrk[i] += 0.5*(flux_eval[i] - rs->maxspeed*uR[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannSolverSetUp_Lax(RiemannSolver rs)
{
  RS_Lax         *lax = (RS_Lax*)rs->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(rs->numfields,&lax->flux_eval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannSolverReset_Lax(RiemannSolver rs)
{
  RS_Lax         *lax = (RS_Lax*)rs->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(lax->flux_eval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannSolverDestroy_Lax(RiemannSolver rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = RiemannSolverReset_Lax(rs);CHKERRQ(ierr);
  ierr = PetscFree(rs->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannSolverSetFromOptions_Lax(PetscOptionItems *PetscOptionsObject,RiemannSolver rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode RiemannSolverView_Lax(RiemannSolver rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

/*MC
    RIEMANNLAXFRIEDRICH - Lax-Friedrich Riemann Solver. 

  Level: beginner

.seealso:  RiemannSolverCreate(), RiemannSolver, RiemannSolverSetType()

M*/
PETSC_EXTERN PetscErrorCode RiemannSolverCreate_Lax(RiemannSolver rs)
{
  RS_Lax         *lax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rs,&lax);CHKERRQ(ierr);
  rs->data = (void*)lax;

  rs->ops->setup           = RiemannSolverSetUp_Lax;
  rs->ops->reset           = RiemannSolverReset_Lax;
  rs->ops->destroy         = RiemannSolverDestroy_Lax;
  rs->ops->setfromoptions  = RiemannSolverSetFromOptions_Lax;
  rs->ops->view            = RiemannSolverView_Lax;
  rs->ops->evaluate        = RiemannSolverEvaluate_Lax;
  PetscFunctionReturn(0);
}

