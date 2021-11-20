#include <petsc/private/netrsimpl.h>        /*I "petscnetrs.h"  I*/
#include <petscmat.h>
#include <petscvec.h>
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */

/*
    Implementation of exact nonlinear network solver for the SWE
*/

typedef struct {
    SNES snes; 
    Vec  x,b; 
} NRS_ExactSWE;

typedef struct {
    NetRS     netrs; 
    const EdgeDirection *dir; 
    const PetscReal *u; 
} ExactSWE_Wrapper;

/* Rework to seperate the Algebaric Coupling Conditions and the lax curve stuff */
static PetscErrorCode ExactSWE_LaxCurveFun(SNES snes,Vec x,Vec f, void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,dof=2,wavenum;
  ExactSWE_Wrapper  *wrapper= (ExactSWE_Wrapper*) ctx;
  const PetscScalar *ustar,*u = wrapper->u,ubar[2]; 
  PetscScalar       *F;
  NetRS             netrs = wrapper->netrs;

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecGetArray(f,&F);CHKERRQ(ierr);
 
  /* Algebraic Coupling Condition */
  F[n-2] = (wrapper->dir[netrs->numedges-1] == EDGEIN) ?  ustar[n-1] : -ustar[n-1];

  for (i=0; i<netrs->numedges-1; i++)
  {
      /* algebraic coupling */
    F[dof*i] = ustar[dof*i] - ustar[dof*(i+1)];
    F[n-2] += (wrapper->dir[i] == EDGEIN) ? ustar[dof*i+1] : -ustar[dof*i+1];
    /* physics based coupling */
    wavenum = (wrapper->dir[i] == EDGEIN) ? 1 : 2;
    ierr = RiemannSolverEvalLaxCurve(netrs->rs,u+dof*i,ustar[dof*i],wavenum,(PetscReal*)ubar);CHKERRQ(ierr);
    F[dof*i+1] = ustar[dof*i+1] - ubar[1]; 
  }
    wavenum = (wrapper->dir[netrs->numedges-1] == EDGEIN) ? 1 : 2;
    ierr = RiemannSolverEvalLaxCurve(netrs->rs,u+dof*(netrs->numedges-1),ustar[dof*(netrs->numedges-1)],wavenum,(PetscReal*)ubar);CHKERRQ(ierr);
    F[n-1] = ustar[n-1] - ubar[1];
  ierr = VecRestoreArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSEvaluate_ExactSWE(NetRS netrs,const PetscScalar *u,const EdgeDirection *dir,PetscScalar *flux,PetscReal *error)
{
  PetscErrorCode   ierr;
  ExactSWE_Wrapper wrapper;
  PetscInt         i,n,dof = netrs->numfields;
  PetscScalar      *x;
  NRS_ExactSWE     *exactswe = (NRS_ExactSWE*)netrs->data;
  void             *ctx;

  PetscFunctionBeginUser;
  wrapper.netrs = netrs; 
  wrapper.dir   = dir; 
  wrapper.u     = u;
  ierr = SNESSetFunction(exactswe->snes,exactswe->b,ExactSWE_LaxCurveFun,&wrapper);
  /* Set initial condition as the reconstructed h,v values*/
  ierr = VecGetArray(exactswe->x,&x);CHKERRQ(ierr);
  ierr = VecGetSize(exactswe->x,&n);CHKERRQ(ierr);
  for (i=0;i<netrs->numedges;i++) {
    x[i*dof]   = u[i*dof];
    x[i*dof+1] = u[i*dof+1];
  }
  ierr = VecRestoreArray(exactswe->x,&x);CHKERRQ(ierr);
  ierr = SNESSolve(exactswe->snes,NULL,exactswe->x);CHKERRQ(ierr);
  ierr = VecGetArray(exactswe->x,&x);CHKERRQ(ierr);

  if(netrs->estimate) {
    for (i=0;i<netrs->numedges;i++) {
      ierr = NetRSErrorEstimate(netrs,dir[i],u+dof*i,x+dof*i,&error[i]);CHKERRQ(ierr); /* compute error esimate on star state */
    }
  }
  ierr = NetRSGetApplicationContext(netrs,&ctx);CHKERRQ(ierr); 
  for (i=0;i<netrs->numedges*dof;i++) {
    flux[i] = x[i];
  }
  ierr = VecRestoreArray(exactswe->x,&x);CHKERRQ(ierr);
   PetscFunctionReturn(0);
}

static PetscReal ShallowRiemannExact_Left(const PetscScalar hl, const PetscScalar vl, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hl ? vl-2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hl)) : vl- (h-hl)*PetscSqrtScalar(g*(h+hl)/(2.0*h*hl));
}

static PetscReal ShallowRiemannExact_Right(const PetscScalar hr, const PetscScalar vr, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hr ? vr+2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hr)) : vr+(h-hr)*PetscSqrtScalar(g*(h+hr)/(2.0*h*hr));
}


static PetscErrorCode RiemannInvariant_Couple_Shallow(SNES snes,Vec x,Vec f, void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i,n,dof=2;
  ExactSWE_Wrapper  *wrapper= (ExactSWE_Wrapper*) ctx;
  const PetscScalar *ustar,*u = wrapper->u; 
  PetscScalar       *F;
  NetRS             netrs = wrapper->netrs;

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecGetArray(f,&F);CHKERRQ(ierr);

  F[n-2] = (wrapper->dir[netrs->numedges-1] == EDGEIN) ?  ustar[n-1] : -ustar[n-1];

  for (i=0; i<netrs->numedges-1; i++)
  {
    F[dof*i] = ustar[dof*i] - ustar[dof*(i+1)];
    F[n-2] += (wrapper->dir[i] == EDGEIN) ? ustar[dof*i+1] : -ustar[dof*i+1];
    /* output for this edges vstar eqn */
    if (wrapper->dir[i] == EDGEIN){
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Left(u[dof*i],u[dof*i+1]/u[dof*i],ustar[dof*i]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Right(u[dof*i],u[dof*i+1]/u[dof*i],ustar[dof*i]);
    }
  }

   if (wrapper->dir[netrs->numedges-1] == EDGEIN){
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Left(u[n-2],u[n-1]/u[n-2],ustar[n-2]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Right(u[n-2],u[n-1]/u[n-2],ustar[n-2]);
    }

  ierr = VecRestoreArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsVertexFlux_Shallow_Full(NetRS netrs,const PetscScalar *u,const EdgeDirection *dir,PetscScalar *flux,PetscReal *error)
{
  PetscErrorCode  ierr;
 NRS_ExactSWE     *exactswe = (NRS_ExactSWE*)netrs->data;
  PetscInt        i,n,dof = netrs->numfields;
  PetscScalar     *x;
  ExactSWE_Wrapper wrapper;
   void             *ctx;


  PetscFunctionBeginUser;
  wrapper.netrs = netrs; 
  wrapper.dir   = dir; 
  wrapper.u     = u;
  ierr = SNESSetFunction(exactswe->snes,exactswe->b,RiemannInvariant_Couple_Shallow,&wrapper);
  /* Set initial condition as the reconstructed h,v values*/
  ierr = VecGetArray(exactswe->x,&x);CHKERRQ(ierr);
  ierr = VecGetSize(exactswe->x,&n);CHKERRQ(ierr);

  for (i=0;i<netrs->numedges;i++) {
    x[i*dof]   = u[i*dof];
    x[i*dof+1] = u[i*dof+1]/u[i*dof];
  }
  ierr = VecRestoreArray(exactswe->x,&x);CHKERRQ(ierr);
  ierr = SNESSolve(exactswe->snes,NULL,exactswe->x);CHKERRQ(ierr);
  ierr = VecGetArray(exactswe->x,&x);CHKERRQ(ierr);

  
  /* Compute flux from the star state */
  ierr = NetRSGetApplicationContext(netrs,&ctx);CHKERRQ(ierr); 
  for (i=0;i<netrs->numedges;i++) {
    flux[i*dof] = x[i*dof];
  }
  /* Compute the Flux from the computed star values */
  for (i=0;i<netrs->numedges;i++) {
    flux[i*dof+1] = x[i*dof+1]*x[i*dof];
  }
  if(netrs->estimate) {
    for (i=0;i<netrs->numedges;i++) {
      ierr = NetRSErrorEstimate(netrs,dir[i],u+dof*i,flux+dof*i,&error[i]);CHKERRQ(ierr); /* compute error esimate on star state */
    }
  }
  ierr = VecRestoreArray(exactswe->x,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetUp_ExactSWE(NetRS rs)
{
  NRS_ExactSWE   *exactswe = (NRS_ExactSWE*)rs->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
    ierr = VecCreateSeq(MPI_COMM_SELF,rs->numfields*rs->numedges,&exactswe->x);CHKERRQ(ierr); /* Specific to the SWE with equal height coupling. To be adjusted */
    ierr = VecDuplicate(exactswe->x,&exactswe->b);CHKERRQ(ierr);
    ierr = SNESCreate(PETSC_COMM_SELF,&exactswe->snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(exactswe->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSReset_ExactSWE(NetRS rs)
{
  NRS_ExactSWE    *exactswe = (NRS_ExactSWE*)rs->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&exactswe->x);CHKERRQ(ierr);
  ierr = VecDestroy(&exactswe->b);CHKERRQ(ierr);
  ierr = SNESDestroy(&exactswe->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSDestroy_ExactSWE(NetRS rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NRSReset_ExactSWE(rs);CHKERRQ(ierr);
  ierr = PetscFree(rs->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetFromOptions_ExactSWE(PetscOptionItems *PetscOptionsObject,NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSView_ExactSWE(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NRSCreate_ExactSWEStar(NetRS rs)
{
  NRS_ExactSWE   *exactswe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rs,&exactswe);CHKERRQ(ierr);
  rs->data = (void*)exactswe;
  if(rs->numfields>-1) {
    if(rs->numfields != 2) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The Exact SWE solver requires numfields to be 2. (and should be solving the SWE)");
    } 
  }
  rs->ops->setup           = NRSSetUp_ExactSWE;
  rs->ops->reset           = NRSReset_ExactSWE;
  rs->ops->destroy         = NRSDestroy_ExactSWE;
  rs->ops->setfromoptions  = NRSSetFromOptions_ExactSWE;
  rs->ops->view            = NRSView_ExactSWE;
  rs->ops->evaluate        = NRSEvaluate_ExactSWE;
  PetscFunctionReturn(0);
}
