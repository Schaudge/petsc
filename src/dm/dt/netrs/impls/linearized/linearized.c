#include <petsc/private/netrsimpl.h>        /*I "petscnetrs.h"  I*/
#include <petscmat.h>
#include <petscvec.h>
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */

/*
   Implementation of linearized Network Riemann Solver. Experimental WIP
*/

typedef struct {
    Mat mat; 
    KSP ksp;
    Vec x,b; 
    PetscReal *ustar; /* to be removed */ 
} NRS_Linear;

/* The linearized Solver Code */
static PetscReal ShallowRiemannEig_Right(const PetscScalar hr, const PetscScalar vr)
{
  const PetscScalar g = 9.81; /* replace with general g for swe ctx */
  return vr + PetscSqrtScalar(g*hr);
}

static PetscReal ShallowRiemannEig_Left(const PetscScalar hl, const PetscScalar vl)
{
  const PetscScalar g = 9.81;
  return vl - PetscSqrtScalar(g*hl);
}

PETSC_INTERN PetscErrorCode NRSEvaluate_Linear(NetRS rs, const PetscReal *u, const PetscBool *dir,PetscReal *flux) 
{
  NRS_Linear     *linear = (NRS_Linear*)rs->data;
  void           *ctx;
  PetscErrorCode ierr;
  PetscInt       i,dof = rs->numfields;
  PetscScalar    *x,*b,eig,h,v,hv,error; 

  PetscFunctionBeginUser;
  ierr = NetRSGetApplicationContext(rs,&ctx);CHKERRQ(ierr); 
  ierr = VecGetArray(linear->b,&b);CHKERRQ(ierr); /*rhs*/
  /* Build the system matrix and rhs vector */
  for (i=1;i<rs->numedges;i++) {
    h  = u[i*dof];
    hv = u[i*dof+1];
    v  = hv/h; 
    eig = dir[i] == EDGEIN ? 
      ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v); /* replace with RiemannSolver Calls */
    ierr = MatSetValue(linear->mat,i+1,i+1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
    ierr = MatSetValue(linear->mat,i+1,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
    ierr = MatSetValue(linear->mat,1,i+1,dir[i] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
    b[i+1] = eig*h-hv; /* riemann invariant at the boundary */
  }
  /* Form the matrix in this reordered form to have nonzeros along the diagonal */
  h  = u[0];
  hv = u[1];
  v  = hv/h; 
  eig = dir[0] == EDGEIN ? 
    ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v);
  ierr = MatSetValue(linear->mat,0,1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
  ierr = MatSetValue(linear->mat,0,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
  ierr = MatSetValue(linear->mat,1,1,dir[0] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
  b[0] = eig*h-hv; /* riemann invariant at the boundary */
  b[1] = 0.0;   
  ierr = VecRestoreArray(linear->b,&b);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(linear->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(linear->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* TODO: add nonzero initial guess data here later */

  ierr = KSPSolve(linear->ksp,linear->b,linear->x);CHKERRQ(ierr);
  ierr = VecGetArray(linear->x,&x);CHKERRQ(ierr);
  /* Compute the ustar state from the linear solve (in this case expand the single h to full ustar) */

/* VERY TEMPORARY TO BE REDONE# */
  for (i=0;i<rs->numedges;i++) {
    linear->ustar[i*dof+1] = x[i+1]; /*hv term */
    linear->ustar[i*dof]   = x[0]; /* copy h value to correct spot */
  }
  if(rs->estimate) {
  ierr = NetRSErrorEstimate(rs,u,linear->ustar,&error);CHKERRQ(ierr); /* compute error esimate on star state */
  }
  /* Compute the Flux from the computed star values
     REDO WITH PROPER FLux class later */
  
  for (i=0;i<rs->numedges;i++) {
    rs->rs->fluxfun(ctx,linear->ustar+i*dof,flux+i*dof);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(linear->x,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetUp_Linear(NetRS rs)
{
  NRS_Linear *linear = (NRS_Linear*)rs->data;
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
    ierr = VecCreateSeq(MPI_COMM_SELF,rs->numedges+1,&linear->x);CHKERRQ(ierr); /* Specific to the SWE with equal height coupling. To be adjusted */
    ierr = VecDuplicate(linear->x,&linear->b);CHKERRQ(ierr);
    ierr = MatCreate(MPI_COMM_SELF,&linear->mat);CHKERRQ(ierr);
    ierr = MatSetSizes(linear->mat,PETSC_DECIDE,PETSC_DECIDE,rs->numedges+1,rs->numedges+1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(linear->mat);CHKERRQ(ierr); /* Maybe change default type option to dense. Experiment */
    ierr = MatSetUp(linear->mat);CHKERRQ(ierr); 
      /* Now set up the linear solver. */ 
    ierr = KSPCreate(PETSC_COMM_SELF,&linear->ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(linear->ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetType(linear->ksp,KSPPREONLY);CHKERRQ(ierr); /* Set to direct solver only */
    ierr = KSPSetOperators(linear->ksp,linear->mat,linear->mat);CHKERRQ(ierr);

    /* Set Default KSP Options Here */

    /* temp */
    ierr = PetscMalloc1(rs->numfields*rs->numedges,&linear->ustar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode NRSReset_Linear(NetRS rs)
{
  NRS_Linear     *linear = (NRS_Linear*)rs->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&linear->x);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->b);CHKERRQ(ierr);
  ierr = KSPDestroy(&linear->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&linear->mat);CHKERRQ(ierr);
  ierr = PetscFree(linear->ustar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSDestroy_Linear(NetRS rs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NRSReset_Linear(rs);CHKERRQ(ierr);
  ierr = PetscFree(rs->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject,NetRS rs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRSView_Linear(NetRS rs,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NRSCreate_Linear(NetRS rs)
{
  NRS_Linear     *linear;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rs,&linear);CHKERRQ(ierr);
  rs->data = (void*)linear;
  if(rs->numfields>-1) {
    if(rs->numfields != 2) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The linearized solver requires numfields to be 2.");
    } 
  }
  rs->ops->setup           = NRSSetUp_Linear;
  rs->ops->reset           = NRSReset_Linear;
  rs->ops->destroy         = NRSDestroy_Linear;
  rs->ops->setfromoptions  = NRSSetFromOptions_Linear;
  rs->ops->view            = NRSView_Linear;
  rs->ops->evaluate        = NRSEvaluate_Linear;
  PetscFunctionReturn(0);
}

