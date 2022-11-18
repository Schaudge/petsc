#include <petsc/private/localnetrpimpl.h>        /*I "petscnetrp.h"  I*/
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */
#include <petscdmnetwork.h>

/*
   Implementation of Exact Shallow Water Network Riemann Solver. Experimental WIP
*/

static PetscErrorCode NetRPNonlinearEval_ExactSWE(NetRP rp, DM network, PetscInt vert, Vec U, Vec Ustar, Vec F)
{
  PetscInt          e,dof=2,wavenum,numedges;
  const PetscScalar *ustar,*u; 
  PetscScalar       *f,ubar[2];
  const PetscInt *edges,*cone;
  PetscBool       edgein; 

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Ustar,&ustar));
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
 
  PetscCall(DMNetworkGetSupportingEdges(network,vert,&numedges,&edges));
  /* this interlaced ordering of algebraic conditions and physics, with the permutation of rows 
  0,1 ensures the jacobian has non-zero diagonal and direct solvers can easily be applied (as petsc 
  by default has no pivoting) */

  /* Algebraic Coupling Condition */
  f[1] = 0; 
  for (e=1; e<numedges; e++)
  {
    PetscCall(DMNetworkGetConnectedVertices(network,edges[e],&cone));
    edgein = (cone[1] == vert) ? PETSC_TRUE : PETSC_FALSE;
    /* algebraic coupling */
    f[e*dof] = ustar[dof*e] - ustar[dof*(e-1)];
    f[1] += (edgein) ? ustar[dof*e+1] : -ustar[dof*e+1];
    /* physics based coupling */
    wavenum = (edgein) ? 1 : 2;
    PetscCall(RiemannSolverEvalLaxCurve(rp->flux,u+dof*e,ustar[dof*e],wavenum,ubar));
    f[e*dof+1] = ustar[dof*e+1] - ubar[1]; 
  }
  /* do edge 0 */
  PetscCall(DMNetworkGetConnectedVertices(network,edges[0],&cone));
  edgein = (cone[1] == vert) ? PETSC_TRUE : PETSC_FALSE;
  f[1] += (edgein) ? ustar[1] : -ustar[1];
  wavenum = (edgein) ? 1 : 2;
  PetscCall(RiemannSolverEvalLaxCurve(rp->flux,u,ustar[0],wavenum,ubar));
  f[0] = ustar[1] - ubar[1];
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Ustar,&ustar));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal gravity;
  PetscReal parenth;
  PetscReal parentv;
} ShallowCtx;

static PetscErrorCode ExactSWELaxCurveJac(RiemannSolver rs,const PetscReal *u,PetscReal hbar,PetscInt wavenumber,PetscReal *DLax)
{
  PetscReal      g,h;
  ShallowCtx     *ctx;

  PetscFunctionBegin;
  PetscCall(RiemannSolverGetApplicationContext(rs,&ctx));
  g    = ctx->gravity;
  h    = u[0];
  /* switch between the 1-wave and 2-wave curves */
  switch (wavenumber)
  {
    case 1:
      DLax[0] = hbar<h ? -1.0*PetscSqrtScalar(g*hbar) : -(hbar+h)*PetscSqrtReal(g)/(2*PetscSqrtReal(2)*h*hbar*PetscSqrtScalar((hbar+h)/(hbar*h)));
      break;
    case 2:
      DLax[0] = hbar<h ?  1.0*PetscSqrtScalar(g*hbar): (hbar+h)*PetscSqrtReal(g)/(2*PetscSqrtReal(2)*h*hbar*PetscSqrtScalar((hbar+h)/(hbar*h)));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Shallow Water Lax Curves have only 2 waves (1,2), requested wave number: %i \n",wavenumber);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode NetRPNonlinearJac_ExactSWE(NetRP rp, DM network, PetscInt vert, Vec U, Vec Ustar, Mat DF)
{
  PetscInt          e,dof=2,wavenum,numedges;
  const PetscScalar *ustar,*u; 
  PetscScalar       ubar;
  const PetscInt *edges,*cone;
  PetscBool       edgein; 

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Ustar,&ustar));
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(DMNetworkGetSupportingEdges(network,vert,&numedges,&edges));

  for (e=1; e<numedges; e++)
  {
    PetscCall(DMNetworkGetConnectedVertices(network,edges[e],&cone));
    edgein = (cone[1] == vert) ? PETSC_TRUE : PETSC_FALSE;
    /* algebraic coupling */
    PetscCall(MatSetValue(DF,e*dof,e*dof,1,INSERT_VALUES));
    PetscCall(MatSetValue(DF,e*dof,(e-1)*dof,-1,INSERT_VALUES));

    PetscCall(MatSetValue(DF,1,dof*e+1,(edgein) ? 1 : -1,INSERT_VALUES));

    /* physics based coupling */
    wavenum = (edgein) ? 1 : 2;
    PetscCall(ExactSWELaxCurveJac(rp->flux,u+dof*e,ustar[dof*e],wavenum,&ubar));
    PetscCall(MatSetValue(DF,dof*e+1,dof*e,-ubar,INSERT_VALUES));
    PetscCall(MatSetValue(DF,dof*e+1,dof*e+1,1,INSERT_VALUES));
  }
  /* do edge 0 */
  PetscCall(DMNetworkGetConnectedVertices(network,edges[0],&cone));
  edgein = (cone[1] == vert) ? PETSC_TRUE : PETSC_FALSE;
  
  PetscCall(MatSetValue(DF,1,1,(edgein) ? 1 : -1,INSERT_VALUES));

  /* physics based coupling */
  wavenum = (edgein) ? 1 : 2;
  PetscCall(ExactSWELaxCurveJac(rp->flux,u,ustar[0],wavenum,&ubar));
  PetscCall(MatSetValue(DF,0,0,-ubar,INSERT_VALUES));
  PetscCall(MatSetValue(DF,0,1,1,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(DF,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(DF,MAT_FINAL_ASSEMBLY));

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Ustar,&ustar));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRPSetFromOptions_ExactSWE(PetscOptionItems *PetscOptionsObject,NetRP rp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRPView_ExactSWE(NetRP rp,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRPCreate_ExactSWE(NetRP rp)
{
  PetscFunctionBegin;
  rp->data = NULL;
  rp->ops->setfromoptions  = NRPSetFromOptions_ExactSWE;
  rp->ops->view            = NRPView_ExactSWE;
  rp->ops->NonlinearEval   = NetRPNonlinearEval_ExactSWE;
  rp->ops->NonlinearJac    = NetRPNonlinearJac_ExactSWE;
  rp->physicsgenerality    = Specific; /* the only specific thing here is the algebraic coupling and jacobian stuff (for now) */
  rp->solvetype            = Nonlinear; 
  
  rp->numfields = 2; 
  PetscFunctionReturn(0);
}

