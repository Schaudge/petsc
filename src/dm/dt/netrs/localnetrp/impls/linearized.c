#include <petsc/private/localnetrpimpl.h>        /*I "petscnetrp.h"  I*/
#include <petsc/private/riemannsolverimpl.h>    /* should not be here */
#include <petscdmnetwork.h>

/*
   Implementation of linearized Network Riemann Solver. Experimental WIP
*/

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

static PetscErrorCode NetRPCreateLinearStar_Linearized(NetRP rp, DM network, PetscInt vert, Vec U, Vec Rhs, Mat A)
{
  PetscScalar h,eig,hv,v;
  PetscScalar *rhs;
  const PetscScalar *u; 
  PetscInt    numedges,i,numfields,index_hv,index_h; 
  const PetscInt *edges,*cone; 
  void           *ctx;

  PetscFunctionBegin; 
  PetscCall(NetRPGetApplicationContext(rp,&ctx));
  PetscCall(VecGetArrayRead(U,&u)); 
  PetscCall(VecGetArray(Rhs,&rhs));
  PetscCall(DMNetworkGetSupportingEdges(network,vert,&numedges,&edges)); 
  PetscCall(NetRPGetNumFields(rp,&numfields));
/* Build the system matrix and rhs vector */
  for (i=1; i<numedges; i++) {
    index_h = i*numfields; 
    index_hv = index_h+1; 
    h  = u[index_h];
    hv = u[index_hv];
    v  = hv/h; 
    PetscCall(DMNetworkGetConnectedVertices(network,edges[i],&cone)); 
    eig = cone[1] == vert ? 
      ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v); /* replace with RiemannSolver Calls */
    PetscCall(MatSetValue(A,index_hv,index_hv,-1 ,INSERT_VALUES)); /* hv* term */
    PetscCall(MatSetValue(A,index_hv,index_h,eig,INSERT_VALUES));      /* h* term */
    PetscCall(MatSetValue(A,1,index_hv,cone[1] == v ? 1:-1,INSERT_VALUES));  /* flux balance bdry condition */
    rhs[index_hv] = eig*h-hv; /* riemann invariant at the boundary */
  /* equal height algebraic coupling condition */
    index_h = i*numfields; 
    PetscCall(MatSetValue(A,index_h,index_h-numfields,-1, INSERT_VALUES)); 
    PetscCall(MatSetValue(A,index_h,index_h,1,INSERT_VALUES)); 
    rhs[index_h] = 0.0; 
  }

  /* first row requires a change as to ensure non-zero diagonal, a permutation of the first
     two rows in the "standard ordering is needed */
    h  = u[0];
    hv = u[1];
    v  = hv/h; 
    PetscCall(DMNetworkGetConnectedVertices(network,edges[i],&cone)); 
    eig = cone[1] == vert ? 
      ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v); /* replace with RiemannSolver Calls */
    PetscCall(MatSetValue(A,0,1,-1 ,INSERT_VALUES)); /* hv* term */
    PetscCall(MatSetValue(A,0,0,eig,INSERT_VALUES));      /* h* term */
    PetscCall(MatSetValue(A,1,1,cone[1] == v ? 1:-1,INSERT_VALUES));  /* flux balance bdry condition */
    rhs[0] = eig*h-hv; /* riemann invariant at the boundary */
    rhs[1] = 0.0; 

  PetscCall(VecRestoreArray(Rhs,&rhs));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode NRPSetFromOptions_Linearized(PetscOptionItems *PetscOptionsObject,NetRP rp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode NRPView_Linearized(NetRP rp,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

PETSC_EXTERN PetscErrorCode NetRPCreate_Linearized(NetRP rp)
{
  PetscFunctionBegin;
  rp->data = NULL;
  rp->ops->setfromoptions  = NRPSetFromOptions_Linearized;
  rp->ops->view            = NRPView_Linearized;
  rp->ops->createLinearStar = NetRPCreateLinearStar_Linearized; 
  rp->physicsgenerality = Specific; /* should be general, this is the wrong implementation */ 
  rp->solvetype         = Linear; 
  
  rp->numfields = 2; /*speficif for now */
  PetscFunctionReturn(0);
}

