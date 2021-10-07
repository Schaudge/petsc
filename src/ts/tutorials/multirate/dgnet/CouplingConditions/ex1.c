static const char help[] = "Test the linearization by computing the outgoing flucutation for each edge for a junction";


/* 
    TODO : Along with most of these examples the physics needs to be cleaned up and additional classes 
    need to be added to remove all of this ugly code, as well as make things more adaptable for my experimentation. 

    Nothing really works properly right now. 
*/
#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petsc/private/kernels/blockinvert.h>
#include <petscriemannsolver.h>

PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }

/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE PetscErrorCode ShallowFlux(void *ctx,const PetscReal *u,PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx*)ctx;
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE void ShallowFluxVoid(void *ctx,const PetscReal *u,PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx*)ctx;
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

PETSC_STATIC_INLINE void ShallowFlux2(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1]*u[0];
  f[1] = PetscSqr(u[1])*u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

PETSC_STATIC_INLINE void ShallowFlux3(ShallowCtx *phys,PetscScalar h, PetscScalar q,PetscScalar *f)
{
  f[0] = q;
  f[1] = PetscSqr(q)/h + 0.5*phys->gravity*PetscSqr(h);
}

PETSC_STATIC_INLINE void ShallowEig(void *ctx,const PetscReal *u,PetscReal *eig)
{
    ShallowCtx *phys = (ShallowCtx*)ctx;
    eig[0] = u[1]/u[0] - PetscSqrtReal(phys->gravity*u[0]); /*left wave*/
    eig[1] = u[1]/u[0] + PetscSqrtReal(phys->gravity*u[0]); /*right wave*/
}
static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};
  PetscReal                 tol = 1e-6;

  PetscFunctionBeginUser;
  /* Positivity preserving modification*/
  if (L.h < tol) L.u = 0.0;
  if (R.h < tol) R.u = 0.0;

  /*simple positivity preserving limiter*/
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;

  ShallowFlux2(phys,(PetscScalar*)&L,fL);
  ShallowFlux2(phys,(PetscScalar*)&R,fR);

  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscReal      tol = 1e-6;

  PetscFunctionBeginUser;
  c         = PetscSqrtScalar(u[0]*phys->gravity);

  if (u[0] < tol) { /*Use conservative variables*/
    X[0*2+0]  = 1;
    X[0*2+1]  = 0;
    X[1*2+0]  = 0;
    X[1*2+1]  = 1;
    speeds[0] = - c;
    speeds[1] =   c;
  } else {
    speeds[0] = u[1]/u[0] - c;
    speeds[1] = u[1]/u[0] + c;
    X[0*2+0]  = 1;
    X[0*2+1]  = speeds[0];
    X[1*2+0]  = 1;
    X[1*2+1]  = speeds[1];
  }

  ierr = PetscArraycpy(Xi,X,4);CHKERRQ(ierr);
  ierr = PetscKernel_A_gets_inverse_A_2(Xi,0,PETSC_FALSE,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_ShallowNetwork(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      if (edgeid == 0) {
        u[0] = 6;
        u[1] = 0.0;
      }  else {
        u[0] = 1;
        u[1] = 0.0;
      }
      break;
    case 1: /* Initial 1-3 are from Jingmei's and Bennedito's paper */
      if (edgeid == 0) {
        u[0] = 0.5;
        u[1] = 0.1;
      } else if (edgeid == 1) {
        u[0] = 0.5;
        u[1] = 0.0;
      } else  {
        u[0] = 1;
        u[1] = 0.0;
      }
      break;
    case 2:
      if (edgeid == 0) {
        u[0] = 1.0+PetscExpReal(-20.0*(x+1.0)*(x+1.0));
        u[1] = u[0]/2.0;
      } else if (edgeid == 1) {
        u[0] = 1.0;
        u[1] = 0.0;
      } else  {
        u[0] = 0.5;
        u[1] = 0.0;
      }
      break;
    case 3:
      if (edgeid == 0) {
        u[0] = ((x>=0 && x<=0.2) || (x>=0.4 && x<=0.6) || (x>=0.8 && x<=1.0)) ? 1.5 : 1.0 ;
        u[1] = u[0]/5.0;
      } else if (edgeid == 1) {
        u[0] = 1.0;
        u[1] = 0.0;
      } else {
        u[0] = 0.5;
        u[1] = 0.0;
      }
      break;
    case 4: /* Sunny's Test Case*/
      if (edgeid == 0) { /* Not sure what the correct IC is here*/
          u[0] = ((x>=7 && x<=9)) ? 2.0-PetscSqr(x-8): 1.0; 
          u[1] = 0.0;
      } else {
          u[0] = 1.0; 
          u[1] = 0.0;
       }
      break;
    case 5: /* Roundabout Pulse */
      u[0] = !(edgeid%2) ? 2 : 1; 
      u[1] = 0;
      break;
 /* The following problems are based on geoemtrically 1d Networks, no notion of edgeid is considered */
    case 6:
      u[0] = (x < 10) ?   1 : 0.1; 
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 7:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 8:
      u[0] = (x < 20) ?  1 : 0;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 9:
      u[0] = (x < 30) ? 0: 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 10:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 11:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x);
      u[1] = 1*u[0];
      break;
    case 12:
      u[0] = 1.0;
      u[1] = 1.0;
      break;
    case 13:
      u[0] = (x < -2) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < -2) ? 0 : 0;
      break;
    case 14:
      u[0] = (x < 25) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 25) ? 0 : 0;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing in and edge 2 pointing out */
static PetscErrorCode PhysicsVertexFlux_2Edge_InOut(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uL, 2nd as uR. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV,uV+dof,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing out and edge 2 pointing in  */
static PetscErrorCode PhysicsVertexFlux_2Edge_OutIn(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uR, 2nd as uL. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV+dof,uV,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

PetscReal ShallowRiemannExact_Left(const PetscScalar hl, const PetscScalar vl, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hl ? vl-2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hl)) : vl- (h-hl)*PetscSqrtScalar(g*(h+hl)/(2.0*h*hl));
}

PetscReal ShallowRiemannExact_Right(const PetscScalar hr, const PetscScalar vr, PetscScalar h)
{
  const PetscScalar g = 9.81;
  return h<hr ? vr+2.0*(PetscSqrtScalar(g*h)-PetscSqrtScalar(g*hr)) : vr+(h-hr)*PetscSqrtScalar(g*(h+hr)/(2.0*h*hr));
}

PetscReal ShallowRiemannEig_Right(const PetscScalar hr, const PetscScalar vr)
{
  const PetscScalar g = 9.81;
  return vr + PetscSqrtScalar(g*hr);
}

PetscReal ShallowRiemannEig_Left(const PetscScalar hl, const PetscScalar vl)
{
  const PetscScalar g = 9.81;
  return vl - PetscSqrtScalar(g*hl);
}

typedef struct {
  Junction          junct;
  const PetscScalar *uV;
} Shallow_Couple_InputWrapper;

PetscErrorCode RiemannInvariant_Couple_Shallow(SNES snes,Vec x,Vec f, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,n,dof=2;
  Shallow_Couple_InputWrapper *wrapper= (Shallow_Couple_InputWrapper*) ctx;
  const Junction  junct = wrapper->junct;
  const PetscScalar *ustar,*uV = wrapper->uV;
  PetscScalar *F;

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecGetArray(f,&F);CHKERRQ(ierr);

  F[n-2] = (junct->dir[junct->numedges-1] == EDGEIN) ?  ustar[n-1] : -ustar[n-1];

  for (i=0; i<junct->numedges-1; i++)
  {
    F[dof*i] = ustar[dof*i] - ustar[dof*(i+1)];
    F[n-2] += (junct->dir[i] == EDGEIN) ? ustar[dof*i+1] : -ustar[dof*i+1];
    /* output for this edges vstar eqn */
    if (junct->dir[i] == EDGEIN){
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Left(uV[dof*i],uV[dof*i+1]/uV[dof*i],ustar[dof*i]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[dof*i+1] = ustar[dof*i+1] - ShallowRiemannExact_Right(uV[dof*i],uV[dof*i+1]/uV[dof*i],ustar[dof*i]);
    }
  }

   if (junct->dir[junct->numedges-1] == EDGEIN){
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Left(uV[n-2],uV[n-1]/uV[n-2],ustar[n-2]);
    } else { /* junct->dir[i] == EDGEOUT */
      F[n-1] = ustar[n-1] - ShallowRiemannExact_Right(uV[n-2],uV[n-1]/uV[n-2],ustar[n-2]);
    }

  ierr = VecRestoreArrayRead(x,&ustar);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsVertexFlux_Shallow_Full(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  const Junction  junct = (Junction) _junct;
  Shallow_Couple_InputWrapper wrapper = {junct,uV};
  PetscInt        i,n,dof = fvnet->physics.dof;
  PetscScalar     *x;


  PetscFunctionBeginUser;

  ierr = SNESSetFunction(fvnet->snes,junct->rcouple,RiemannInvariant_Couple_Shallow,&wrapper);
  /* Set initial condition as the reconstructed h,v values*/
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  ierr = VecGetSize(junct->xcouple,&n);CHKERRQ(ierr);
  for (i=0;i<junct->numedges;i++) {
    x[i*dof]   = uV[i*dof];
    x[i*dof+1] = uV[i*dof+1]/uV[i*dof];
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  ierr = SNESSolve(fvnet->snes,NULL,junct->xcouple);CHKERRQ(ierr);
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  /* Compute the Flux from the computed star values */
  for (i=0;i<junct->numedges;i++) {
    ShallowFlux2(fvnet->physics.user,x+(i*dof),flux+(i*dof));
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  *maxspeed = 0.0; /* Ignore the computation of the maxspeed */
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsVertexFlux_Shallow_Full_Linear(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  const Junction  junct = (Junction) _junct;
  PetscInt        i,dof = fvnet->physics.dof;
  PetscScalar     *x,*r,eig,h,v,hv;
  PetscBool       nonzeroinitial; 

  PetscFunctionBeginUser;

  ierr = VecGetArray(junct->rcouple,&r);CHKERRQ(ierr);
  /* Build the system matrix and rhs vector */
  for (i=1;i<junct->numedges;i++) {
    h  = uV[i*dof];
    hv = uV[i*dof+1];
    v  = hv/h; 
    eig = junct->dir[i] == EDGEIN ? 
      ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v);
    ierr = MatSetValue(junct->mat,i+1,i+1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
    ierr = MatSetValue(junct->mat,i+1,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
    ierr = MatSetValue(junct->mat,1,i+1,junct->dir[i] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
    r[i+1] = eig*h-hv; /* riemann invariant at the boundary */
  }
  /* Form the matrix in this reordered form to have nonzeros along the diagonal */
  h  = uV[0];
  hv = uV[1];
  v  = hv/h; 
  eig = junct->dir[0] == EDGEIN ? 
    ShallowRiemannEig_Left(h,v) : ShallowRiemannEig_Right(h,v);
  ierr = MatSetValue(junct->mat,0,1,-1 ,INSERT_VALUES);CHKERRQ(ierr); /* hv* term */
  ierr = MatSetValue(junct->mat,0,0,eig,INSERT_VALUES);CHKERRQ(ierr);      /* h* term */
  ierr = MatSetValue(junct->mat,1,1,junct->dir[0] == EDGEIN ? 1:-1,INSERT_VALUES);CHKERRQ(ierr);  
  r[0] = eig*h-hv; /* riemann invariant at the boundary */
  r[1] = 0.0;   
  ierr = VecRestoreArray(junct->rcouple,&r);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(junct->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(junct->mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = KSPGetInitialGuessNonzero(fvnet->ksp,&nonzeroinitial);CHKERRQ(ierr);
  if(nonzeroinitial) {
    /* Set initial guess as the reconstructed h,v values */
    ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
    for (i=0;i<junct->numedges;i++) {
      x[i+1] = uV[i*dof+1];
    }
    x[0] = uV[0];
    ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  }

  ierr = KSPSetOperators(fvnet->ksp,junct->mat,junct->mat);CHKERRQ(ierr);
  ierr = KSPSolve(fvnet->ksp,junct->rcouple,junct->xcouple);CHKERRQ(ierr);
  ierr = VecGetArray(junct->xcouple,&x);CHKERRQ(ierr);
  /* Compute the Flux from the computed star values */
  for (i=0;i<junct->numedges;i++) {
    ShallowFlux3(fvnet->physics.user,x[0],x[i+1],flux+(i*dof));
  }
  ierr = VecRestoreArray(junct->xcouple,&x);CHKERRQ(ierr);
  *maxspeed = 0.0; /* Ignore the computation of the maxspeed */
  PetscFunctionReturn(0);
}
/* THESE DONT WORK BE WAY WAY MORE CAREFUL HERE */
/* Okay the issue was a subtle instability due to floating point error (I think). Using differenet flux calculation functions 
for the riemann solver and exact fluxes, so small errors accumulate and removes the balancing of fluxs causing a blow up from nonlinearities*/
/* using the riemann solver for outflow instead of the exact flux fixes the issue */
static PetscErrorCode PhysicsVertexFlux_Outflow_Simple(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct) {
  PetscErrorCode  ierr;
  const DGNetwork dgnet = (DGNetwork)_dgnet;
  PetscInt        dof = dgnet->physics.dof;

  PetscFunctionBeginUser;
  *maxspeed = 0.0; 
  ierr = dgnet->physics.riemann(dgnet->physics.user,dof,uV,uV,flux,maxspeed);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsAssignVertexFlux_Shallow(const void* _dgnet, Junction junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        dof = 2;

  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        if (junct->dir[0] == EDGEIN) {
          if (junct->dir[1] == EDGEIN) {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          } else { /* dir[1] == EDGEOUT */
            junct->couplingflux = PhysicsVertexFlux_2Edge_InOut;
          }
        } else { /* dir[0] == EDGEOUT */
          if (junct->dir[1] == EDGEIN) {
            junct->couplingflux = PhysicsVertexFlux_2Edge_OutIn;
          } else { /* dir[1] == EDGEOUT */
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          }
        }
      } else if (junct->numedges == 1) {
         if (junct->dir[0] == EDGEIN) {
            junct->couplingflux =   junct->couplingflux = PhysicsVertexFlux_Outflow_Simple;
         } else {
            junct->couplingflux =   junct->couplingflux = PhysicsVertexFlux_Outflow_Simple;
         }
      } else {
        if(!fvnet->linearcoupling) {
          ierr = VecCreateSeq(MPI_COMM_SELF,junct->numedges*dof,&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDuplicate(junct->rcouple,&junct->xcouple);CHKERRQ(ierr);
          junct->couplingflux = PhysicsVertexFlux_Shallow_Full;
        } else {
          ierr = VecCreateSeq(MPI_COMM_SELF,junct->numedges+1,&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDuplicate(junct->rcouple,&junct->xcouple);CHKERRQ(ierr);
          ierr = MatCreate(MPI_COMM_SELF,&junct->mat);CHKERRQ(ierr);
          ierr = MatSetSizes(junct->mat,PETSC_DECIDE,PETSC_DECIDE,junct->numedges+1,junct->numedges+1);CHKERRQ(ierr);
          ierr = MatSetFromOptions(junct->mat);CHKERRQ(ierr); 
          ierr = MatSetUp(junct->mat);CHKERRQ(ierr); /* Could use a specific create seq mat here for improved performance */
          junct->couplingflux = PhysicsVertexFlux_Shallow_Full_Linear;
        }
      }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsDestroyVertexFlux_Shallow(const void* _fvnet, Junction junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_fvnet;

  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        // Nothing to Destroy 
      } else {
        if(!fvnet->linearcoupling) {
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
        } else {
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
          ierr = MatDestroy(&junct->mat);CHKERRQ(ierr);
        }
      }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Shallow(DGNetwork fvnet)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0;
  char              rname[256] = "rusanov";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  fvnet->physics.samplenetwork   = PhysicsSample_ShallowNetwork;
  fvnet->physics.destroy         = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.riemann         = PhysicsRiemann_Shallow_Rusanov;
  fvnet->physics.characteristic  = PhysicsCharacteristic_Shallow;
  fvnet->physics.vfluxassign     = PhysicsAssignVertexFlux_Shallow;
  fvnet->physics.vfluxdestroy    = PhysicsDestroyVertexFlux_Shallow;
  fvnet->physics.flux            = ShallowFlux;
  fvnet->physics.user            = user;
  fvnet->physics.dof             = 2;
  fvnet->physics.flux2           = ShallowFluxVoid; 
  fvnet->physics.fluxeig         = ShallowEig;
  ierr = PetscStrallocpy("height",&fvnet->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&fvnet->physics.fieldname[1]);CHKERRQ(ierr);

  user->gravity = 9.81;
  PetscFunctionReturn(0);
}

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  ierr = DGNetworkMonitorView(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode TSDGNetworkMonitor_GLVis(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor_Glvis   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  ierr = DGNetworkMonitorView_Glvis(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode TSDGNetworkMonitor_GLVis_NET(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor_Glvis   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  ierr = DGNetworkMonitorView_Glvis_NET(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  char              lname[256] = "minmod",physname[256] = "shallow",tname[256] = "adaptive";
  PetscFunctionList limiters = 0,physics = 0,timestep = 0;
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         fvnet;
  PetscInt          draw = 0,maxorder=1,*order;
  PetscBool         viewdm = PETSC_FALSE;
  PetscReal         maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscBool         singlecoupleeval,view3d=PETSC_FALSE,viewglvis=PETSC_FALSE,glvismode=PETSC_FALSE,viewfullnet=PETSC_FALSE,limit=PETSC_TRUE;
  DGNetworkMonitor  monitor=NULL;
  DGNetworkMonitor_Glvis monitor_gl = NULL;

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc1(1,&fvnet);CHKERRQ(ierr);
  ierr = PetscMemzero(fvnet,sizeof(*fvnet));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);

  /* Set default values */
  fvnet->comm           = comm;
  fvnet->cfl            = 0.9;
  fvnet->networktype    = 3;
  fvnet->hratio         = 1;
  maxtime               = 2.0;
  fvnet->Mx             = 10;
  fvnet->bufferwidth    = 0;
  fvnet->initial        = 1;
  fvnet->ymin           = 0;
  fvnet->ymax           = 2.0;
  fvnet->bufferwidth    = 4;
  fvnet->ndaughters     = 2;
  fvnet->linearcoupling = PETSC_FALSE;
  singlecoupleeval      = PETSC_FALSE; 
  fvnet->length         = 3.0;
  fvnet->view           = PETSC_TRUE;
  fvnet->jumptol        = 0.5; 

  /* Command Line Options */

  /* 
    TODO: Clean up all of these options. This is related to making new classes in PETScDT in order to ease all of 
    this ugly overhead, and promote reusability
  */
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux imiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-stepsize","Name of function to adapt the timestep size","",timestep,tname,tname,sizeof(tname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",fvnet->initial,&fvnet->initial,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",fvnet->networktype,&fvnet->networktype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact","Compare errors with exact solution","",fvnet->exact,&fvnet->exact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simulation","Compare errors with reference solution","",fvnet->simulation,&fvnet->simulation,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-lincouple","Use linearized coupling condition when available","",fvnet->linearcoupling,&fvnet->linearcoupling,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",fvnet->cfl,&fvnet->cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-hratio","Spacing ratio","",fvnet->hratio,&fvnet->hratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_max_time","Max Time to Run TS","",maxtime,&maxtime,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymin","Min y-value in plotting","",fvnet->ymin,&fvnet->ymin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymax","Max y-value in plotting","",fvnet->ymax,&fvnet->ymax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-length","Length of Edges in the Network","",fvnet->length,&fvnet->length,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",fvnet->Mx,&fvnet->Mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-bufferwidth","width of the buffer regions","",fvnet->bufferwidth,&fvnet->bufferwidth,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewdm","View DMNetwork Info in stdout","",viewdm,&viewdm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ndaughters","Number of daughter branches for network type 3","",fvnet->ndaughters,&fvnet->ndaughters,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-lincouplediff","Compare the results for linearcoupling and nonlinear","",fvnet->lincouplediff,&fvnet->lincouplediff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-singlecoupleeval","Use the single couple eval rhs functions","",singlecoupleeval,&singlecoupleeval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view","View the DG solution","",fvnet->view,&fvnet->view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_dump","Dump the Glvis view or socket","",glvismode,&glvismode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_3d","View a 3d version of edge","",view3d,&view3d,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_glvis","View GLVis of Edge","",viewglvis,&viewglvis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_full_net","View GLVis of Entire Network","",viewfullnet,&viewfullnet,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-jumptol","Set jump tolerance for lame one-sided limiter","",fvnet->jumptol,&fvnet->jumptol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(fvnet);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(fvnet->physics.dof,&order);CHKERRQ(ierr);
  ierr = MakeOrder(fvnet->physics.dof,order,maxorder);CHKERRQ(ierr);
  fvnet->physics.order = order;
  /* Generate Network Data */
  ierr = DGNetworkCreate(fvnet,fvnet->networktype,fvnet->Mx);CHKERRQ(ierr);
  /* Create DMNetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&fvnet->network);CHKERRQ(ierr);

  if (size == 1 && fvnet->view) {
    if (viewglvis) {
      ierr = DGNetworkMonitorCreate_Glvis(fvnet,&monitor_gl);CHKERRQ(ierr);
    } else {
      ierr = DGNetworkMonitorCreate(fvnet,&monitor);CHKERRQ(ierr);
    }
  }
  /* Set Network Data into the DMNetwork (on proc[0]) */
  ierr = DGNetworkSetComponents(fvnet);CHKERRQ(ierr);
  /* Delete unneeded data in fvnet */
  ierr = DGNetworkCleanUp(fvnet);CHKERRQ(ierr);
  ierr = DGNetworkBuildTabulation(fvnet);CHKERRQ(ierr);
    if (viewglvis) {
      if (viewfullnet) { 
        ierr =  DGNetworkMonitorAdd_Glvis_3D_NET(monitor_gl,"localhost",glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
      } else {
        if(view3d) {
          ierr = DGNetworkAddMonitortoEdges_Glvis_3D(fvnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        } else {
          ierr = DGNetworkAddMonitortoEdges_Glvis(fvnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = DGNetworkAddMonitortoEdges(fvnet,monitor);CHKERRQ(ierr);
    }

  /* Set up Riemann Solver */

    ierr = RiemannSolverCreate(fvnet->comm,&fvnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(fvnet->physics.rs,fvnet->physics.user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(fvnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(fvnet->physics.rs,1,fvnet->physics.dof,fvnet->physics.flux2);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(fvnet->physics.rs,fvnet->physics.fluxeig);CHKERRQ(ierr);

  /* Create Vectors */
  ierr = DGNetworkCreateVectors(fvnet);CHKERRQ(ierr);
  /* Set up component dynamic data structures */
  ierr = DGNetworkBuildDynamic(fvnet);CHKERRQ(ierr);
  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,fvnet->network);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,fvnet);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,DGNetRHS_RSVERSION,fvnet);CHKERRQ(ierr);
 
  ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,fvnet->cfl/fvnet->Mx/(2*maxorder+1));CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = DGNetworkProject(fvnet,fvnet->X,0.0);CHKERRQ(ierr);
  ierr = DGNetRHS(ts,0,fvnet->X,fvnet->Ftmp,fvnet);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
   if (size == 1 && fvnet->view) {
      if (viewglvis) {
        if(viewfullnet) {
          ierr = TSMonitorSet(ts, TSDGNetworkMonitor_GLVis_NET, monitor_gl, NULL);CHKERRQ(ierr);
        } else {
          ierr = TSMonitorSet(ts, TSDGNetworkMonitor_GLVis, monitor_gl, NULL);CHKERRQ(ierr);
        } 
      } else {
        ierr = TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL);CHKERRQ(ierr);
      }
    }
  if (limit) {
      /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
  for some reason ... no idea why prestage and post-stage callback functions have different forms) */  
    ierr = DGNetlimiter(ts,0,0,&fvnet->X);CHKERRQ(ierr);
    ierr = TSSetPostStage(ts,DGNetlimiter);CHKERRQ(ierr);
  }

  /* Evolve the PDE network in time */
  ierr = TSSolve(ts,fvnet->X);CHKERRQ(ierr);

  /* Clean up */
  if(fvnet->view && size==1){
    if(viewglvis) {
      ierr = DGNetworkMonitorDestroy_Glvis(&monitor_gl);
    } else {
      ierr = DGNetworkMonitorDestroy(&monitor);
    }
  } 
  ierr = RiemannSolverDestroy(&fvnet->physics.rs);CHKERRQ(ierr);
  ierr = DGNetworkDestroy(fvnet);CHKERRQ(ierr); /* Destroy all data within the network and within fvnet */
  ierr = DMDestroy(&fvnet->network);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&timestep);CHKERRQ(ierr);
  ierr = PetscFree(fvnet);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}