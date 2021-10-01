static const char help[] = "Comparing the general lax solver with known working implementations for specific physics";


#include <petscriemannsolver.h>

/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
} ShallowCtx;

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

  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(L.u)+PetscSqrtScalar(g*L.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{

    PetscErrorCode    ierr;
    RiemannSolver     rs;
    ShallowCtx        ctx;
    PetscReal         uL[2] = {1.0, 0.0}, uR[2] = {2.0,0.0},*flux, maxspeed; 

    ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
    ctx.gravity = 9.81; 
    ierr = PetscOptionsBegin(MPI_COMM_SELF,NULL,"Lax ex1 options","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-gravity","strength of gravity","",ctx.gravity,&ctx.gravity,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    
    ierr = RiemannSolverCreate(MPI_COMM_SELF,&rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(rs,&ctx);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(rs,1,2,ShallowFluxVoid);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(rs,ShallowEig);CHKERRQ(ierr);
  
    ierr = RiemannSolverEvaluate(rs,uL,uR,&flux,&maxspeed);CHKERRQ(ierr);
    ierr = PetscPrintf(MPI_COMM_SELF,"Shallow Water Lax-Friedrich Test: \n \n General Riemann Solver \n Flux 0: %e \n Flux 1: %e \n Maxspeed: %e \n\n", 
           flux[0],flux[1],maxspeed);CHKERRQ(ierr);

    ierr = PhysicsRiemann_Shallow_Rusanov(&ctx,2,uL,uR,flux,&maxspeed);CHKERRQ(ierr);
    ierr = PetscPrintf(MPI_COMM_SELF,"HandCoded Shallow Water Riemann Solver \n Flux 0: %e \n Flux 1: %e \n Maxspeed: %e \n\n", 
           flux[0],flux[1],maxspeed);CHKERRQ(ierr);

    ierr = RiemannSolverDestroy(&rs);CHKERRQ(ierr); 
    ierr = PetscFinalize();CHKERRQ(ierr);
}