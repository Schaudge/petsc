
static char help[] = "Uses MLMC on a simple wave equation problem\n\n";

/*

    This code is a copy of the MATLAB code developed by Mohammad Motamed and presented at
    an Argonne National Laboratory tutorial on uncertainty quantification
    May 20 through May 24th, organized by Oana Marin who suggested this
    example.
*/

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscmatlab.h>

/*
    Data structure for MLMC algorithm
*/
#define MLMC_MAX_LEVELS 10
typedef struct {
  PetscInt  q;               /* order of accuracy of deterministic solver */
  PetscInt  q1;              /* convergence rate of weak error */
  PetscInt  q2;              /* convergence rate of strong error */
  PetscInt  beta;            /* mesh refinement parameter */
  PetscReal gamma;           /* it appears in the work formula Wm = h^(-gamma) */
  PetscReal theta;           /* splitting parameter */
  PetscReal C_alpha;         /* constant in statistical error, given a failure probability */
  PetscInt  M0;              /* initial number of grid realizations for the first three levels */
  PetscInt  Nl[MLMC_MAX_LEVELS]; /* number of realizations run on each level */
  PetscInt  L;                   /* total number of levels */
} MLMC;

/*
    Data structure for the solution of a single wave equation problem on a given grid
*/
typedef struct {
  DM        da;
  PetscReal hx,hy,dt,T;
  PetscInt  nx,ny;
  PetscInt  nt;
  PetscInt  kx,ky;
  Vec       x,y;      /* coordinates of local part of tensor product mesh */
  PetscReal xQ;       /* location of QoI  */
} WaveSimulation;

PetscErrorCode wave_solver(WaveSimulation*,PetscReal,PetscReal*);
PetscErrorCode mlmc_wave(WaveSimulation **,PetscInt,PetscInt,PetscReal[]);
PetscErrorCode fmlmc(MLMC *,WaveSimulation **,PetscReal,PetscReal*);
PetscErrorCode WaveSimulationCreate(MPI_Comm,PetscReal,PetscInt,PetscInt,PetscInt,PetscInt,WaveSimulation**);
PetscErrorCode WaveSimulationRefine(WaveSimulation*,WaveSimulation**);
PetscErrorCode WaveSimulationDestroy(WaveSimulation**);

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscReal       EQ,h0 = .1;  /* initial step size */
  PetscInt        nx,ny,i;
  WaveSimulation  *ws[10];
  MLMC            mlmc;
  PetscReal       eps = .1;

  /* PetscReal sum[2]; */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  mlmc.q       = 2;
  mlmc.q1      = mlmc.q;
  mlmc.q2      = 2*mlmc.q;
  mlmc.beta    = 2;
  mlmc.gamma   = 3;
  mlmc.theta   = 0.5;
  mlmc.C_alpha = 4;
  mlmc.M0      = 100;

  nx = ny = 1 + (PetscInt)PetscRoundReal(2.0/h0);
  /*ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&nx,NULL);
   ny = nx;*/
  ierr = WaveSimulationCreate(PETSC_COMM_WORLD,.5,nx,ny,6,4,&ws[0]);CHKERRQ(ierr);
  ierr = WaveSimulationRefine(ws[0],&ws[1]);CHKERRQ(ierr);
  ierr = WaveSimulationRefine(ws[1],&ws[2]);CHKERRQ(ierr);

  /*  PetscReal w = 1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-w",&w,NULL);
  PetscReal QoI;
   wave_solver(ws[0],w,&QoI);*/

  /*  ierr = mlmc_wave(ws,1,5,sum);CHKERRQ(ierr);
   printf("%g %g \n",sum[0],sum[1]); */

  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),"rng('default')");CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-eps",&eps,NULL);CHKERRQ(ierr);
  ierr = fmlmc(&mlmc,ws,eps,&EQ);CHKERRQ(ierr);
  /* printf("%g \n",EQ);*/

  for (i=0; i<mlmc.L; i++) {
    ierr = WaveSimulationDestroy(&ws[i]);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode WaveSimulationSetup(WaveSimulation *ws,PetscReal T,PetscInt kx,PetscInt ky)
{
  PetscErrorCode ierr;
  PetscInt       nx,ny,xs,ys,xn,yn,i;
  PetscReal      *x,*y;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(ws->da,NULL,&nx,&ny,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ws->kx = kx;
  ws->ky = ky;
  ws->hx = 2.0/(nx-1);
  ws->hy = 2.0/(ny-1);
  ws->nx = nx;
  ws->ny = ny;
  ws->dt = 0.5*ws->hx;
  ws->T  = T;
  ws->nt = 1 + (PetscInt) PetscFloorReal(ws->T/ws->dt);
  ws->dt = ws->T/ws->nt;
  ierr = DMDAGetCorners(ws->da,&xs,&ys,NULL,&xn,&yn,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,xn,&ws->x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,yn,&ws->y);CHKERRQ(ierr);
  ierr = VecGetArray(ws->x,&x);CHKERRQ(ierr);
  for (i=0; i<xn; i++) x[i] = -1.0 + ws->hx*(i+xs);
  ierr = VecRestoreArray(ws->x,&x);CHKERRQ(ierr);
  ierr = VecGetArray(ws->y,&y);CHKERRQ(ierr);
  for (i=0; i<yn; i++) y[i] = -1.0 + ws->hy*(i+ys);
  ierr = VecRestoreArray(ws->y,&y);CHKERRQ(ierr);
  ws->xQ = .5;
  PetscFunctionReturn(0);
}

PetscErrorCode WaveSimulationDestroy(WaveSimulation **ws)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMDestroy(&(*ws)->da);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ws)->x);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ws)->y);CHKERRQ(ierr);
  ierr = PetscFree(*ws);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode WaveSimulationCreate(MPI_Comm comm,PetscReal T,PetscInt nx,PetscInt ny,PetscInt kx,PetscInt ky,WaveSimulation **ws)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(ws);CHKERRQ(ierr);
  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,nx,ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(*ws)->da);CHKERRQ(ierr);
  ierr = DMSetUp((*ws)->da);CHKERRQ(ierr);
  WaveSimulationSetup(*ws,T,kx,ky);
  PetscFunctionReturn(0);
}

PetscErrorCode WaveSimulationRefine(WaveSimulation *ws,WaveSimulation **wsf)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(wsf);CHKERRQ(ierr);
  ierr = DMRefine(ws->da,PetscObjectComm((PetscObject)ws->da),&(*wsf)->da);CHKERRQ(ierr);
  ierr = WaveSimulationSetup(*wsf,ws->T,ws->kx,ws->ky);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecOutterProduct(Vec vx,Vec vy,Vec vxy)
{
  PetscErrorCode  ierr;
  const PetscReal *x,*y;
  PetscReal       **xy;
  PetscInt        i,j,m,n;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(vx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vy,&y);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vx,&m);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vy,&n);CHKERRQ(ierr);
  ierr = VecGetArray2d(vxy,m,n,0,0,&xy);CHKERRQ(ierr);
  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) {
      xy[j][i] = x[i]*y[j];
    }
  }
  ierr = VecRestoreArray2d(vxy,m,n,0,0,&xy);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode u_exact_x(WaveSimulation *ws, PetscReal t,PetscReal w,Vec vf)
{
  PetscErrorCode  ierr;
  PetscInt        i,m;
  const PetscReal *x;
  PetscReal       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(ws->x,&x);CHKERRQ(ierr);
  ierr = VecGetArray(vf,&f);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vf,&m);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    f[i] = PetscSinReal(w*t-ws->kx*x[i]);
  }
  ierr = VecRestoreArrayRead(ws->x,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(vf,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode u_exact_y(WaveSimulation *ws, PetscReal t,PetscReal w,Vec vf)
{
  PetscErrorCode  ierr;
  PetscInt        i,m;
  const PetscReal *y;
  PetscReal       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(ws->y,&y);CHKERRQ(ierr);
  ierr = VecGetArray(vf,&f);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vf,&m);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    f[i] = PetscSinReal(ws->ky*y[i]);
  }
  ierr = VecRestoreArrayRead(ws->x,&y);CHKERRQ(ierr);
  ierr = VecRestoreArray(vf,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     function u=u_exact(t,x,y,w,kx,ky)
     u=sin(w*t-kx*x)'*sin(ky*y)';
*/
static PetscErrorCode u_exact(WaveSimulation *ws,PetscReal t,PetscReal w,Vec vf)
{
  PetscErrorCode ierr;
  Vec            vfx,vfy;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(ws->x,&vfx);CHKERRQ(ierr);
  ierr = VecDuplicate(ws->y,&vfy);CHKERRQ(ierr);
  ierr = u_exact_x(ws,t,w,vfx);CHKERRQ(ierr);
  ierr = u_exact_y(ws,t,w,vfy);CHKERRQ(ierr);
  ierr = VecOutterProduct(vfx,vfy,vf);CHKERRQ(ierr);
  ierr = VecDestroy(&vfx);CHKERRQ(ierr);
  ierr = VecDestroy(&vfy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode f2_fun_x(WaveSimulation *ws, PetscReal t,PetscReal w,Vec vf)
{
  PetscErrorCode  ierr;
  PetscInt        i,m;
  const PetscReal *x;
  PetscReal       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(ws->x,&x);CHKERRQ(ierr);
  ierr = VecGetArray(vf,&f);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vf,&m);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    f[i] = w*PetscCosReal(ws->kx*x[i]);
  }
  ierr = VecRestoreArrayRead(ws->x,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(vf,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     function f=f2_fun(x,y,w,kx,ky)
     f=w*cos(kx*x)'*sin(ky*y)'
*/
static PetscErrorCode f2_fun(WaveSimulation *ws,PetscReal w,Vec vf)
{
  PetscErrorCode ierr;
  Vec            vfx,vfy;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(ws->x,&vfx);CHKERRQ(ierr);
  ierr = VecDuplicate(ws->y,&vfy);CHKERRQ(ierr);
  ierr = f2_fun_x(ws,0.0,w,vfx);CHKERRQ(ierr);
  ierr = u_exact_y(ws,0.0,w,vfy);CHKERRQ(ierr);
  ierr = VecOutterProduct(vfx,vfy,vf);CHKERRQ(ierr);
  ierr = VecDestroy(&vfx);CHKERRQ(ierr);
  ierr = VecDestroy(&vfy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    function force=forcing(t,x,y,w,kx,ky)
    force=-(w^2-kx^2-ky^2)*sin(w*t-kx*x)'*sin(ky*y)';
*/
static PetscErrorCode forcing(WaveSimulation *ws,PetscReal t,PetscReal w,Vec vf)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = u_exact(ws,t,w,vf);CHKERRQ(ierr);
  ierr = VecScale(vf,-1.0*(w*w - ws->kx*ws->kx - ws->ky*ws->ky));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
    function L=laplac(u,hx,hy,nx,ny)
    L=zeros(size(u));
    for j=2:ny-1
        I=2:nx-1;
          L(j,I)=(1/hx^2)*u(j,I+1)+(1/hx^2)*u(j,I-1)+(1/hy^2)*u(j+1,I)+(1/hy^2)*u(j-1,I)-((2/hx^2)+(2/hy^2))*u(j,I);
    end
*/
static PetscErrorCode laplac(WaveSimulation *ws, Vec vu,Vec vL)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,xs,ys,xm,ym;
  const PetscReal **u;
  PetscReal       **L,ihx2 = 1.0/(ws->hx*ws->hx), ihy2 = 1.0/(ws->hy*ws->hy);
  Vec             ulocal;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(ws->da,&ulocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(ws->da,vu,INSERT_VALUES,ulocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(ws->da,vu,INSERT_VALUES,ulocal);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ws->da,ulocal,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ws->da,vL,&L);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(ws->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys+1; j<ys+ym-1; j++) {
    for (i=xs+1; i<xs+xm-1; i++) {
      L[j][i] = ihx2*u[j][i+1] + ihx2*u[j][i-1] + ihy2*u[j+1][i] + ihy2*u[j-1][i] - 2.0*(ihx2 + ihy2)*u[j][i];
    }
  }
  for (j=ys; j<ys+ym; j++) L[j][xs] = L[j][xs+xm-1] = 0.0;
  for (i=xs; i<xs+xm; i++) L[ys][i] = L[ys+ym-1][i] = 0.0;
  ierr = DMDAVecRestoreArray(ws->da,ulocal,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ws->da,vL,&L);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ws->da,&ulocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Sets the exact solution onto the boundary values of the domain
     This is the same as u_exact() using a tensor product except it only updates boundary values
*/
static PetscErrorCode WaveSimulationPatchBoundary(WaveSimulation *ws, PetscReal t, PetscReal w, Vec vu)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,xs,ys,xm,ym,nx,ny;
  PetscReal       **u;
  const PetscReal *fx,*fy;
  Vec             vfx,vfy;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(ws->x,&vfx);CHKERRQ(ierr);
  ierr = VecDuplicate(ws->y,&vfy);CHKERRQ(ierr);
  ierr = u_exact_x(ws,t,w,vfx);CHKERRQ(ierr);
  ierr = u_exact_y(ws,t,w,vfy);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vfx,&fx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vfy,&fy);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(ws->da,vu,&u);CHKERRQ(ierr);
  ierr = DMDAGetInfo(ws->da,NULL,&nx,&ny,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(ws->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  if (xs == 0) {
    for (j=ys; j<ys+ym; j++) u[j][0] = fx[0]*fy[j-ys];       /* Note that u[][] uses the parallel numbering while fx[] and fy[] use the local */
  }
  if (xs+xm == nx) {
    for (j=ys; j<ys+ym; j++) u[j][nx-1] = fx[xm-1]*fy[j-ys];
  }
  if (ys == 0) {
    for (i=xs; i<xs+xm; i++) u[0][i] = fx[i-xs]*fy[0];
  }
  if (ys+ym == ny) {
    for (i=xs; i<xs+xm; i++) u[ny-1][i] = fx[i-xs]*fy[ym-1];
  }
  ierr = DMDAVecRestoreArray(ws->da,vu,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vfx,&fx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vfy,&fy);CHKERRQ(ierr);
  ierr = VecDestroy(&vfx);CHKERRQ(ierr);
  ierr = VecDestroy(&vfy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Solves a single wave equation on a fixed grid

   Input Parameters:
    ws - the information about the grid the problem is solved on including timestep etc
    w - parameter of the problem

   Output Parameter:
    QoI - the quantity of interest computed from the solution
*/
PetscErrorCode wave_solver(WaveSimulation *ws,PetscReal w,PetscReal *QoI)
{
  PetscErrorCode  ierr;
  DM              da = ws->da;
  Vec             u0,u1,u2,f2,L,f,u0temp;
  PetscInt        k,mQ;
  PetscReal       t;
  const PetscReal **u1_array;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(da,&u0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&f2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&L);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&f);CHKERRQ(ierr);

  /* compute initial condition */
  ierr = u_exact(ws,0,w,u0);CHKERRQ(ierr);

  /* compute solution at first time step */
  ierr = f2_fun(ws,w,f2);CHKERRQ(ierr);
  ierr = laplac(ws,u0,L);CHKERRQ(ierr);
  ierr = forcing(ws,0,w,f);CHKERRQ(ierr);
  /* u1 = u0 + dt*f2 + 0.5*(dt^2)*(L + f); */
  ierr = VecAXPY(L,1.0,f);CHKERRQ(ierr);                  /*  L = L + f */
  ierr = VecAYPX(L,.5*ws->dt*ws->dt,u0);CHKERRQ(ierr);    /*  L = u0 + 0.5*(dt^2)L */
  ierr = VecWAXPY(u1,ws->dt,f2,L);CHKERRQ(ierr);          /*  u1 = dt*f2 + L */
  ierr = WaveSimulationPatchBoundary(ws,ws->dt,w,u1);CHKERRQ(ierr);

  for (k=0; k<ws->nt-1; k++) {
    t = (k+1)*ws->dt;
    /* f=forcing(t,x,y,w,kx,ky); */
    ierr = forcing(ws,t,w,f);CHKERRQ(ierr);
    /* L=laplac(u1,hx,hy,nx,ny); */
    ierr = laplac(ws,u1,L);CHKERRQ(ierr);
    /* u2=2*u1-u0+(dt^2)*(L+f); */
    ierr = VecAXPY(L,1.0,f);CHKERRQ(ierr);                   /*  L = L + f */
    ierr = VecAXPBY(L,-1.0,ws->dt*ws->dt,u0);CHKERRQ(ierr);  /*  L = -u0 + (dt^2)L */
    ierr = VecWAXPY(u2,2.0,u1,L);CHKERRQ(ierr);              /*  u2 = 2.0*u1 + L */
    ierr = WaveSimulationPatchBoundary(ws,t+ws->dt,w,u2);CHKERRQ(ierr);

    /*  switch solution at different time levels */
    u0temp = u0;
    u0     = u1;
    u1     = u2;
    u2     = u0temp;
  }

  /* compute quantity of interest; currently only works sequentially */
  /* Note that u1 contains the current solution, unlike the Matlab code where it is in u2 */
  mQ   = (PetscInt)PetscRoundReal(0.5*(ws->nx-1)*(1.0+ws->xQ));
  ierr = DMDAVecGetArray(ws->da,u1,&u1_array);CHKERRQ(ierr);
  *QoI =u1_array[mQ][mQ];
  ierr = DMDAVecRestoreArray(ws->da,u1,&u1_array);CHKERRQ(ierr);

  /* printf("%g \n",*QoI); */

  ierr = VecDestroy(&u0);CHKERRQ(ierr);
  ierr = VecDestroy(&u1);CHKERRQ(ierr);
  ierr = VecDestroy(&u2);CHKERRQ(ierr);
  ierr = VecDestroy(&f2);CHKERRQ(ierr);
  ierr = VecDestroy(&L);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Evaluates the sum of the differences of the QoI for two adjacent levels
*/
PetscErrorCode mlmc_wave(WaveSimulation **ws,PetscInt l,PetscInt M,PetscReal sum1[])
{
  PetscErrorCode ierr;
  PetscInt       N1;
  PetscReal      Qf,Qc,w;

  PetscFunctionBeginUser;
  sum1[0] = sum1[1] = 0;
  for (N1=0; N1<M; N1++) {
    ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),"x = rand(1,1);");CHKERRQ(ierr);
    ierr = PetscMatlabEngineGetArray(PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF),1,1,&w,"x");CHKERRQ(ierr);
    w += 10;
    ierr = wave_solver(ws[l],w,&Qf);CHKERRQ(ierr);
    if (l == 0) {
      Qc = 0;
    } else {
      ierr = wave_solver(ws[l-1],w,&Qc);CHKERRQ(ierr);
    }
    sum1[0] += (Qf-Qc);
    sum1[1] += (Qf-Qc)*(Qf-Qc);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode fmlmc(MLMC *mlmc,WaveSimulation **ws,PetscReal eps,PetscReal *EQ)
{
  PetscErrorCode ierr;
  PetscInt       L,dNl[MLMC_MAX_LEVELS];
  PetscReal      suml[MLMC_MAX_LEVELS][2],mul[MLMC_MAX_LEVELS],Vl[MLMC_MAX_LEVELS],Wl[MLMC_MAX_LEVELS],sumVlWl;
  PetscInt       sumdNl = 0,i,l,Ns[MLMC_MAX_LEVELS];

  PetscFunctionBeginUser;
  L    = 3;
  ierr = PetscMemzero(mlmc->Nl,sizeof(mlmc->Nl));CHKERRQ(ierr);
  ierr = PetscMemzero(suml,sizeof(suml));CHKERRQ(ierr);
  ierr = PetscMemzero(dNl,sizeof(dNl));CHKERRQ(ierr);
  dNl[0] = dNl[1] = dNl[2] = mlmc->M0;

  for (i=0; i<L; i++) sumdNl += dNl[i];
  while (sumdNl) {
    ierr = PetscInfo1(NULL,"Starting the MLMC loop: total dNl %D individual dNl follow\n",sumdNl);CHKERRQ(ierr);
    for (i=0; i<L; i++) {
      ierr = PetscInfo4(NULL,"MLMC Level dNl[%D] %D  Vl[%D] %g\n",i,dNl[i],i,(double)Vl[i]);CHKERRQ(ierr);
    }
    /* update sample sums */
    for (l=0; l<L; l++) {
      if (dNl[l] > 0) {
        PetscReal sums[2];
        /*        sums = feval(mlmc_l,l,dNl(l+1),q,T,h0,beta); */
        mlmc_wave(ws,l,dNl[l],sums);CHKERRQ(ierr);
        mlmc->Nl[l] += dNl[l];
        suml[l][0]  += sums[0];
        suml[l][1]  += sums[1];
      }
    }

    /*  compute variances for levels l=0:L (fromula (6) in lecture notes)
        mul = abs(suml(1,:)./Nl);
        Vl = max(0, suml(2,:)./Nl - mul.^2);
    */
    for (i=0; i<L; i++) mul[i] = PetscAbsReal(suml[i][0]/mlmc->Nl[i]);
    for (i=0; i<L; i++) Vl[i] = PetscMax(0.0,suml[i][1]/mlmc->Nl[i] - mul[i]*mul[i]);

    /* update optimal number of samples (fromula (4) in lecture notes)
       Wl  = beta.^(gamma*(0:L));
       Ns  = ceil(sqrt(Vl./Wl) * sum(sqrt(Vl.*Wl)) / (theta*eps^2/C_alpha));
       dNl = max(0, Ns-Nl);
    */
    for (i=0; i<L; i++) Wl[i] = PetscPowReal(mlmc->beta,mlmc->gamma*i);
    sumVlWl = 0.0; for (i=0; i<L; i++) sumVlWl += PetscSqrtReal(Vl[i]*Wl[i]);
    for (i=0; i<L; i++) Ns[i]   = (PetscInt)PetscCeilReal(PetscSqrtReal(Vl[i]/Wl[i])*sumVlWl/(mlmc->theta*eps*eps/mlmc->C_alpha));
    for (i=0; i<L; i++) dNl[i]  = PetscMax(0,Ns[i]-mlmc->Nl[i]);

    /*  if (almost) converged, estimate remaining error and decide
         whether a new level is required

        if sum( dNl > 0.01*Nl ) == 0
          range = -2:0;
          rem = max(mul(L+1+range).*beta.^(q1*range))/(beta^q1 - 1); %formula (5)
          if rem > (1-theta)*eps
            L=L+1;
            Vl(L+1) = Vl(L) / beta^q2;   %formula (7) in lecture notes
            Nl(L+1) = 0;
            suml(1:2,L+1) = 0;

            Wl  = beta.^(gamma*(0:L));
            Ns  = ceil(sqrt(Vl./Wl) * sum(sqrt(Vl.*Wl)) / (theta*eps^2/C_alpha));
            dNl = max(0, Ns-Nl);
          end
       end
    */
    sumdNl = 0.0; for (i=0; i<L; i++) sumdNl += (dNl[i] > .01*mlmc->Nl[i]);
    if (!sumdNl) {
      PetscReal rem = 0.0;
      for (i=-2; i<1; i++) {
        rem = PetscMax(rem,mul[L-1+i]*PetscPowReal(mlmc->beta,mlmc->q1*i)/(PetscPowReal(mlmc->beta,mlmc->q1) - 1.0));
      }
      /* printf("%g\n",rem);*/
      if (rem > (1.0 - mlmc->theta)*eps) {
        PetscInt i;
        ierr = PetscInfo(NULL,"Adding another MCML level to the hiearchy\n");CHKERRQ(ierr); printf("adding level\n");
        L = L + 1;
        ierr = WaveSimulationRefine(ws[L-2],&ws[L-1]);CHKERRQ(ierr);
        Vl[L-1] = Vl[L-2]/PetscPowReal(mlmc->beta,mlmc->q2);
        for (i=0; i<L; i++) Wl[i] = PetscPowReal(mlmc->beta,mlmc->gamma*i);
        sumVlWl = 0.0; for (i=0; i<L; i++) sumVlWl += PetscSqrtReal(Vl[i]*Wl[i]);
        for (i=0; i<L; i++) {Ns[i]  = (PetscInt)PetscCeilReal(PetscSqrtReal(Vl[i]/Wl[i])*sumVlWl/(mlmc->theta*eps*eps/mlmc->C_alpha));printf("NS %d %d\n",i,Ns[i]);}
        for (i=0; i<L; i++) {dNl[i] = PetscMax(0,Ns[i]-mlmc->Nl[i]); printf("dN %d %d\n",i,dNl[i]);}
      }
    }
    sumdNl = 0.0; for (i=0; i<L; i++) sumdNl += dNl[i]; printf("sumdNl %d\n",sumdNl);
  }


  
  /* finally, evaluate multilevel estimator
     EQ = sum(suml(1,:)./Nl);
  */
  *EQ = 0; for (i=0; i<L; i++) *EQ += suml[i][0]/mlmc->Nl[i];
  mlmc->L = L;
  ierr = PetscInfo2(NULL,"Completed MLMC algorith QoI %g Number of levels %D\n",*EQ,L);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

/*TEST

  build:
    requires: !complex

TEST*/
