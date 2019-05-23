
static char help[] = "Uses MLMC on a simple wave equation problem\n\n";

/*

    This code is a copy of the MATLAB code developed by XXXX and presented at
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

/*
    Data structure for MLMC algorithm
*/
typedef struct {
  PetscInt  q;                /* order of accuracy of deterministic solver */
  PetscInt  q1;               /* convergence rate of weak error */
  PetscInt  q2;               /* convergence rate of strong error */
  PetscInt  beta;             /* mesh refinement parameter */
  PetscReal gamma;           /* it appears in the work formula Wm = h^(-gamma) */
  PetscReal theta;           /* splitting parameter */
  PetscReal C_alpha;         /* constant in statistical error, given a failure probability */
} MLMC;

/*
    Data structure for the solution of a single wave equation problem on a given grid
*/
typedef struct {
  DM        da;
  PetscReal hx,hy,dt,T;
  PetscInt  nt;
  PetscInt  kx,ky;
  Vec       x,y;      /* coordinates of local part of tensor product mesh */
} WaveSimulation;

PetscErrorCode wave_solver(WaveSimulation*,PetscReal,PetscReal*);
PetscErrorCode WaveSimulationCreate(MPI_Comm,PetscReal,PetscInt,PetscInt,PetscInt,PetscInt,WaveSimulation**);
PetscErrorCode WaveSimulationRefine(WaveSimulation*,WaveSimulation**);
PetscErrorCode WaveSimulationDestroy(WaveSimulation**);

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscReal       h0 = .1;  /* initial step size */
  PetscInt        nx,ny;
  WaveSimulation  *ws;
  MLMC            mlmc;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  mlmc.q       = 2;
  mlmc.q1      = mlmc.q;
  mlmc.q2      = 2*mlmc.q;
  mlmc.beta    = 2;
  mlmc.gamma   = 3;
  mlmc.theta   = 0.5;
  mlmc.C_alpha = 4;

  nx = ny = 1 + (PetscInt)PetscRoundReal(2.0/h0);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&nx,NULL);
  ny = nx;
  ierr = WaveSimulationCreate(PETSC_COMM_WORLD,.5,nx,ny,6,4,&ws);CHKERRQ(ierr);

  PetscReal w = 1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-w",&w,NULL);
  PetscReal QoI;
  wave_solver(ws,w,&QoI);

  ierr = WaveSimulationDestroy(&ws);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode WaveSimulationSetup(WaveSimulation *ws,PetscReal T,PetscInt kx,PetscInt ky)
{
  PetscErrorCode ierr;
  PetscInt       nx,ny,nt,xs,ys,xn,yn,i;
  PetscReal      *x,*y;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(ws->da,NULL,&nx,&ny,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ws->kx = kx;
  ws->ky = ky;
  ws->hx = 2.0/(nx-1);
  ws->hy = 2.0/(ny-1);
  ws->dt = 0.5*ws->hx;
  nt     = 1 + (PetscInt) PetscFloorReal(T/ws->dt);
  ws->dt = T/nt;
  ierr = DMDAGetCorners(ws->da,&xs,&ys,NULL,&xn,&yn,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,xn,&ws->x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,yn,&ws->y);CHKERRQ(ierr);
  ierr = VecGetArray(ws->x,&x);CHKERRQ(ierr);
  for (i=0; i<xn; i++) x[i] = -1.0 + ws->hx*(i+xs);
  ierr = VecRestoreArray(ws->x,&x);CHKERRQ(ierr);
  ierr = VecGetArray(ws->y,&y);CHKERRQ(ierr);
  for (i=0; i<yn; i++) y[i] = -1.0 + ws->hy*(i+ys);
  ierr = VecRestoreArray(ws->y,&y);CHKERRQ(ierr);
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
  ierr = PetscNew(&wsf);CHKERRQ(ierr);
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
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      xy[i][j] = x[i]*y[j];
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
  ierr = DMDAVecRestoreArray(ws->da,ulocal,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(ws->da,vL,&L);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(ws->da,&ulocal);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DM             da = ws->da;
  Vec            u0,u1,u2,f2,L,f;

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
  ierr = f2_fun(ws,w,f2);CHKERRQ(ierr); VecView(f2,0);
  ierr = laplac(ws,u0,L);CHKERRQ(ierr); VecView(L,0);
  ierr = forcing(ws,0,w,f);CHKERRQ(ierr); VecView(f,0);
  /* u1 = u0 + dt*f2 + 0.5*(dt^2)*(L + f); */
  printf("%g\n",ws->dt);
  ierr = VecAXPY(L,1.0,f);CHKERRQ(ierr);                  /*  L = L + f */
  ierr = VecAYPX(L,.5*ws->dt*ws->dt,u0);CHKERRQ(ierr);    /*  L = u0 + 0.5*(dt^2)L */
  ierr = VecWAXPY(u1,ws->dt,f2,L);CHKERRQ(ierr);          /*  u1 = dt*f2 + L */

            VecView(u1,0);
            
  ierr = VecDestroy(&u0);CHKERRQ(ierr);
  ierr = VecDestroy(&u1);CHKERRQ(ierr);
  ierr = VecDestroy(&u2);CHKERRQ(ierr);
  ierr = VecDestroy(&f2);CHKERRQ(ierr);
  ierr = VecDestroy(&L);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*TEST

  build:
    requires: !complex

TEST*/
