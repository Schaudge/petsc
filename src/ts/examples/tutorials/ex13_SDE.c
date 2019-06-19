

static char help[] = "Time-dependent SPDE in 2d. Adapted from ex13.c. \n";
/*
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1;
   At t=0: u(x,y) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

    mpiexec -n 2 ./ex13 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
    mpiexec -n 1 ./ex13 -snes_fd_color -ts_monitor_draw_solution
    mpiexec -n 2 ./ex13 -ts_type sundials -ts_monitor 
*/

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscmatlab.h>
#include "header.h"
#include <time.h>

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscInt  Lx;
  PetscInt  Ly;
  PetscReal c;
  Mat       A;
  DM        da;
} AppCtx;

extern PetscErrorCode FormInitialSolution(DM,Vec,void*);
extern PetscErrorCode BuildA(AppCtx*);
extern PetscErrorCode BuildCN(AppCtx*, PetscScalar);
extern PetscErrorCode FormCNRHS(AppCtx*, Vec, PetscScalar, Vec);
extern PetscErrorCode BuildR(Vec, AppCtx*);

int main(int argc,char **argv)
{
  char           *output;
  KSP            ksp;               /* KSP solver for Crank-Nicolson scheme*/
  Vec            rhs;               /* rhs vector for Crank-Nicolson scheme*/
  Vec            u,r;               /* solution vector , random vector */
  Vec            unew, uold;        /* vector for time stepping */
  PetscErrorCode ierr;
  AppCtx         user;              /* user-defined work context */
  PetscInt       Nx=8,Ny=8;
  PetscInt       i,tsteps;
  PetscReal      ftime=1.00;
  PetscScalar    dt   =0.01;
//  PetscViewer    viewfile;
  
  /* Initialize user application context */
  user.Lx = 1;
  user.Ly = 1;
  user.c = -30.0;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscMatlabEngineGetOutput(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),&output);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da,0.0,user.Lx,0.0,user.Ly,0.0,0.0);CHKERRQ(ierr);
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&unew);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uold);CHKERRQ(ierr);
  /* Set Matrix A */
  ierr = DMSetMatType(user.da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,&user.A);CHKERRQ(ierr);
    
  /* Euler scheme */
//  ierr = BuildA(&user);
//  ierr = MatScale(user.A,dt);
    
  /* Crank-Nicolson scheme */
  ierr = DMCreateGlobalVector(user.da,&rhs);CHKERRQ(ierr);
  ierr = BuildCN(&user,dt);
    
  /* Set Random vector r */
  ierr = DMSetVecType(user.da,VECSEQ);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&r);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(user.da,u,&user);CHKERRQ(ierr);
    
//    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"u0.m",&viewfile);CHKERRQ(ierr);
//    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
//    ierr = PetscObjectSetName((PetscObject)u,"sol0");CHKERRQ(ierr);
//    ierr = VecView(u,viewfile);CHKERRQ(ierr);
//    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
//    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Time stepping in Euler
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDAGetInfo(user.da,PETSC_IGNORE,&Nx,&Ny,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  tsteps = PetscRoundReal(ftime/dt);
  PetscInt tout = 2;
  PetscReal dim[2], domain[2], tm[2];
  dim[0] = Nx; dim[1] = Ny;
  domain[0] = user.Lx; domain[1] = user.Ly;
  tm[0] = dt; tm[1] = 0;
  ierr = PetscMatlabEnginePutArray(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),2,1,dim,"dim");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePutArray(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),2,1,domain,"domain");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");
  ierr = PetscObjectSetName((PetscObject)r,"r");
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"Nx = dim(1,1); Ny = dim(2,1); Lx = domain(1,1); Ly = domain(2,1);");CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"Nx");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"Ny");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"Lx");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"Ly");CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
  for (i=0; i<tsteps; i++)
  {
      if ( i%tout == 0)
      {
          ierr = PetscMatlabEnginePutArray(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),2,1,tm,"tm");CHKERRQ(ierr);
          ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"tm");CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
          
          ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),(PetscObject)u);CHKERRQ(ierr);
          ierr = PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),(PetscObject)r);CHKERRQ(ierr);
          ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"subplot(1,2,1);[X,Y]=meshgrid(linspace(0,Lx,Nx),linspace(0,Ly,Ny));surf(X,Y,reshape(u,Nx,Ny)');title({['Solution'],['Time t= ',num2str(tm(2,1))]});shading interp;axis([0 1 0 1 -0.5 1]);axis square;xlabel('X');ylabel('Y');view(2);colorbar;set(gca,'fontsize', 16);pause(0.01);");CHKERRQ(ierr);
          ierr = PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD),"subplot(1,2,2);[X,Y]=meshgrid(linspace(0,Lx,Nx),linspace(0,Ly,Ny));surf(X,Y,reshape(r,Nx,Ny)');title({['Random field'],['Time t= ',num2str(tm(2,1))]});shading interp;axis([0 1 0 1 -0.5 1]);axis square;xlabel('X');ylabel('Y');view(2);colorbar;set(gca,'fontsize', 16);pause(0.01);");CHKERRQ(ierr);
          ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s",output);CHKERRQ(ierr);
      }
      tm[1] = tm[1] + dt;
      ierr = VecCopy(u,uold);CHKERRQ(ierr);
      ierr = BuildR(r,&user);
      ierr = VecScale(r,PetscSqrtReal(dt));CHKERRQ(ierr);
      
      /* Euler scheme */
//      ierr = MatMultAdd(user.A,u,r,unew);CHKERRQ(ierr);
//      ierr = VecAXPY(unew,1.0,uold);CHKERRQ(ierr);
      
      /* Crank-Nicolson scheme */
      ierr = FormCNRHS(&user,uold,dt,rhs);
      ierr = VecAXPY(rhs,1.0,r);CHKERRQ(ierr);
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);
      ierr = KSPSetOperators(ksp,user.A,user.A);
      ierr = KSPSetFromOptions(ksp);
      ierr = VecDuplicate(rhs,&unew);
      ierr = KSPSolve(ksp,rhs,unew);

      ierr = VecCopy(unew,u);CHKERRQ(ierr);CHKERRQ(ierr);


//      if ( (i+1)%tout == 0)
//      {
//          ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//          char s1[10], s2[10];
//          sprintf(s1,"%s%d%s","u",(i+1)/tout,".m");
//          printf("%s\n",s1);
//          sprintf(s2,"%s%d","sol",(i+1)/tout);
//          printf("%s\n",s2);
//          ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,s1,&viewfile);CHKERRQ(ierr);
//          ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
//          ierr = PetscObjectSetName((PetscObject)u,s2);CHKERRQ(ierr);
//          ierr = VecView(u,viewfile);CHKERRQ(ierr);
//          ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
//          ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
//       }
      
  }
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&unew);CHKERRQ(ierr);
  ierr = VecDestroy(&uold);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr); /* rhs vector for Crank-Nicolson scheme*/
  ierr = KSPDestroy(&ksp);               /* KSP solver for Crank-Nicolson scheme*/

  ierr = PetscFinalize();
  return ierr;
}


PetscErrorCode BuildA(AppCtx *user)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i,j;
  PetscReal      hx,hy,sx,sy;
  PetscViewer    viewfile;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(user->da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
  hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscInt    nc = 0;
      MatStencil  row,col[5];
      PetscScalar val[5];
      row.i = i; row.j = j;
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
      } else {
        col[nc].i = i-1; col[nc].j = j;   val[nc++] = sx;
        col[nc].i = i+1; col[nc].j = j;   val[nc++] = sx;
        col[nc].i = i;   col[nc].j = j-1; val[nc++] = sy;
        col[nc].i = i;   col[nc].j = j+1; val[nc++] = sy;
        col[nc].i = i;   col[nc].j = j;   val[nc++] = -2*sx - 2*sy;
      }
      ierr = MatSetValuesStencil(user->A,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(user->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"fdmat.m",&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)user->A,"mat");CHKERRQ(ierr);
  ierr = MatView(user->A,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
       // to check pattern in Matlab >>fdmat;spy(mat)
  
  PetscFunctionReturn(0);
}

PetscErrorCode BuildCN(AppCtx *user, PetscScalar dt)
{
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscInt       i,j;
    PetscReal      hx,hy,sx,sy;
    PetscViewer    viewfile;
    
    PetscFunctionBeginUser;
    ierr = DMDAGetLocalInfo(user->da,&info);CHKERRQ(ierr);
    hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
    hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            PetscInt    nc = 0;
            MatStencil  row,col[5];
            PetscScalar val[5];
            row.i = i; row.j = j;
            if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
                col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
            } else {
                col[nc].i = i-1; col[nc].j = j;   val[nc++] = -.5*sx*dt;
                col[nc].i = i+1; col[nc].j = j;   val[nc++] = -.5*sx*dt;
                col[nc].i = i;   col[nc].j = j-1; val[nc++] = -.5*sy*dt;
                col[nc].i = i;   col[nc].j = j+1; val[nc++] = -.5*sy*dt;
                col[nc].i = i;   col[nc].j = j;   val[nc++] = 1 + sx*dt + sy*dt;
            }
            ierr = MatSetValuesStencil(user->A,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(user->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(user->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"fdmat.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)user->A,"mat");CHKERRQ(ierr);
    ierr = MatView(user->A,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
    // to check pattern in Matlab >>fdmat;spy(mat)
    
    PetscFunctionReturn(0);
}

PetscErrorCode FormCNRHS(AppCtx *user, Vec U, PetscScalar dt, Vec RHS)
{
    PetscScalar    *u, *rhs;
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscInt       i,j;
    PetscReal      hx,hy,sx,sy;
    PetscViewer    viewfile;
    
    PetscFunctionBeginUser;
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    ierr = VecGetArray(RHS,&rhs);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(user->da,&info);CHKERRQ(ierr);
    hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
    hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
    
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
                rhs[info.mx*j+i] = u[info.mx*j+i];
            } else {
                rhs[info.mx*j+i] = u[info.mx*j+i]                            * (1-sx*dt-sy*dt)
                               + ( u[info.mx*j+(i+1)] + u[info.mx*j+(i-1)] ) * .5*sx*dt
                               + ( u[info.mx*(j+1)+i] + u[info.mx*(j-1)+i] ) * .5*sy*dt;
            }
        }
    }

    ierr = VecRestoreArray(RHS,&rhs);
    ierr = VecAssemblyBegin(RHS);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(RHS);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"rhsvec.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)U,"vec");CHKERRQ(ierr);
    ierr = VecView(RHS,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U,void* ptr)
{
    AppCtx         *user=(AppCtx*)ptr;
    PetscReal      c=user->c;
    PetscErrorCode ierr;
    PetscInt       i,j,xs,ys,xm,ym,Mx,My;
    PetscScalar    **u;
    PetscReal      hx,hy,x,y,r;
    
    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    
    hx = 1.0/(PetscReal)(Mx-1);
    hy = 1.0/(PetscReal)(My-1);
    
    /* Get pointers to vector data */
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    
    /* Get local grid boundaries */
    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    
    /* Compute function over the locally owned part of the grid */
    for (j=ys; j<ys+ym; j++) {
        y = j*hy;
        for (i=xs; i<xs+xm; i++) {
            x = i*hx;
            r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
            if (r < .7) u[j][i] = PetscExpReal(c*r*r*r);
            else u[j][i] = 0.0;
        }
    }
    
    /* Restore vectors */
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode BuildR(Vec R,AppCtx* user)
{
    DM             cda;
    DMDACoor2d     **coors;
    Vec            global;
    PetscInt       xs, xm, ys, ym, N2, Nx, Ny;
    PetscInt       i, j, i0, j0, i1, j1;
    PetscReal      mu=0.0, sigma=0.1; /* sigma here stands for noise strength */
    PetscReal      lc=2.0;
//    PetscReal      lx=0.2, ly=0.1;
    PetscScalar    **Cov;
    PetscScalar    **U, **V, *S;
    PetscScalar    *W;
    PetscScalar    *r, tmp;           /* r: vector issued from random field by KL expansion */
    PetscScalar    x1, y1, x0, y0, rr;
    PetscErrorCode ierr;

  PetscFunctionBeginUser;
    
    /*------------------------------------------------------------------------
     Access coordinate field
     ---------------------------------------------------------------------*/
    ierr = DMDAGetInfo(user->da,PETSC_IGNORE,&Nx,&Ny,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    N2=Nx*Ny;
    
    /* Get pointers to vector data */
    ierr = VecGetArray(R,&r);CHKERRQ(ierr);

    /* Get local grid boundaries */
    ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(user->da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinates(user->da,&global);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
    
    /// allocate covariance matrix, its SVD associates and random vector r
    ierr = PetscMalloc1(N2,&Cov);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&Cov[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) Cov[i] = Cov[i-1]+N2;
    
    ierr = PetscMalloc1(N2,&U);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&U[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) U[i] = U[i-1]+N2;
    
    ierr = PetscMalloc1(N2,&V);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2*N2,&V[0]);CHKERRQ(ierr);
    for (i=1; i<N2; i++) V[i] = V[i-1]+N2;
    
    ierr = PetscMalloc1(N2,&S);CHKERRQ(ierr);
    
/* Compute covariance function over the locally owned part of the grid */
    
        //        printf("\ntest coordinates:\n");
    for (j0=ys; j0<ys+ym; j0++)
    {for (i0=xs; i0<xs+xm; i0++)
    {
        //        printf("coord[%d][%d]", j0, i0);
        //        printf(".x=%1.2f  ", coors[j0][i0].x);
        //        printf(".y=%1.2f\n", coors[j0][i0].y);
        x0=coors[j0][i0].x;
        y0=coors[j0][i0].y;
        for (j1=ys; j1<ys+ym; j1++)
        {for (i1=xs; i1<xs+xm; i1++)
        {x1=coors[j1][i1].x;
            y1=coors[j1][i1].y;
//            rr = PetscAbsReal(x1-x0)/lx+PetscAbsReal(y1-y0)/ly; //Seperable Exp
//            rr = PetscSqrtReal(PetscPowReal(x1-x0,2)+PetscPowReal(y1-y0,2))/lc; //Exp
            rr = (PetscPowReal(x1-x0,2)+PetscPowReal(y1-y0,2))/(2 * lc * lc); //Gaussian
            Cov[j0*xm+i0][j1*xm+i1]=PetscExpReal(-rr);
        }
        }
    }
    }
    ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
    
    //    //   Print covariance matrix Cov (before adding weights)
    //    printf("Cov\n");
    //    for (i = 0; i < N2; i++)
    //    {
    //        for (j = 0; j < N2; j++) printf("%6.2f", Cov[i][j]);
    //        printf("\n");
    //    }
    
/* Approximate the covariance integral operator via collocation and vertex-based quadrature */
    
    // allocate quadrature weights W along the diagonal
    ierr = PetscMalloc1(N2,&W);CHKERRQ(ierr);
    
    // fill the weights (trapezoidal rule in 2d uniform mesh)
    // fill the first and the last
    W[0]=1; W[Nx-1]=1; W[N2-Nx]=1; W[N2-1]=1;
    for (i=1; i<Nx-1; i++) {W[i] = 2; W[N2-Nx+i]=2;}
    // fill in between
    for (i=0; i<Nx; i++)
    {
        for (j=1; j<Ny-1; j++) W[j*Nx+i] = 2.0 * W[i];
    }
    
    //    // Print W before scaling
    //    printf("\nW\n");
    //    for (i = 0; i < N2; i++) printf("%f\n", W[i]);
    
    // Scale W
    for (i = 0; i < N2; i++) W[i] = W[i] * ((user->Lx)*(user->Ly))/(4*(Nx-1)*(Ny-1));
    
    //    // Print W after scaling
    //    printf("\nW\n");
    //    for (i = 0; i < N2; i++) printf("%f\n", W[i]);
    
    /* Combine W with covariance matrix Cov to form covariance operator K
     K = sqrt(W) * Cov * sqrt(W) (modifed to be symmetric)             */
    for (i=0; i<N2; i++)
    {
        for (j=0; j<N2; j++) Cov[i][j] = Cov[i][j] * PetscSqrtReal(W[i]) * PetscSqrtReal(W[j]);
    }
    
//    //  Print the approximation of covariance operator K (modified to be symmetric)
//        printf("\nK = sqrt(W) * Cov * sqrt(W)\n");
//        for (i = 0; i < N2; i++)
//        {
//            for (j = 0; j < N2; j++) printf("%6.2f", Cov[i][j]);
//            printf("\n");
//        }
    
/* Use SVD to decompose the PSD matrix K to get its eigen decomposition */
    svd(Cov,U,V,S,N2);
    
    // Recover eigenvectors by divding sqrt(W)
    for (i = 0; i < N2; i++)
    {
        for (j = 0; j < N2; j++) U[i][j] = U[i][j] / PetscSqrtReal(W[j]);
    }
    
//    // Print decomposition results: K=USV'
//        printf("\nK=USV':\n");
//
//     // Print eigenvalues
//        printf("\nEigenvalues S (in non-increasing order)\n");
//        for (j = 0; j < N2; j++)
//        {
//            printf("%8.4f", S[j]);
//            printf("\n");
//        }
//
//     // Print eigenvectors W^(-1/2) * U
//        printf("\nIts corresponding eigenvectors (columns)\n");
//        for (i = 0; i < N2; i++)
//        {
//            for (j = 0; j < N2; j++) printf("%6.2f", U[i][j]);
//            printf("\n");
//        }
    
/* Generate normal random numbers by transforming from the uniform one */
    PetscScalar *rndu, *rndn;
    ierr = PetscMalloc1(N2,&rndu);CHKERRQ(ierr);
    ierr = PetscMalloc1(N2,&rndn);CHKERRQ(ierr);
    //  PetscRandom rnd;
    //    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);
    /* force imaginary part of random number to always be zero; thus obtain reproducible results with real and complex numbers */
    //    ierr = PetscRandomSetInterval(rnd,0.0,1.0);CHKERRQ(ierr);
    //    ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
    //    ierr = PetscRandomGetValue(rnd,&rndu);CHKERRQ(ierr);
    
    time_t t;
    srand((unsigned) time(&t));rand();//initialize random number generator in C
//        PetscScalar mean = 0;
    for (i = 0; i < N2; i++)
    {
        rndu[i] = (PetscScalar) rand()/RAND_MAX;
        rndn[i] = ltqnorm(rndu[i]);// transform from uniform(0,1) to normal(0,1) by N = norminv(U)
        //        printf("\nuniform random sample= %f\n",rndu[i]);
        //        printf("normal random sample= %f\n",rndn[i]);
//                mean = mean + rndu[i];
    }
//        mean = mean/N2;
//        printf("%f\n",mean);
    
/* Do KL expansion by combining the above eigen decomposition and normal random numbers*/
    for (i = 0; i < N2; i++)
    {
        tmp=0.0;
        for (j = 0; j < N2; j++) tmp = tmp + U[i][j] * PetscSqrtReal(S[j]) * rndn[j];
        r[i] = mu + sigma * tmp;
    }
    for (j=0; j<Ny; j++)
        {for (i=0; i<Nx; i++)
            {
                if (i == 0 || j == 0 || i == Nx-1 || j == Ny-1) {
                    r[Nx*j+i] = 0.0;}
            }
        }
    // Pring the random vector r issued from random field by KL expansion
//    printf("\nRandom vector r issued from random field by KL expansion\n");
//    for (i = 0; i < N2; i++) printf("%6.8f\n", r[i]);
    /* plot r in Matlab:
       >> Lx=1;Ly=1;Nx=8;Ny=8;[X,Y]=meshgrid(linspace(0,Lx,Nx),linspace(0,Ly,Ny));
       surf(X,Y,reshape(r,Nx,Ny)');shading interp;view(2);colorbar; */

    //    ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
    ierr = PetscFree(Cov);CHKERRQ(ierr);
    ierr = PetscFree(U);CHKERRQ(ierr);
    ierr = PetscFree(V);CHKERRQ(ierr);
    ierr = PetscFree(S);CHKERRQ(ierr);

  /* Restore vectors */
    ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

void svd(PetscScalar **A_input, PetscScalar **U, PetscScalar **V, PetscScalar *S, PetscInt n)
/* svd.c: Perform a singular value decomposition A = USV' of square matrix.
 *
 * Input: The A_input matrix must has n rows and n columns.
 * Output: The product are U, S and V(not V').
           The S vector returns the singular values. */
{
  PetscInt  i, j, k, EstColRank = n, RotCount = n, SweepCount = 0,
    slimit = (n<120) ? 30 : n/4;
  PetscScalar eps = 1e-15, e2 = 10.0*n*eps*eps, tol = 0.1*eps, vt, p, x0,
    y0, q, r, c0, s0, d1, d2;
  PetscScalar *S2;
  PetscScalar **A;
    
  PetscMalloc1(n,&S2);
  for (i=1; i<n; i++) S2[i] = S2[i-1]+n;
    
  PetscMalloc1(2*n,&A);
  PetscMalloc1(2*n*n,&A[0]);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;
    
  for (i=0; i<n; i++)
    {
        A[i] = malloc(n * sizeof(PetscScalar));
        A[n+i] = malloc(n * sizeof(PetscScalar));
        for (j=0; j<n; j++)
        {
            A[i][j]   = A_input[i][j];
            A[n+i][j] = 0.0;
        }
        A[n+i][i] = 1.0;
    }
  while (RotCount != 0 && SweepCount++ <= slimit) {
    RotCount = EstColRank*(EstColRank-1)/2;
    for (j=0; j<EstColRank-1; j++)
      for (k=j+1; k<EstColRank; k++) {
        p = q = r = 0.0;
        for (i=0; i<n; i++) {
          x0 = A[i][j]; y0 = A[i][k];
          p += x0*y0; q += x0*x0; r += y0*y0;
        }
        S2[j] = q; S2[k] = r;
        if (q >= r) {
          if (q<=e2*S2[0] || fabs(p)<=tol*q)
            RotCount--;
          else {
            p /= q; r = 1.0-r/q; vt = sqrt(4.0*p*p+r*r);
            c0 = sqrt(0.5*(1.0+r/vt)); s0 = p/(vt*c0);
            for (i=0; i<2*n; i++) {
              d1 = A[i][j]; d2 = A[i][k];
              A[i][j] = d1*c0+d2*s0; A[i][k] = -d1*s0+d2*c0;
            }
          }
        } else {
          p /= r; q = q/r-1.0; vt = sqrt(4.0*p*p+q*q);
          s0 = sqrt(0.5*(1.0-q/vt));
          if (p<0.0) s0 = -s0;
          c0 = p/(vt*s0);
          for (i=0; i<2*n; i++) {
            d1 = A[i][j]; d2 = A[i][k];
            A[i][j] = d1*c0+d2*s0; A[i][k] = -d1*s0+d2*c0;
          }
        }
      }
    while (EstColRank>2 && S2[EstColRank-1]<=S2[0]*tol+tol*tol) EstColRank--;
      }
  if (SweepCount > slimit)
    printf("Warning: Reached maximum number of sweeps (%d) in SVD routine...\n"
       ,slimit);
    for (i=0; i<n; i++) S[i] = PetscSqrtReal(S2[i]);
    for (i=0; i<n; i++)
    {
        for (j=0; j<n; j++)
        {
            U[i][j] = A[i][j]/S[j];
            V[i][j] = A[n+i][j];
        }
    }
    PetscFree(S2);
    PetscFree(A);
}

/*
 * Lower tail quantile for standard normal distribution function.
 *
 * This function returns an approximation of the inverse cumulative
 * standard normal distribution function.  I.e., given P, it returns
 * an approximation to the X satisfying P = Pr{Z <= X} where Z is a
 * random variable from the standard normal distribution.
 *
 * The algorithm uses a minimax approximation by rational functions
 * and the result has a relative error whose absolute value is less
 * than 1.15e-9.
 *
 * Author:      Peter John Acklam
 * Time-stamp:  2002-06-09 18:45:44 +0200
 * E-mail:      jacklam@math.uio.no
 * WWW URL:     http://www.math.uio.no/~jacklam
 *
 * C implementation adapted from Peter's Perl version
 */

#include <math.h>
#include <errno.h>

/* Coefficients in rational approximations. */
static const double a[] =
{
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
};

static const double b[] =
{
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
};

static const double c[] =
{
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
};

static const double d[] =
{
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
};

#define LOW 0.02425
#define HIGH 0.97575

double ltqnorm(double p)
{
    double q, r;
    
    errno = 0;
    
    if (p < 0 || p > 1)
    {
        errno = EDOM;
        return 0.0;
    }
    else if (p == 0)
    {
        errno = ERANGE;
        return -HUGE_VAL /* minus "infinity" */;
    }
    else if (p == 1)
    {
        errno = ERANGE;
        return HUGE_VAL /* "infinity" */;
    }
    else if (p < LOW)
    {
        /* Rational approximation for lower region */
        q = sqrt(-2*log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    else if (p > HIGH)
    {
        /* Rational approximation for upper region */
        q  = sqrt(-2*log(1-p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    else
    {
        /* Rational approximation for central region */
        q = p - 0.5;
        r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    }
}


