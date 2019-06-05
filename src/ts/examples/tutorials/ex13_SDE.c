

static char help[] = "Time-dependent PDE in 2d. Simplified from ex7.c for illustrating how to use TS on a structured domain. \n";
/*
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1;
   At t=0: u(x,y) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

    mpiexec -n 2 ./ex13 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
    mpiexec -n 1 ./ex13 -snes_fd_color -ts_monitor_draw_solution
    mpiexec -n 2 ./ex13 -ts_type sundials -ts_monitor 
*/

#include <petscdm.h>
#include <petscdmda.h>
#include "svd.h"

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscReal c;
  Mat       A;
  DM        da;
} AppCtx;

extern PetscErrorCode BuildA(AppCtx*);
extern PetscErrorCode BuildCov(Vec, AppCtx*);

int main(int argc,char **argv)
{
    //test svd.c
    PetscInt m,n,rows=4;
    PetscScalar **A, *S2;
    //allocate test matrix A and vector s to store the square of singular values
    A = (PetscScalar **) malloc(2 * rows * sizeof(double*));
    S2 = (PetscScalar *) malloc(rows * sizeof(double));
    
    //Set test matrix A
    printf("test svd:\n\nMatrix A\n");
    for (m = 0; m < 2 * rows; m++)
    {
        A[m] = malloc(2 * rows * sizeof(double));
        for (n = 0; n < rows; n++)
            A[m][n] = m + n;
    }

    //Print test matrix A
    for (m=0; m < rows; m++)
    {
        for (n=0; n < rows; n++)
            printf("%1.0f ", A[m][n]);
        printf("\n");
    }

    //Do SVD
    svd(A,S2,rows);
    
    //Print Results
    printf("\nA=USV'\n");
    //Print S2
    printf("\nThe square of singular values S2\n");
    for (n = 0; n < rows; n++)
    {
        printf("%6.2f", S2[n]);
        printf("\n");
    }

    printf("\nUS\n");
    for (m = 0; m < rows; m++)
    {
        for (n = 0; n < rows; n++)
            printf("%6.2f", A[m][n]);
        printf("\n");
    }

    printf("\nV\n");
    for (m = rows; m < 2 * rows; m++)
    {
        for (n = 0; n < rows; n++)
            printf("%6.2f", A[m][n]);
        printf("\n");
    }
    
  Vec            u;                  /* solution vector */
  PetscErrorCode ierr;
  DM             cda;
  DMDACoor2d     **coors;
  Vec            global;
  AppCtx         user;              /* user-defined work context */
  PetscInt       N=4;
  PetscScalar    **Cov, sigma, lc;
 
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,N,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  /*------------------------------------------------------------------------
    Access coordinate field
    ---------------------------------------------------------------------*/
  PetscInt Lx=2, Ly=3, xs,xm,ys,ym,ix,iy;
  PetscScalar x1,y1,x0,y0,rr;
  PetscInt N2, i,j;

  sigma=1.0;
  lc=5;
   
  N2=N*N;
  /// allocate covariance matrix
  ierr = PetscMalloc1(N2,&Cov);CHKERRQ(ierr);
  ierr = PetscMalloc1(N2*N2,&Cov[0]);CHKERRQ(ierr);
  for (i=1; i<N2; i++) Cov[i] = Cov[i-1]+N2;

  ierr = DMDASetUniformCoordinates(user.da,0.0,Lx,0.0,Ly,0.0,0.0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(user.da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(user.da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(user.da,&global);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
             printf("\ntest coordinates:\n");
  for (iy=ys; iy<ys+ym; iy++)
     {for (ix=xs; ix<xs+xm; ix++)
            {
             printf("coord[%d][%d]", iy, ix);
             printf(".x=%f  ", coors[iy][ix].x);
             printf(".y=%f\n", coors[iy][ix].y);
             x0=coors[iy][ix].x;
             y0=coors[iy][ix].y;
             for (j=ys; j<ys+ym; j++)
                {for (i=xs; i<xs+xm; i++)
                    {x1=coors[j][i].x;
                     y1=coors[j][i].y;
                     rr=PetscPowReal((x1-x0),2)+PetscPowReal((y1-y0),2);
                     Cov[iy*ym+ix][j*xm+i]=sigma*PetscExpReal(-rr/lc);
                    }
                }
            }
     }
  ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
  // Print covariance matrix
    printf("\ncovariance matrix:\n");
    for (m = 0; m < N2; m++)
    {
        for (n = 0; n < N2; n++)
            printf("%6.2f", Cov[m][n]);
        printf("\n");
    }
 
  /* Initialize user application context */
  user.c = -30.0;

 
  /* Set Matrix */
  ierr = DMSetMatType(user.da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,&user.A);CHKERRQ(ierr);
  
  ierr = BuildA(&user);
//  ierr = MatView(user.A,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //ierr = FormInitialSolution(user.da,u,&user);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
//  ierr = VecDestroy(&global);CHKERRQ(ierr); //error occurs if turning on
//  ierr = DMDestroy(&cda);CHKERRQ(ierr);     //error occurs if turnung on
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);

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
  ierr = PetscObjectSetName((PetscObject)user->A,"fdmat");CHKERRQ(ierr);
  ierr = MatView(user->A,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
       // to check pattern in Matlab >>fdmat;spy(fdmat)
  
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------- */
PetscErrorCode BuildCov(Vec U,AppCtx* user)
{
  PetscReal      c=user->c;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    **u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(user->da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(user->da,U,&u);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) u[j][i] = PetscExpReal(c*r*r*r);
      else u[j][i] = 0.0;
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(user->da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void svd(PetscScalar **A, PetscScalar *S2, PetscInt n)
/* svd.c: Perform a singular value decomposition A = USV' of square matrix.
 *
 * This routine has been adapted with permission from a Pascal implementation
 * (c) 1988 J. C. Nash, "Compact numerical methods for computers", Hilger 1990.
 * The A matrix must be pre-allocated with 2n rows and n columns. On calling
 * the matrix to be decomposed is contained in the first n rows of A. On return
 * the n first rows of A contain the product US and the lower n rows contain V
 * (not V'). The S2 vector returns the square of the singular values.
 *
 * (c) Copyright 1996 by Carl Edward Rasmussen. */
{
  int  i, j, k, EstColRank = n, RotCount = n, SweepCount = 0,
    slimit = (n<120) ? 30 : n/4;
  double eps = 1e-15, e2 = 10.0*n*eps*eps, tol = 0.1*eps, vt, p, x0,
    y0, q, r, c0, s0, d1, d2;

  for (i=0; i<n; i++) { for (j=0; j<n; j++) A[n+i][j] = 0.0; A[n+i][i] = 1.0; }
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
}
