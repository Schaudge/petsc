
static char help[] = "Implements a simple tensor contraction over 3D fields with 2nd order tensors\n\n";

/*
-tao_type test -tao_test_gradient
    Not yet tested in parallel
*/
/*
   Concepts: Tensor contractions
   Concepts: BLAS DGEMM
   Processors: 1
*/

/* ------------------------------------------------------------------------

   This program uses non-batched BLAS calls to compute
       v = (A\kron B \kron C) u
   where A,B,C are N by N and u is N by N by N

   To be extended to variable sizes, and ported to GPU BLAS

  ------------------------------------------------------------------------- */
#include <petscdmda.h>
#include <stdio.h>
#include <petscblaslapack.h>

typedef struct
{
  PetscInt N;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode PetscAllocateEl3d(PetscReal ****, AppCtx *);
extern PetscErrorCode PetscDestroyEl3d(PetscReal ****, AppCtx *);
extern PetscErrorCode PetscAllocateEl2d(PetscReal ***, AppCtx *);
extern PetscErrorCode PetscDestroyEl2d(PetscReal ***, AppCtx *);
extern PetscErrorCode PetscTens3dSEM(PetscReal ***, PetscReal ***, PetscReal ***, PetscReal ****, PetscReal ****, PetscReal **,AppCtx *appctx);

PetscErrorCode PetscAllocateEl3d(PetscReal ****AA, AppCtx *appctx)
{
  PetscReal ***A, **B, *C;
  PetscErrorCode ierr;
  PetscInt Nl, Nl2, Nl3;
  PetscInt ix, iy;

  PetscFunctionBegin;
  Nl = appctx->N;
  Nl2 = Nl * Nl;
  Nl3 = Nl2 * Nl;

  ierr = PetscMalloc1(Nl, &A);   CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl2, &B);  CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl3, &C);  CHKERRQ(ierr);

  for (ix = 0; ix < Nl; ix++)
  {
    A[ix] = B + ix * Nl;
  }
  for (ix = 0; ix < Nl; ix++)
  {
    for (iy = 0; iy < Nl; iy++)
    {
      A[ix][iy] = C + ix * Nl * Nl + iy * Nl;
    }
  }
  /* Fill up the 3d array as a 1d array */
  for (ix = 0; ix < Nl3; ix++)
  {
    C[ix] = ix;
  }
  *AA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDestroyEl3d(PetscReal ****AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0][0]);   CHKERRQ(ierr);
  ierr = PetscFree((*AA)[0]);  CHKERRQ(ierr);
  ierr = PetscFree(*AA);  CHKERRQ(ierr);

  *AA = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode PetscAllocateEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscReal **A;
  PetscErrorCode ierr;
  PetscInt Nl, Nl2, i;

  PetscFunctionBegin;
  Nl = appctx->N;
  Nl2 = Nl * Nl;

  ierr = PetscMalloc1(Nl, &A);   CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl2, &A[0]);  CHKERRQ(ierr);
  for (i = 1; i < Nl; i++)
    A[i] = A[i - 1] + Nl;

  *AA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDestroyEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);   CHKERRQ(ierr);
  ierr = PetscFree(*AA);  CHKERRQ(ierr);
  *AA = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode PetscTens3dSEM(PetscReal ***A, PetscReal ***B, PetscReal ***C, PetscReal ****ulb, PetscReal ****out, PetscReal **alphavec, AppCtx *appctx)
{
  PetscErrorCode ierr;
  PetscInt Nl, Nl2;
  PetscInt jx;
  PetscScalar *temp1, *temp2;
  PetscScalar ***wrk1, ***wrk2, ***wrk3;
  PetscReal beta;
 
  PetscFunctionBegin;
  Nl = appctx->N;
  Nl2 = Nl * Nl;
  
  beta=0.0;
  
  PetscAllocateEl3d(&wrk1, appctx);
  PetscAllocateEl3d(&wrk2, appctx);
  PetscAllocateEl3d(&wrk3, appctx);

  BLASgemm_("T", "N", &Nl, &Nl2, &Nl, alphavec[0], A[0][0], &Nl, ulb[0][0][0], &Nl, &beta, &wrk1[0][0][0], &Nl);
  for (jx = 0; jx < Nl; jx++)
  {
    temp1 = &wrk1[0][0][0] + jx * Nl2;
    temp2 = &wrk2[0][0][0] + jx * Nl2;

    BLASgemm_("N", "N", &Nl, &Nl, &Nl, alphavec[0]+1, temp1, &Nl, B[0][0], &Nl, &beta, temp2, &Nl);
  }

  BLASgemm_("N", "N", &Nl2, &Nl, &Nl, alphavec[0]+2, &wrk2[0][0][0], &Nl2, C[0][0], &Nl, &beta, &wrk3[0][0][0], &Nl2);

  *out = wrk3;

  PetscFunctionReturn(0);
}
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  AppCtx appctx; /* user-defined application context */
  PetscScalar ***wrk1, ***wrk2, ***wrk3, ***outfun;
  PetscScalar **A, **B, **C;
  PetscScalar ***ulb;
  PetscScalar *temp1, *temp2;
  PetscInt jx, jy, jz;
  PetscInt Nl, Nl2;
  PetscScalar beta;
  PetscReal *alphavec;

  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr)
    return ierr;
  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &appctx.N, NULL);
  CHKERRQ(ierr);

  appctx.N = 5;
  Nl = appctx.N;
  Nl2 = Nl * Nl;
  /*
     Allocate arrays
  */
  PetscAllocateEl3d(&ulb, &appctx);
  PetscAllocateEl3d(&wrk1, &appctx);
  PetscAllocateEl3d(&wrk2, &appctx);
  PetscAllocateEl3d(&wrk3, &appctx);
  PetscAllocateEl3d(&outfun, &appctx);
  PetscAllocateEl2d(&A, &appctx);
  PetscAllocateEl2d(&B, &appctx);
  PetscAllocateEl2d(&C, &appctx);

  beta = 0.0;
  
  srand(time(NULL));
 
  for (jx = 0; jx < Nl; jx++)
  {
    for (jy = 0; jy < Nl; jy++)
    {
      for (jz = 0; jz < Nl; jz++)
      {

        ulb[jx][jy][jz] = (double)rand() / RAND_MAX * 2.0 - 1.0;
      }
      A[jx][jy] = (double)rand() / RAND_MAX * 2.0 - 1.0;
      B[jx][jy] = (double)rand() / RAND_MAX * 2.0 - 1.0;
      C[jx][jy] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
  }
  
  ierr = PetscMalloc1(3, &alphavec);
  alphavec[0]=1.0;
  alphavec[1]=1.0;
  alphavec[2]=1.0;

  BLASgemm_("T", "N", &Nl, &Nl2, &Nl, &alphavec[0], &A[0][0], &Nl, &ulb[0][0][0], &Nl, &beta, &wrk1[0][0][0], &Nl);
  for (jx = 0; jx < Nl; jx++)
  {
    temp1 = &wrk1[0][0][0] + jx * Nl2;
    temp2 = &wrk2[0][0][0] + jx * Nl2;

    BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alphavec[1], temp1, &Nl, &B[0][0], &Nl, &beta, temp2, &Nl);
  }

  BLASgemm_("N", "N", &Nl2, &Nl, &Nl, &alphavec[2], &wrk2[0][0][0], &Nl2, &C[0][0], &Nl, &beta, &wrk3[0][0][0], &Nl2);

  PetscTens3dSEM(&A, &B, &C, &ulb, &outfun, &alphavec, &appctx);
  
  FILE *fp;
  fp = fopen("arrays.m", "w");
  fprintf(fp, "N=%d;\n", Nl);
  fclose(fp);
  fp = fopen("arrays.m", "a");
  for (jx = 0; jx < Nl; jx++)
  {
    for (jy = 0; jy < Nl; jy++)
    {
      for (jz = 0; jz < Nl; jz++)
      {
        fprintf(fp, "ulb(%d,%d,%d)=%.10f;", jx + 1, jy + 1, jz + 1, ulb[jx][jy][jz]);
        fprintf(fp, "wrk1(%d,%d,%d)=%.10f;", jx + 1, jy + 1, jz + 1, wrk1[jx][jy][jz]);
        fprintf(fp, "wrk2(%d,%d,%d)=%.10f;", jx + 1, jy + 1, jz + 1, wrk2[jx][jy][jz]);
        fprintf(fp, "wrk3(%d,%d,%d)=%.10f;", jx + 1, jy + 1, jz + 1, wrk3[jx][jy][jz]);
        fprintf(fp, "outfun(%d,%d,%d)=%.10f;", jx + 1, jy + 1, jz + 1, outfun[jx][jy][jz]);
        //fprintf(fp, "temp2(%d,%d)=%.10f;", jy + 1, jz + 1, temp2[jy * Nl + jz]);
        fprintf(fp, "A(%d,%d)=%.10f;", jy + 1, jz + 1, A[jy][jz]);
        fprintf(fp, "B(%d,%d)=%.10f;", jy + 1, jz + 1, B[jy][jz]);
        fprintf(fp, "C(%d,%d)=%.10f;", jy + 1, jz + 1, C[jy][jz]);
        //fprintf(fp, "temp1(%d,%d)=%.10f;\n", jy + 1, jz + 1, temp1[jy * Nl + jz]);
      }
    }
  }
  fclose(fp);
  //memset(myarray, 0, N*sizeof(*myarray));

  PetscDestroyEl3d(&ulb, &appctx);

  PetscDestroyEl3d(&wrk1, &appctx);
  PetscDestroyEl3d(&wrk2, &appctx);
  PetscDestroyEl3d(&wrk3, &appctx);

  PetscDestroyEl2d(&A, &appctx);
  PetscDestroyEl2d(&B, &appctx);
  PetscDestroyEl2d(&C, &appctx);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: BLAS

   test:
     requires: !single
     args: -N 10

   test:
     suffix: N
     requires: !single
     args: -N 10

TEST*/
