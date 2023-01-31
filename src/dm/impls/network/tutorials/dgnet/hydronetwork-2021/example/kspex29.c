/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
  This uses structured grids to build linear PDEs. Then solves the linear system resulted from PDE of heat diffusion in 2D.

  U = U(x,y),
  -div grad U = 0.0, 0.0 < x,y < 12.0
  u(x,0) = 100.0, u(x,12) = 0.0, u(0,y) = 100.0, u(12,y) = 100.0

  Contributed by Getnet Betrie. Modified after: petsc/src/ksp/ksp/examples/tutorials/ex29.c.
*/

static char help[] = "Solves homogeneous heat diffusion in 2D using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHS(KSP, Vec, void *);

typedef struct {
  PetscScalar oo, pp;
} UserContext;

int main(int argc, char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  PetscErrorCode ierr;
  Vec            b, x;
  PetscBool      flg;

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);
  CHKERRQ(ierr);
  ierr = DMSetUp(da);
  CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0, 12, 0, 12, 0, 0);
  CHKERRQ(ierr);
  ierr = DMDASetFieldName(da, 0, "Pressure");
  CHKERRQ(ierr);

  user.oo = 1.0;
  user.pp = 1.0;

  ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user);
  CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp, ComputeMatrix, &user);
  CHKERRQ(ierr);
  ierr = KSPSetDM(ksp, da);
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);
  CHKERRQ(ierr);
  ierr = KSPSolve(ksp, NULL, NULL);
  CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp, &x);
  CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp, &b);
  CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-view_x", &flg, NULL);
  CHKERRQ(ierr);
  if (flg) {
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);
  }

  ierr = DMDestroy(&da);
  CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i, j, mx, my, xm, ym, xs, ys;
  PetscScalar    Hx;
  PetscScalar  **array;
  DM             da;

  PetscFunctionBeginUser;
  ierr = VecSet(b, 0.0);
  CHKERRQ(ierr);
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  CHKERRQ(ierr);
  Hx   = 12.0 / (PetscReal)(mx - 1);
  ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
  CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &array);
  CHKERRQ(ierr);
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      if (j == 0 || i == 0) {
        array[j][i] = 100.0;
      } else if ((i * Hx) == 12.0) {
        array[j][i] = 100.0;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);
  CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i, j, mx, my, xm, ym, xs, ys;
  PetscScalar    v[5];
  PetscReal      Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  CHKERRQ(ierr);
  Hx    = 12.0 / (PetscReal)(mx - 1);
  Hy    = 12.0 / (PetscReal)(my - 1);
  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;
  ierr  = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
  CHKERRQ(ierr);
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;

      if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
        v[0] = 2.0 * (HxdHy + HydHx);
        ierr = MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES);
        CHKERRQ(ierr);
      } else {
        v[0]     = -HxdHy;
        col[0].i = i;
        col[0].j = j - 1;
        v[1]     = -HydHx;
        col[1].i = i - 1;
        col[1].j = j;
        v[2]     = 2.0 * (HxdHy + HydHx);
        col[2].i = i;
        col[2].j = j;
        v[3]     = -HydHx;
        col[3].i = i + 1;
        col[3].j = j;
        v[4]     = -HxdHy;
        col[4].i = i;
        col[4].j = j + 1;
        ierr     = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
