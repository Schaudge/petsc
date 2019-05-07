
static char help[] = "Solves a simple data assimilation problem with one dimensional advection diffusion equation using TSAdjoint\n\n";

/*
-tao_type test -tao_test_gradient
    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Concepts: adjoints
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional advection-diffusion equation),
       u_t = mu*u_xx - a u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimilation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   The operators are discretized with the spectral element method

  ------------------------------------------------------------------------- */

#include <petscdt.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petscdmda.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt  n;                /* number of nodes */
  PetscReal *nodes;           /* GLL nodes */
  PetscReal *weights;         /* GLL weights */
} PetscGLL;

typedef struct
{
  PetscInt N;                /* grid points per elements*/
  PetscInt Ex;               /* number of elements */
  PetscInt Ey;               /* number of elements */
  PetscReal tol_L2, tol_max; /* error norms */
  PetscInt steps;            /* number of timesteps */
  PetscReal Tend;            /* endtime */
  PetscReal mu;              /* viscosity */
  PetscReal Lx;              /* total length of domain */
  PetscReal Ly;              /* total length of domain */
  PetscReal Lex;
  PetscReal Ley;
  PetscInt lenx;
  PetscInt leny;
} PetscParam;

typedef struct
{
  PetscScalar u, v; /* wind speed */
} Field;

typedef struct
{
  Vec grid; /* total grid */
  Vec grad;
  Vec ic;
  Vec curr_sol;
  Vec pass_sol;
  Vec true_solution; /* actual initial conditions for the final solution */
} PetscData;

typedef struct
{
  Vec grid; /* total grid */
  Vec mass; /* mass matrix for total integration */
  //Mat         stiff;             /* stifness matrix */
  //Mat         keptstiff;
  //Mat         grad;
  PetscGLL gll;
} PetscSEMOperators;

typedef struct
{
  DM da; /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam param;
  PetscData dat;
  TS ts;
  PetscReal initial_dt;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec, AppCtx *);
extern PetscErrorCode TrueSolution(Vec, AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode MyMatMult(Mat, Vec, Vec);
extern PetscErrorCode MyMatMultTransp(Mat, Vec, Vec);
extern PetscErrorCode PetscAllocateEl2d(PetscReal ***, AppCtx *);
extern PetscErrorCode PetscPointWiseMult(PetscInt Nl,PetscScalar *A, PetscScalar *B, PetscScalar *out);


int main(int argc, char **argv)
{
  AppCtx appctx; /* user-defined application context */
  Vec u;         /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt xs, xm, ys, ym, ix, iy;
  PetscInt indx, indy, m, nn;
  PetscReal x, y;
  Field **bmass;
  DMDACoor2d **coors;
  Vec global, loc;
  DM cda;
  PetscInt jx, jy;
  PetscViewer viewfile;
  Mat H_shell;
  //MatNullSpace   nsp;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr)
    return ierr;

  /*initialize parameters */
  appctx.param.N = 4;      /* order of the spectral element */
  appctx.param.Ex = 2;     /* number of elements */
  appctx.param.Ey = 2;     /* number of elements */
  appctx.param.Lx = 4.0;   /* length of the domain */
  appctx.param.Ly = 4.0;   /* length of the domain */
  appctx.param.mu = 0.005; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend = 2.0;

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &appctx.param.N, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ex", &appctx.param.Ex, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ey", &appctx.param.Ey, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Tend", &appctx.param.Tend, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-mu", &appctx.param.mu, NULL);
  CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx / appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly / appctx.param.Ey;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //ierr = PetscGLLCreate(appctx.param.N, PETSCGLL_VIA_LINEARALGEBRA, &appctx.SEMop.gll);
  ierr = PetscMalloc2(appctx.param.N,&appctx.SEMop.gll.nodes,appctx.param.N,&appctx.SEMop.gll.weights);CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(appctx.param.N,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights);CHKERRQ(ierr);
  appctx.SEMop.gll.n = appctx.param.N;
  CHKERRQ(ierr);

  appctx.param.lenx = appctx.param.Ex * (appctx.param.N - 1);
  appctx.param.leny = appctx.param.Ey * (appctx.param.N - 1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, appctx.param.lenx, appctx.param.leny, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &appctx.da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);
  CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);
  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 0, "u");
  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 1, "v");
  CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da, &u);
  CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.ic);
  CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.true_solution);
  CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.SEMop.mass);
  CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.curr_sol);
  CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.pass_sol);
  CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx.da, &xs, &ys, NULL, &xm, &ym, NULL);
  CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid */
  xs = xs / (appctx.param.N - 1);
  xm = xm / (appctx.param.N - 1);
  ys = ys / (appctx.param.N - 1);
  ym = ym / (appctx.param.N - 1);

  VecSet(appctx.SEMop.mass, 0.0);

  DMCreateLocalVector(appctx.da, &loc);
  ierr = DMDAVecGetArray(appctx.da, loc, &bmass);
  CHKERRQ(ierr);

  /*
     Build mass over entire mesh (multi-elemental) 

  */

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (jx = 0; jx < appctx.param.N; jx++)
    {
      for (iy = ys; iy < ys + ym; iy++)
      {
        for (jy = 0; jy < appctx.param.N; jy++)
        {
          x = (appctx.param.Lex / 2.0) * (appctx.SEMop.gll.nodes[jx] + 1.0) + appctx.param.Lex * ix;
          y = (appctx.param.Ley / 2.0) * (appctx.SEMop.gll.nodes[jy] + 1.0) + appctx.param.Ley * iy;
          indx = ix * (appctx.param.N - 1) + jx;
          indy = iy * (appctx.param.N - 1) + jy;
          bmass[indy][indx].u += appctx.SEMop.gll.weights[jx] * appctx.SEMop.gll.weights[jy] * .25 * appctx.param.Ley * appctx.param.Lex;
          bmass[indy][indx].v += appctx.SEMop.gll.weights[jx] * appctx.SEMop.gll.weights[jy] * .25 * appctx.param.Ley * appctx.param.Lex;
        }
      }
    }
  }

  DMDAVecRestoreArray(appctx.da, loc, &bmass);
  CHKERRQ(ierr);
  DMLocalToGlobalBegin(appctx.da, loc, ADD_VALUES, appctx.SEMop.mass);
  DMLocalToGlobalEnd(appctx.da, loc, ADD_VALUES, appctx.SEMop.mass);

  DMDASetUniformCoordinates(appctx.da, 0.0, appctx.param.Lx, 0.0, appctx.param.Ly, 0.0, 0.0);
  DMGetCoordinateDM(appctx.da, &cda);

  DMGetCoordinates(appctx.da, &global);
  VecSet(global, 0.0);
  DMDAVecGetArray(cda, global, &coors);

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (jx = 0; jx < appctx.param.N - 1; jx++)
    {
      for (iy = ys; iy < ys + ym; iy++)
      {
        for (jy = 0; jy < appctx.param.N - 1; jy++)
        {
          x = (appctx.param.Lex / 2.0) * (appctx.SEMop.gll.nodes[jx] + 1.0) + appctx.param.Lex * ix - 2.0;
          y = (appctx.param.Ley / 2.0) * (appctx.SEMop.gll.nodes[jy] + 1.0) + appctx.param.Ley * iy - 2.0;
          indx = ix * (appctx.param.N - 1) + jx;
          indy = iy * (appctx.param.N - 1) + jy;
          coors[indy][indx].x = x;
          coors[indy][indx].y = y;
        }
      }
    }
  }
  DMDAVecRestoreArray(cda, global, &coors);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "tomesh.m", &viewfile);
  CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB);
  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global, "grid");
  ierr = VecView(global, viewfile);
  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass, "mass");
  ierr = VecView(appctx.SEMop.mass, viewfile);
  CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);
  ierr = PetscViewerDestroy(&viewfile);
  CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);
  CHKERRQ(ierr);

  /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  ierr = TSCreate(PETSC_COMM_WORLD, &appctx.ts);
  CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx.ts, TS_NONLINEAR);
  CHKERRQ(ierr);
  ierr = TSSetType(appctx.ts, TSRK);
  CHKERRQ(ierr);
  ierr = TSSetDM(appctx.ts, appctx.da);
  CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts, 0.0);
  CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx.ts, appctx.initial_dt);
  CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx.ts, appctx.param.steps);
  CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx.ts, appctx.param.Tend);
  CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx.ts, TS_EXACTFINALTIME_MATCHSTEP);
  CHKERRQ(ierr);

  VecGetLocalSize(u, &m);
  VecGetSize(u, &nn);

  MatCreateShell(PETSC_COMM_WORLD, m, m, nn, nn, &appctx, &H_shell);
  MatShellSetOperation(H_shell, MATOP_MULT, (void (*)(void))MyMatMult);
  MatShellSetOperation(H_shell, MATOP_MULT_TRANSPOSE, (void (*)(void))MyMatMultTransp);

  /* attach the null space to the matrix, this probably is not needed but does no harm */

  /*
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(H_shell,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,H_shell,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  */

  ierr = TSSetTolerances(appctx.ts, 1e-7, NULL, 1e-7, NULL);
  CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);
  CHKERRQ(ierr);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx.ts, &appctx.initial_dt);
  CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts, H_shell, H_shell, RHSJacobian, &appctx);
  CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts, NULL, RHSFunction, &appctx);
  CHKERRQ(ierr);

  ierr = InitialConditions(appctx.dat.ic, &appctx);
  CHKERRQ(ierr);

  ierr = VecDuplicate(appctx.dat.ic, &appctx.dat.curr_sol);
  CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic, appctx.dat.curr_sol);
  CHKERRQ(ierr);
  ierr = TSSolve(appctx.ts, appctx.dat.curr_sol);
  CHKERRQ(ierr);

  /*
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"sol2d.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.obj,"sol");
    ierr = VecView(appctx.dat.obj,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"ic");
    ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
 */
  ierr = TSSetSaveTrajectory(appctx.ts);
  CHKERRQ(ierr);

  /* Set Objective and Initial conditions for the problem and compute Objective function (evolution of true_solution to final time */

  ierr = TrueSolution(appctx.dat.true_solution, &appctx);
  CHKERRQ(ierr);

  ierr = VecDestroy(&u);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);
  CHKERRQ(ierr);
 // ierr = PetscGLLDestroy(&appctx.SEMop.gll);
  ierr = PetscFree2(appctx.SEMop.gll.nodes,appctx.SEMop.gll.weights);CHKERRQ(ierr);
  CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);
  CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);
  CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  ierr = PetscFinalize();
  return ierr;
}

/* --------------------------------------------------------------------- 

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u, AppCtx *appctx)
{
  PetscScalar tt;
  Field **s;
  PetscErrorCode ierr;
  PetscInt i, j;
  DM cda;
  Vec global;
  DMDACoor2d **coors;

  ierr = DMDAVecGetArray(appctx->da, u, &s);
  CHKERRQ(ierr);

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

  tt = 0.0;
  for (i = 0; i < appctx->param.lenx; i++)
  {
    for (j = 0; j < appctx->param.leny; j++)
    {
      s[j][i].u = PetscExpScalar(-appctx->param.mu * tt) * (PetscCosScalar(0.5 * PETSC_PI * coors[j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10.0;
      s[j][i].v = PetscExpScalar(-appctx->param.mu * tt) * (PetscSinScalar(0.5 * PETSC_PI * coors[j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10.0;
    }
  }
  ierr = DMDAVecRestoreArray(appctx->da, u, &s);
  CHKERRQ(ierr);

  return 0;
}

/*
         InitialConditions() computes the initial conditions for the begining of the Tao iterations

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode TrueSolution(Vec u, AppCtx *appctx)
{
  PetscScalar tt;
  Field **s;
  PetscErrorCode ierr;
  PetscInt i, j;
  DM cda;
  Vec global;
  DMDACoor2d **coors;

  ierr = DMDAVecGetArray(appctx->da, u, &s);
  CHKERRQ(ierr);

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

  tt = 4.0 - appctx->param.Tend;
  for (i = 0; i < appctx->param.lenx; i++)
  {
    for (j = 0; j < appctx->param.leny; j++)
    {
      s[j][i].u = PetscExpScalar(-appctx->param.mu * tt) * (PetscCosScalar(0.5 * PETSC_PI * coors[j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10.0;
      s[j][i].v = PetscExpScalar(-appctx->param.mu * tt) * (PetscSinScalar(0.5 * PETSC_PI * coors[j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10.0;
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, u, &s);
  CHKERRQ(ierr);
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
  return 0;
}

PetscErrorCode PetscAllocateEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscReal **A;
  PetscErrorCode ierr;
  PetscInt Nl, Nl2, i;

  PetscFunctionBegin;
  Nl = appctx->param.N;
  Nl2 = appctx->param.N * appctx->param.N;

  ierr = PetscMalloc1(Nl, &A);
  CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl2, &A[0]);
  CHKERRQ(ierr);
  for (i = 1; i < Nl; i++)
    A[i] = A[i - 1] + Nl;

  *AA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDestroyEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);
  CHKERRQ(ierr);
  ierr = PetscFree(*AA);
  CHKERRQ(ierr);
  *AA = NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *appctx = (AppCtx *)ctx;
  PetscScalar **wrk3, **wrk1, **wrk2, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar **stiff, **mass, **grad;
  PetscScalar **ulb, **vlb;
  const Field **ul;
  Field **outl;
  PetscInt ix, iy, jx, jy, indx, indy;
  PetscInt xs, xm, ys, ym, Nl, Nl2;
  DM cda;
  Vec uloc, outloc, global;
  DMDACoor2d **coors;
  PetscScalar alpha, beta;
  PetscInt inc;

  PetscFunctionBegin;

  //ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll, &stiff);
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);
 

  /* unwrap local vector for the input solution */
  /* globalin, the global array
     uloc, the local array
     ul, the pointer to uloc*/

  DMCreateLocalVector(appctx->da, &uloc);

  DMGlobalToLocalBegin(appctx->da, globalin, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, globalin, INSERT_VALUES, uloc);

  ierr = DMDAVecGetArrayRead(appctx->da, uloc, &ul);
  CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da, &outloc);

  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);

  //ierr = DMDAVecGetArray(appctx->da,gradloc,&outgrad);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);
  CHKERRQ(ierr);
  Nl = appctx->param.N;

  //DMCreateGlobalVector(appctx->da,&gradgl);

  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);

  inc = 1;

  /*
     Initialize work arrays
  */
  PetscAllocateEl2d(&ulb, appctx);
  PetscAllocateEl2d(&vlb, appctx);
  PetscAllocateEl2d(&wrk1, appctx);
  PetscAllocateEl2d(&wrk2, appctx);
  PetscAllocateEl2d(&wrk3, appctx);
  PetscAllocateEl2d(&wrk4, appctx);
  PetscAllocateEl2d(&wrk5, appctx);
  PetscAllocateEl2d(&wrk6, appctx);
  PetscAllocateEl2d(&wrk7, appctx);

  alpha = 1.0;
  beta = 0.0;
  Nl2 = Nl * Nl;

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (iy = ys; iy < ys + ym; iy++)
    {
      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)

        {
          ulb[jy][jx] = 0.0;
          vlb[jy][jx] = 0.0;
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          ulb[jy][jx] = ul[indy][indx].u;
          vlb[jy][jx] = ul[indy][indx].v;
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy)u=W2 (u_yy)
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2. / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk2[0][0], &Nl);

      //second product (K_xx x B) u=W3 (u_xx)
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk3[0][0], &inc, &wrk2[0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2

      // for the v component now
      //first product (B x K_yy)v=W3
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2.0 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      //second product (K_xx x B)v=W4
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk4[0][0], &inc, &wrk3[0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3

      //now the gradient operator for u
      // first (D_x x B) u =W4 this multiples u
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      // first (B x D_y) u =W5 this mutiplies v
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      //now the gradient operator for v
      // first (D_x x B) v =W6 this multiples u
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      // first (B x D_y) v =W7 this mutiplies v
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk7[0][0], &Nl);

      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)
        {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;

          outl[indy][indx].u += appctx->param.mu * (wrk2[jy][jx]) + vlb[jy][jx] * wrk5[jy][jx] + ulb[jy][jx] * wrk4[jy][jx]; //+rr.u*mass[jy][jx];
          outl[indy][indx].v += appctx->param.mu * (wrk3[jy][jx]) + ulb[jy][jx] * wrk6[jy][jx] + vlb[jy][jx] * wrk7[jy][jx]; //+rr.v*mass[jy][jx];
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc, &ul);
  CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);

  VecSet(globalout, 0.0);
  DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, globalout);
  DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, globalout);

  VecScale(globalout, -1.0);

  ierr = VecPointwiseDivide(globalout, globalout, appctx->SEMop.mass);
  CHKERRQ(ierr);

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

  //ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll, &stiff);
  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);

  PetscDestroyEl2d(&ulb, appctx);
  PetscDestroyEl2d(&vlb, appctx);
  PetscDestroyEl2d(&wrk1, appctx);
  PetscDestroyEl2d(&wrk2, appctx);
  PetscDestroyEl2d(&wrk3, appctx);
  PetscDestroyEl2d(&wrk4, appctx);
  PetscDestroyEl2d(&wrk5, appctx);
  PetscDestroyEl2d(&wrk6, appctx);
  PetscDestroyEl2d(&wrk7, appctx);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMult"

PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
{
  AppCtx *appctx;

  const Field **ul, **uj;
  Field **outl;
  PetscScalar **stiff, **mass, **grad;
  PetscScalar **wrk1, **wrk2, **wrk3, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar **ulb, **vlb, **ujb, **vjb;
  PetscInt Nl, Nl2, inc;
  PetscInt xs, ys, xm, ym, ix, iy, jx, jy, indx, indy;
  PetscErrorCode ierr;
  Vec uloc, outloc, ujloc;
  PetscScalar alpha, beta;

  MatShellGetContext(H, &appctx);

  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);
 
  /* unwrap local vector for the input solution */
  DMCreateLocalVector(appctx->da, &uloc);
  /*   in, the global array
     uloc, the local array
     ul, the pointer to uloc*/

  DMGlobalToLocalBegin(appctx->da, in, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, in, INSERT_VALUES, uloc);

  DMDAVecGetArrayRead(appctx->da, uloc, &ul);
  CHKERRQ(ierr);

  /* unwrap the vector for the forward variable */
  /* appctx->dat.pass_sol, the global array
     ujloc, the local array
     uj, the pointer to uloc*/
  DMCreateLocalVector(appctx->da, &ujloc);

  DMGlobalToLocalBegin(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);
  DMGlobalToLocalEnd(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);

  DMDAVecGetArrayRead(appctx->da, ujloc, &uj);
  CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da, &outloc);
  VecSet(outloc, 0.0);

  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);
  CHKERRQ(ierr);
  Nl = appctx->param.N;

  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);

  /*
     Initialize work arrays
  */
  PetscAllocateEl2d(&ulb, appctx);
  PetscAllocateEl2d(&vlb, appctx);
  PetscAllocateEl2d(&ujb, appctx);
  PetscAllocateEl2d(&vjb, appctx);
  PetscAllocateEl2d(&wrk1, appctx);
  PetscAllocateEl2d(&wrk2, appctx);
  PetscAllocateEl2d(&wrk3, appctx);
  PetscAllocateEl2d(&wrk4, appctx);
  PetscAllocateEl2d(&wrk5, appctx);
  PetscAllocateEl2d(&wrk6, appctx);
  PetscAllocateEl2d(&wrk7, appctx);

  alpha = 1.0;
  beta = 0.0;
  Nl2 = Nl * Nl;
  inc = 1;
  for (ix = xs; ix < xs + xm; ix++)
  {
    for (iy = ys; iy < ys + ym; iy++)
    {
      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)

        {
          ulb[jy][jx] = 0.0;
          ujb[jy][jx] = 0.0;
          vlb[jy][jx] = 0.0;
          vjb[jy][jx] = 0.0;
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          ujb[jy][jx] = uj[indy][indx].u;
          vjb[jy][jx] = uj[indy][indx].v;
          ulb[jy][jx] = ul[indy][indx].u;
          vlb[jy][jx] = ul[indy][indx].v;
          wrk4[jy][jx] = 0.0;
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy) u=W2 (u_yy)
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2. / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk2[0][0], &Nl);

      //second product (K_xx x B) u=W3 (u_xx)
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk3[0][0], &inc, &wrk2[0][0], &inc); //I freed wrk3 and saved the lalplacian in wrk2

      // for the v component now
      //first product (B x K_yy) v=W3
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2.0 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      //second product (K_xx x B) v=W4
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk4[0][0], &inc, &wrk3[0][0], &inc); //I freed wrk4 and saved the lalplacian in wrk3

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      //now the gradient operator for u
      // first (D_x x B) wu the term ujb.(D_x x B) wu
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk4[0][0], &ujb[0][0], &wrk4[0][0]);

      // (D_x x B) u the term ulb.(D_x x B) u
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ujb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &ulb[0][0], &wrk5[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk4

      // first (B x D_y) wu the term vjb.(B x D_x) wu
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &vjb[0][0], &wrk5[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk4

      // first (B x D_y) u the term vlb.(B x D_x) u !!!
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &ujb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &vlb[0][0], &wrk5[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk4

      //////////////////////////////////// the second equation

      // (D_x x B) wv the term ujb.(D_x x B) wv
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &ujb[0][0], &wrk5[0][0]);

      // (D_x x B) v the term ulb.(D_x x B) v !!!
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vjb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk6[0][0], &ulb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      // first (B x D_y) v the term vlb.(B x D_x) v
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &vjb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk6[0][0], &vlb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      // first (B x D_y) wv the term vjb.(B x D_x) wv
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk6[0][0], &vjb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)
        {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;

          outl[indy][indx].u += appctx->param.mu * (wrk2[jy][jx]) + wrk4[jy][jx];
          outl[indy][indx].v += appctx->param.mu * (wrk3[jy][jx]) + wrk5[jy][jx];

          //printf("outl[%d][%d]=%0.15f\n", indx,indy, outl[indy][indx]);
        }
      }
    }
  }
  //ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  //DMDAVecRestoreArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);
  //DMDAVecRestoreArrayRead(appctx->da,ujloc,&uj);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da, in, &uloc);
  CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da, appctx->dat.pass_sol, &ujloc);
  CHKERRQ(ierr);

  VecSet(out, 0.0);

  DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, out);
  DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, out);

  VecScale(out, -1.0);
  ierr = VecPointwiseDivide(out, out, appctx->SEMop.mass);
  CHKERRQ(ierr);

  //VecView(out,0);
  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);


  PetscDestroyEl2d(&ulb, appctx);
  PetscDestroyEl2d(&vlb, appctx);
  PetscDestroyEl2d(&wrk1, appctx);
  PetscDestroyEl2d(&wrk2, appctx);
  PetscDestroyEl2d(&wrk3, appctx);
  PetscDestroyEl2d(&wrk4, appctx);
  PetscDestroyEl2d(&wrk5, appctx);
  PetscDestroyEl2d(&wrk6, appctx);
  PetscDestroyEl2d(&wrk7, appctx);

  return (0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMultTransp"

PetscErrorCode MyMatMultTransp(Mat H, Vec in, Vec out)
{
  AppCtx *appctx;

  const Field **ul, **uj;
  Field **outl;
  PetscScalar **stiff, **mass, **grad;
  PetscScalar **wrk1, **wrk2, **wrk3, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar **ulb, **vlb, **ujb, **vjb;
  PetscInt Nl, Nl2, inc;
  PetscInt xs, ys, xm, ym, ix, iy, jx, jy, indx, indy;
  PetscErrorCode ierr;
  Vec uloc, outloc, ujloc, incopy;
  PetscScalar alpha, beta;

  MatShellGetContext(H, &appctx);

 
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);
 

  VecDuplicate(in, &incopy);
  VecCopy(in, incopy);
  ierr = VecPointwiseDivide(incopy, in, appctx->SEMop.mass);
  CHKERRQ(ierr);

  /* unwrap local vector for the input solution */
  /* incopy, the global array (copy needed cause it needs rescaling by mass matrix)
     uloc, the local array
     ul, the pointer to uloc*/

  DMCreateLocalVector(appctx->da, &uloc);

  DMGlobalToLocalBegin(appctx->da, incopy, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, incopy, INSERT_VALUES, uloc);

  DMDAVecGetArrayRead(appctx->da, uloc, &ul);
  CHKERRQ(ierr);

  /* unwrap the vector for the forward variable */
  /* appctx->dat.pass_sol, the global array
     ujloc, the local array
     uj, the pointer to uloc*/
  DMCreateLocalVector(appctx->da, &ujloc);

  DMGlobalToLocalBegin(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);
  DMGlobalToLocalEnd(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);

  DMDAVecGetArrayRead(appctx->da, ujloc, &uj);
  CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da, &outloc);
  VecSet(outloc, 0.0);

  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);
  CHKERRQ(ierr);
  Nl = appctx->param.N;

  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);

  /*
     Initialize work arrays
  */
  PetscAllocateEl2d(&ulb, appctx);
  PetscAllocateEl2d(&vlb, appctx);
  PetscAllocateEl2d(&ujb, appctx);
  PetscAllocateEl2d(&vjb, appctx);
  PetscAllocateEl2d(&wrk1, appctx);
  PetscAllocateEl2d(&wrk2, appctx);
  PetscAllocateEl2d(&wrk3, appctx);
  PetscAllocateEl2d(&wrk4, appctx);
  PetscAllocateEl2d(&wrk5, appctx);
  PetscAllocateEl2d(&wrk6, appctx);
  PetscAllocateEl2d(&wrk7, appctx);

  alpha = 1.0;
  beta = 0.0;
  Nl2 = Nl * Nl;
  inc = 1;
  for (ix = xs; ix < xs + xm; ix++)
  {
    for (iy = ys; iy < ys + ym; iy++)
    {
      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)

        {
          ulb[jy][jx] = 0.0;
          ujb[jy][jx] = 0.0;
          vlb[jy][jx] = 0.0;
          vjb[jy][jx] = 0.0;
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          ujb[jy][jx] = uj[indy][indx].u;
          vjb[jy][jx] = uj[indy][indx].v;
          ulb[jy][jx] = ul[indy][indx].u;
          vlb[jy][jx] = ul[indy][indx].v;
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy)u=W2 (u_yy)
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2. / appctx->param.Ley;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk2[0][0], &Nl);

      //second product (K_xx x B) u=W3 (u_xx)
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk3[0][0], &inc, &wrk2[0][0], &inc); //I freed wrk3 and saved the lalplacian in wrk2

      // for the v component now
      //first product (B x K_yy)v=W3
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2.0 / appctx->param.Ley;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      //second product (K_xx x B)v=W4
      alpha = 2.0 / appctx->param.Lex;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk4[0][0], &inc, &wrk3[0][0], &inc); //I freed wrk4 and saved the lalplacian in wrk3

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      //now the gradient operator for u
      // first (D_x x B) wu the term (D_x x B) wu.ujb

      PetscPointWiseMult(Nl2, &ulb[0][0], &ujb[0][0], &wrk6[0][0]);

      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &wrk6[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      // (D_x x B) u the term ulb.(D_x x B) u
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ujb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &ulb[0][0], &wrk5[0][0]); //same term

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk4

      // first (B x D_y) wu the term vjb.(B x D_x) wu

      PetscPointWiseMult(Nl2, &ulb[0][0], &vjb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &wrk6[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk4

      // (D_x x B) v the term vlb.(D_x x B) v
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vjb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk5[0][0], &vlb[0][0], &wrk5[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk5[0][0], &inc, &wrk4[0][0], &inc); // saving in wrk5

      //////////////////////////////////// the second equation

      // (D_x x B) wv the term ujb.(D_x x B) wv

      PetscPointWiseMult(Nl2, &vlb[0][0], &ujb[0][0], &wrk7[0][0]);
      alpha = appctx->param.Lex / 2.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &wrk7[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      // first (B x D_y) u the term ulb.(B x D_x) u       /////////same term B
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &ujb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk6[0][0], &ulb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      // first (B x D_y) v the term vjb.(B x D_x) wv
      PetscPointWiseMult(Nl2, &vlb[0][0], &vjb[0][0], &wrk7[0][0]);
      alpha = 1.0;
      BLASgemm_("T", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &wrk7[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      // first (B x D_y) wv the term vlb.(B x D_x) v
      alpha = 1.0;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &vjb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2.0;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      PetscPointWiseMult(Nl2, &wrk6[0][0], &vlb[0][0], &wrk6[0][0]);

      alpha = 1.0;
      BLASaxpy_(&Nl2, &alpha, &wrk6[0][0], &inc, &wrk5[0][0], &inc); // saving in wrk5

      for (jx = 0; jx < appctx->param.N; jx++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)
        {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          outl[indy][indx].u += appctx->param.mu * (wrk2[jy][jx]) + wrk4[jy][jx];
          outl[indy][indx].v += appctx->param.mu * (wrk3[jy][jx]) + wrk5[jy][jx];
        }
      }
    }
  }

  //ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(appctx->da,ujloc,&uj);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);
  CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da, in, &uloc);
  CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da, appctx->dat.pass_sol, &ujloc);
  CHKERRQ(ierr);

  DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, out);
  DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, out);

  VecScale(out, -1.0);
  //ierr = VecPointwiseDivide(out,out,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n,appctx->SEMop.gll.nodes,appctx->SEMop.gll.weights,&mass);CHKERRQ(ierr);


  PetscDestroyEl2d(&ulb, appctx);
  PetscDestroyEl2d(&vlb, appctx);
  PetscDestroyEl2d(&wrk1, appctx);
  PetscDestroyEl2d(&wrk2, appctx);
  PetscDestroyEl2d(&wrk3, appctx);
  PetscDestroyEl2d(&wrk4, appctx);
  PetscDestroyEl2d(&wrk5, appctx);
  PetscDestroyEl2d(&wrk6, appctx);
  PetscDestroyEl2d(&wrk7, appctx);

  return (0);
}

PetscErrorCode PetscPointWiseMult(PetscInt Nl, PetscScalar *A, PetscScalar *B, PetscScalar *out) 
  {                                                              
    PetscErrorCode ierr; 
    PetscInt i;

    for (i=0; i<Nl; i++)
     { out[i]=A[i]*B[i];
     } 
   
   PetscFunctionReturn(0);
  }

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec globalin, Mat A, Mat B, void *ctx)
{
  AppCtx *appctx = (AppCtx *)ctx;
  PetscFunctionBegin;

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  VecCopy(globalin, appctx->dat.pass_sol);

  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 

   test:
     suffix: cn
     requires: !single
     args: -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 

TEST*/
