
static char help[] = "Solves a simple PDE inverse problem with two dimensional Burgers equation using TSAdjoint\n\n";

/* ------------------------------------------------------------------------

   This program solves the two-dimensional Heat equation,

   on the domain -2 <= x,y <= 2, with periodic boundary conditions

   The implementation is based on high order spectral element method
   using efficient vectorized BLAS calls

     ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscdt.h>
#include <petscdraw.h>
#include <petscdmda.h>
#include <petscblaslapack.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct
{
  PetscInt  n;        /* number of nodes */
  PetscReal *nodes;   /* GLL nodes */
  PetscReal *weights; /* GLL weights */
} PetscGLL;

typedef struct
{
  PetscInt  N;                /* grid points per elements*/
  PetscInt  Ex;               /* number of elements */
  PetscInt  Ey;               /* number of elements */
  PetscReal tol_L2, tol_max;  /* error norms */
  PetscInt  steps;            /* number of timesteps */
  PetscReal Tend;             /* endtime */
  PetscReal Tinit;            /* initial time */
  PetscReal mu;               /* viscosity */
  PetscReal Lx;               /* total length of domain */
  PetscReal Ly;               /* total length of domain */
  PetscReal Lex;
  PetscReal Ley;
  PetscInt  lenx;
  PetscInt  leny;
} PetscParam;

typedef struct
{
  PetscScalar u, v; /* wind speed */
} Field;

typedef struct
{
  Vec obj;           /* desired end state, that is the solution to the PDE at TEND */
  Vec grad;
  Vec ic;            /* this contains the intial conditions for the optimization and then the solution for the optimization at each optimization iteration */
  Vec curr_sol;
  Vec pass_sol;      /* this is the base for the Jacobian */
  Vec true_solution; /* actual initial conditions for the final solution */
} PetscData;

typedef struct
{
  Vec      grid;
  Vec      mass;         /* mass matrix for total integration */
  Mat      stiff;        /* stifness matrix */
  Mat      keptstiff;
  Mat      grad;
  Mat      opadd;
  PetscGLL gll;
} PetscSEMOperators;

typedef struct
{
  DM                da; /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec, AppCtx *);
extern PetscErrorCode TrueSolution(Vec, AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode MyMatMult(Mat, Vec, Vec);
extern PetscErrorCode PetscAllocateEl2d(PetscReal ***, AppCtx *);
extern PetscErrorCode PetscDestroyEl2d(PetscReal ***, AppCtx *);
extern PetscErrorCode PetscPointWiseMult(PetscInt, const PetscScalar *, const PetscScalar *, PetscScalar *);
extern PetscErrorCode InitializeSpectral(AppCtx *);

int main(int argc, char **argv)
{
  AppCtx         appctx; /* user-defined application context */
  Vec            u; /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       m, nn;
  Vec            global, loc;
  PetscViewer    viewfile;
  Mat            H_shell;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N     = 8;       /* order of the spectral element */
  appctx.param.Ex    = 6;       /* number of elements */
  appctx.param.Ey    = 6;       /* number of elements */
  appctx.param.Lx    = 4;       /* length of the domain */
  appctx.param.Ly    = 4;       /* length of the domain */
  appctx.param.mu    = 0.005Q;  /* diffusion coefficient */
  appctx.initial_dt  = 5e-3Q;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 1;//0.2;
  appctx.param.Tinit = 0;//1.0;

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &appctx.param.N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ex", &appctx.param.Ex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ey", &appctx.param.Ey, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Tend", &appctx.param.Tend, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-mu", &appctx.param.mu, NULL);CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx / appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly / appctx.param.Ey;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscMalloc2(appctx.param.N, &appctx.SEMop.gll.nodes, appctx.param.N, &appctx.SEMop.gll.weights);CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(appctx.param.N, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights);CHKERRQ(ierr);
  appctx.SEMop.gll.n = appctx.param.N;
  appctx.param.lenx = appctx.param.Ex * (appctx.param.N - 1);
  appctx.param.leny = appctx.param.Ey * (appctx.param.N - 1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, appctx.param.lenx, appctx.param.leny, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 0, "u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 1, "v");CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.pass_sol);CHKERRQ(ierr);
  
  ierr = InitializeSpectral(&appctx);  CHKERRQ(ierr);
  ierr = DMGetCoordinates(appctx.da, &global);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "meshout.m", &viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global, "grid");CHKERRQ(ierr);
  ierr = VecView(global, viewfile);
  ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass, "mass");CHKERRQ(ierr);
  ierr = VecView(appctx.SEMop.mass, viewfile);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  ierr = TSCreate(PETSC_COMM_WORLD, &appctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx.ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(appctx.ts, TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(appctx.ts, appctx.da);CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts, 0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx.ts, appctx.initial_dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx.ts, appctx.param.steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx.ts, appctx.param.Tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx.ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTolerances(appctx.ts, 1e-7, NULL, 1e-7, NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);CHKERRQ(ierr);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx.ts, &appctx.initial_dt);CHKERRQ(ierr);
  /*
     Use the initial condition of the PDE evaluated the grid points as the initial guess for the optimization problem 
  */
  ierr = InitialConditions(appctx.dat.ic, &appctx);CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic,appctx.dat.curr_sol);CHKERRQ(ierr);
  /* Create matrix-free matrices for applying Jacobian of RHS function */
  ierr = VecGetLocalSize(u, &m);CHKERRQ(ierr);
  ierr = VecGetSize(u, &nn);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD, m, m, nn, nn, &appctx, &H_shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H_shell, MATOP_MULT, (void (*)(void))MyMatMult);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts, H_shell, H_shell, RHSJacobian, &appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts, NULL, RHSFunction, &appctx);CHKERRQ(ierr);

  ierr = TSSolve(appctx.ts,appctx.dat.curr_sol);CHKERRQ(ierr);
  /*
  Vec   ref, wrk_vec, jac, vec_jac, vec_rhs, temp, vec_trans;
  Field     **s;
  PetscScalar vareps;
  PetscInt i;   
  PetscInt its=0;
  char var[15] ;
  
  ierr = VecDuplicate(appctx.dat.ic,&wrk_vec);CHKERRQ(ierr);
  //ierr = VecDuplicate(appctx.dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_jac);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_rhs);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_trans);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&ref);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&jac);CHKERRQ(ierr);

  ierr = VecCopy(appctx.dat.ic,appctx.dat.pass_sol);CHKERRQ(ierr);
  //VecSet(appctx.dat.pass_sol,1.0);

  RHSFunction(appctx.ts,0.0,appctx.dat.ic,ref,&appctx);
  //ierr = FormFunctionGradient(tao,wrk_vec,&wrk1,wrk2_vec,&appctx);CHKERRQ(ierr);
  // Note computed gradient is in wrk2_vec, original cost is in wrk1 
  ierr = VecZeroEntries(vec_jac);
  ierr = VecZeroEntries(vec_rhs); 
  vareps = 1e-05;
    for (i=0; i<2*(appctx.param.lenx*appctx.param.leny); i++) 
    //for (i=0; i<6; i++) 
     {
      its=its+1;
      ierr = VecCopy(appctx.dat.ic,wrk_vec); CHKERRQ(ierr); //reset J(eps) for each point
      ierr = VecZeroEntries(jac);
      VecSetValue(wrk_vec,i, vareps,ADD_VALUES);
      VecSetValue(jac,i,1.0,ADD_VALUES);
      RHSFunction(appctx.ts,0.0,wrk_vec,vec_rhs,&appctx);
      VecAXPY(vec_rhs,-1.0,ref);
      VecScale(vec_rhs, 1.0/vareps);
      MyMatMult(H_shell,jac,vec_jac);
      //VecView(jac,0);
      MyMatMultTransp(H_shell,jac,vec_trans);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"testjac.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"jac(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_jac,var);
    ierr = VecView(vec_jac,viewfile);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"rhs(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_rhs,var);
    ierr = VecView(vec_rhs,viewfile);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"trans(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_trans,var);
    ierr = VecView(vec_trans,viewfile);CHKERRQ(ierr);
    //ierr = PetscObjectSetName((PetscObject)ref,"ref");
    //ierr = VecView(ref,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
    //printf("test i %d length %d\n",its, appctx.param.lenx*appctx.param.leny);
    } 
//exit(1);

  //ierr = VecDuplicate(appctx.dat.ic,&uu);CHKERRQ(ierr);
  //ierr = VecCopy(appctx.dat.ic,uu);CHKERRQ(ierr);
  //MatView(H_shell,0);
  */
  /* attach the null space to the matrix, this is not needed for periodic BCs as here */

  /*
  MatNullSpace nsp;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(H_shell,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,H_shell,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  */
  /*  test code {
    Vec a,b;
    VecDuplicate(appctx.dat.ic,&a);
    VecDuplicate(appctx.dat.ic,&b);
    VecSetValue(a,0,1,INSERT_VALUES);
    VecAssemblyBegin(a);
    VecAssemblyEnd(a);
    ierr =  RHSFunction(appctx.ts, 0,a,b,&appctx);CHKERRQ(ierr);
    VecView(b,0);
   }*/

  
  /* Additional test, not needed
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"sol2d.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.obj,"sol");
    ierr = VecView(appctx.dat.obj,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"ic");
    ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
 */
  ierr = TSSetSaveTrajectory(appctx.ts);CHKERRQ(ierr);

  /* The solution to the continous PDE optimization problem evaluted at the discrete grid points */
  /* In the limit as one refines the mesh one hopes the TAO solution converges to this value */
  ierr = TrueSolution(appctx.dat.true_solution, &appctx);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);
  ierr = VecDestroy(&appctx.dat.pass_sol);
  ierr = VecDestroy(&loc);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);
  ierr = MatDestroy(&H_shell);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);

  ierr = PetscFree2(appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights);CHKERRQ(ierr); 
  ierr = PetscFinalize();
  return ierr;
}
/*
Initialize Spectral grid and mass matrix
*/
PetscErrorCode InitializeSpectral(AppCtx *appctx)
{
  PetscErrorCode ierr;
  DM cda;
  DMDACoor2d **coors;
  PetscInt       xs, xm, ys, ym, ix, iy, jx, jy;
  PetscInt       indx, indy;
  PetscReal      x, y;
  Field          **bmass;
  Vec            global, loc;
  
  PetscFunctionBegin;
  
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid */
  xs = xs / (appctx->param.N - 1);
  xm = xm / (appctx->param.N - 1);
  ys = ys / (appctx->param.N - 1);
  ym = ym / (appctx->param.N - 1);

  ierr = DMCreateLocalVector(appctx->da, &loc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, loc, &bmass);CHKERRQ(ierr);

  /*
     Build mass over entire mesh
  */

  for (ix = xs; ix < xs + xm; ix++) {
    for (jx = 0; jx < appctx->param.N; jx++) {
      for (iy = ys; iy < ys + ym; iy++) {
        for (jy = 0; jy < appctx->param.N; jy++) {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          bmass[indy][indx].u += appctx->SEMop.gll.weights[jx] * appctx->SEMop.gll.weights[jy] * appctx->param.Ley * appctx->param.Lex/4;
          bmass[indy][indx].v += appctx->SEMop.gll.weights[jx] * appctx->SEMop.gll.weights[jy] * appctx->param.Ley * appctx->param.Lex/4;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(appctx->da, loc, &bmass);CHKERRQ(ierr);
  ierr = VecSet(appctx->SEMop.mass, 0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, loc, ADD_VALUES, appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, loc, ADD_VALUES, appctx->SEMop.mass);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(appctx->da, 0, appctx->param.Lx, 0, appctx->param.Ly, 0, 0);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(appctx->da, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(appctx->da, &global);CHKERRQ(ierr);
  ierr = VecSet(global, 0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda, global, &coors);CHKERRQ(ierr);

  for (ix = xs; ix < xs + xm; ix++) {
    for (jx = 0; jx < appctx->param.N - 1; jx++) {
      for (iy = ys; iy < ys + ym; iy++) {
        for (jy = 0; jy < appctx->param.N - 1; jy++) {
          x = (appctx->param.Lex / 2) * (appctx->SEMop.gll.nodes[jx] + 1) + appctx->param.Lex * ix - 2;
          y = (appctx->param.Ley / 2) * (appctx->SEMop.gll.nodes[jy] + 1) + appctx->param.Ley * iy - 2;
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          coors[indy][indx].x = x;
          coors[indy][indx].y = y;
        }
      }
    }
  }
  DMDAVecRestoreArray(cda, global, &coors);

  PetscFunctionReturn(0);
}
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

                       The routine TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u, AppCtx *appctx)
{
  Field **s;
  PetscErrorCode ierr;
  PetscInt i, j;
  DM cda;
  Vec global;
  DMDACoor2d **coors;
  PetscReal tt;

  PetscFunctionBegin; 
  ierr = DMDAVecGetArray(appctx->da, u, &s);
  CHKERRQ(ierr);
  tt=appctx->param.Tinit;

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

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

  PetscFunctionReturn(0);
}

/*
   TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function. 

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
  tt = appctx->param.Tend;
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
/* --------------------------------------------------------------------- */

PetscErrorCode PetscPointWiseMult(PetscInt Nl, const PetscScalar *A, const PetscScalar *B, PetscScalar *out)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < Nl; i++) {
    out[i] = A[i] * B[i];
  }
  PetscFunctionReturn(0);
}


/*
   This uses the explicit formula for the PDE solution to compute the solution at the grid points for any given time

   Input Parameters:
   t -  time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ContinuumSolution(PetscReal t, Vec obj, AppCtx *appctx)
{
  Field            **s;
  PetscErrorCode   ierr;
  PetscInt         i, j;
  DM               cda;
  Vec              global;
  const DMDACoor2d **coors;

  PetscFunctionBegin;
  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArrayRead(cda, global, &coors);
  ierr = DMDAVecGetArray(appctx->da, obj, &s);CHKERRQ(ierr);

  for (i = 0; i < appctx->param.lenx; i++) {
    for (j = 0; j < appctx->param.leny; j++) {
      s[j][i].u = PetscExpScalar(-appctx->param.mu * t) * (PetscCosScalar(PETSC_PI * coors[j][i].x/2) + PetscSinScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10;
      s[j][i].v = PetscExpScalar(-appctx->param.mu * t) * (PetscSinScalar(PETSC_PI * coors[j][i].x/2) + PetscCosScalar(0.5 * PETSC_PI * coors[j][i].y)) / 10;
    }
  }

  ierr = DMDAVecRestoreArrayRead(cda, global, &coors);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da, obj, &s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscAllocateEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscReal      **A;
  PetscErrorCode ierr;
  PetscInt       Nl, Nl2, i;

  PetscFunctionBegin;
  Nl = appctx->param.N;
  Nl2 = appctx->param.N * appctx->param.N;

  ierr = PetscMalloc1(Nl, &A);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nl2, &A[0]);CHKERRQ(ierr);
  for (i = 1; i < Nl; i++)
    A[i] = A[i - 1] + Nl;
  *AA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDestroyEl2d(PetscReal ***AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx *)ctx;
  PetscScalar    **wrk3, **wrk1, **wrk2, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar    **stiff, **mass, **grad;
  PetscScalar    **ulb, **vlb;
  const Field    **ul;
  Field          **outl;
  PetscInt       ix, iy, jx, jy, indx, indy;
  PetscInt       xs, xm, ys, ym, Nl, Nl2;
  Vec            uloc, outloc;
  PetscScalar    alpha, beta;
  PetscInt       inc;

  PetscFunctionBegin;
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  /* ul contains local array of input vector */
  ierr = DMCreateLocalVector(appctx->da, &uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx->da, globalin, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da, globalin, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);

  /* outl contains local array of output vector */
  ierr = DMCreateLocalVector(appctx->da, &outloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
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
  PetscAllocateEl2d(&wrk1, appctx);
  PetscAllocateEl2d(&wrk2, appctx);
  PetscAllocateEl2d(&wrk3, appctx);
  PetscAllocateEl2d(&wrk4, appctx);
  PetscAllocateEl2d(&wrk5, appctx);
  PetscAllocateEl2d(&wrk6, appctx);
  PetscAllocateEl2d(&wrk7, appctx);

  beta = 0;
  Nl2  = Nl * Nl;
  inc  = 1;
  for (ix = xs; ix < xs + xm; ix++) {
    for (iy = ys; iy < ys + ym; iy++) {
      for (jx = 0; jx < appctx->param.N; jx++) {
        for (jy = 0; jy < appctx->param.N; jy++) {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          ulb[jy][jx] = ul[indy][indx].u;
          vlb[jy][jx] = ul[indy][indx].v;
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy)u=W2 (u_yy)
      alpha = appctx->param.Lex / 2;
      //      PetscRealView(4,&mass[0][0],PETSC_VIEWER_STDOUT_WORLD);
      // PetscRealView(4,&ulb[0][0],PETSC_VIEWER_STDOUT_WORLD);
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      // PetscRealView(4,&wrk1[0][0],PETSC_VIEWER_STDOUT_WORLD);
      alpha = 2 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk2[0][0], &Nl);
      //PetscRealView(4,&stiff[0][0],PETSC_VIEWER_STDOUT_WORLD);
      //          PetscRealView(4,&wrk2[0][0],PETSC_VIEWER_STDOUT_WORLD);
                  
      //second product (K_xx x B) u=W3 (u_xx)
      alpha = 2 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      alpha = 1;
      BLASaxpy_(&Nl2, &alpha, &wrk3[0][0], &inc, &wrk2[0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2

      // for the v component now
      //first product (B x K_yy)v=W3
      alpha = appctx->param.Lex / 2;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      //second product (K_xx x B)v=W4
      alpha = 2 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      alpha = 1;
      BLASaxpy_(&Nl2, &alpha, &wrk4[0][0], &inc, &wrk3[0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3

      //      PetscRealView(4,&wrk2[0][0],PETSC_VIEWER_STDOUT_WORLD);
      //PetscRealView(4,&wrk3[0][0],PETSC_VIEWER_STDOUT_WORLD);

      //now the gradient operator for u
      // first (D_x x B) u =W4 this multiples u
      alpha = appctx->param.Lex / 2;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      // first (B x D_y) u =W5 this mutiplies v
      alpha = 1;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk5[0][0], &Nl);

      //now the gradient operator for v
      // first (D_x x B) v =W6 this multiples u
      alpha = appctx->param.Lex / 2;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 1;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &grad[0][0], &Nl, &beta, &wrk6[0][0], &Nl);

      // first (B x D_y) v =W7 this mutiplies v
      alpha = 1;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &grad[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk7[0][0], &Nl);

      for (jx = 0; jx < appctx->param.N; jx++) {
        for (jy = 0; jy < appctx->param.N; jy++) {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          outl[indy][indx].u += appctx->param.mu * (wrk2[jy][jx]);// + vlb[jy][jx] * wrk5[jy][jx] + ulb[jy][jx] * wrk4[jy][jx];
          outl[indy][indx].v += appctx->param.mu * (wrk3[jy][jx]);// + ulb[jy][jx] * wrk6[jy][jx] + vlb[jy][jx] * wrk7[jy][jx];
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);

  ierr = VecSet(globalout, 0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, globalout);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, globalout);CHKERRQ(ierr);

  ierr = VecScale(globalout, -1);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(globalout, globalout, appctx->SEMop.mass);CHKERRQ(ierr);

  /*  printf("RHSFunction\n");
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
  VecView(globalin,PETSC_VIEWER_STDOUT_WORLD);VecView(uloc,PETSC_VIEWER_STDOUT_WORLD);VecView(outloc,PETSC_VIEWER_STDOUT_WORLD);VecView(globalout,PETSC_VIEWER_STDOUT_WORLD);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); */

  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  PetscDestroyEl2d(&ulb, appctx);
  PetscDestroyEl2d(&vlb, appctx);
  PetscDestroyEl2d(&wrk1, appctx);
  PetscDestroyEl2d(&wrk2, appctx);
  PetscDestroyEl2d(&wrk3, appctx);
  PetscDestroyEl2d(&wrk4, appctx);
  PetscDestroyEl2d(&wrk5, appctx);
  PetscDestroyEl2d(&wrk6, appctx);
  PetscDestroyEl2d(&wrk7, appctx);

  ierr = VecDestroy(&outloc);CHKERRQ(ierr);
  ierr = VecDestroy(&uloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
{
  AppCtx         *appctx;
  const Field    **ul, **uj;
  Field          **outl;
  PetscScalar    **stiff, **mass, **grad;
  PetscScalar    **wrk1, **wrk2, **wrk3, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar    **ulb, **vlb, **ujb, **vjb;
  PetscInt       Nl, Nl2, inc;
  PetscInt       xs, ys, xm, ym, ix, iy, jx, jy, indx, indy;
  PetscErrorCode ierr;
  Vec            uloc, outloc, ujloc;
  PetscScalar    alpha, beta;

  PetscFunctionBegin;
  MatShellGetContext(H, &appctx);
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  /*  ul contains the input vector as a local array */
  DMCreateLocalVector(appctx->da, &uloc);
  DMGlobalToLocalBegin(appctx->da, in, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, in, INSERT_VALUES, uloc);
  DMDAVecGetArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);

  /*  uj contains the base Jacobian vector (the point the Jacobian is evaluated) as a local array */
  DMCreateLocalVector(appctx->da, &ujloc);
  DMGlobalToLocalBegin(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);
  DMGlobalToLocalEnd(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, ujloc);
  DMDAVecGetArrayRead(appctx->da, ujloc, &uj);CHKERRQ(ierr);

  /* outl contains the output vector as a local array */
  DMCreateLocalVector(appctx->da, &outloc);
  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
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

  beta = 0;
  Nl2  = Nl * Nl;
  inc  = 1;
  for (ix = xs; ix < xs + xm; ix++) {
    for (iy = ys; iy < ys + ym; iy++) {
      for (jx = 0; jx < appctx->param.N; jx++) {
        for (jy = 0; jy < appctx->param.N; jy++) {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          ujb[jy][jx]  = uj[indy][indx].u;         /* base Jacobian vector */
          vjb[jy][jx]  = uj[indy][indx].v;
          ulb[jy][jx]  = ul[indy][indx].u;         /* takes the matrix vector product of this array */
          vlb[jy][jx]  = ul[indy][indx].v;
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy) u=W2 (u_yy)
      alpha = appctx->param.Lex / 2;
      //         PetscRealView(4,&mass[0][0],PETSC_VIEWER_STDOUT_WORLD);
      //      PetscRealView(4,&ulb[0][0],PETSC_VIEWER_STDOUT_WORLD);
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      //         PetscRealView(4,&wrk1[0][0],PETSC_VIEWER_STDOUT_WORLD);
      alpha = 2 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk2[0][0], &Nl);
      //         PetscRealView(4,&stiff[0][0],PETSC_VIEWER_STDOUT_WORLD);
      //                  PetscRealView(4,&wrk2[0][0],PETSC_VIEWER_STDOUT_WORLD);
         
      //second product (K_xx x B) u=W3 (u_xx)
      alpha = 2 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &ulb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      alpha = 1;
      BLASaxpy_(&Nl2, &alpha, &wrk3[0][0], &inc, &wrk2[0][0], &inc); //I freed wrk3 and saved the lalplacian in wrk2

      // for the v component now
      //first product (B x K_yy) v=W3
      alpha = appctx->param.Lex / 2;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &mass[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = 2 / appctx->param.Ley;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &stiff[0][0], &Nl, &beta, &wrk3[0][0], &Nl);

      //second product (K_xx x B) v=W4
      alpha = 2 / appctx->param.Lex;
      BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alpha, &stiff[0][0], &Nl, &vlb[0][0], &Nl, &beta, &wrk1[0][0], &Nl);
      alpha = appctx->param.Ley / 2;
      BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alpha, &wrk1[0][0], &Nl, &mass[0][0], &Nl, &beta, &wrk4[0][0], &Nl);

      alpha = 1;
      BLASaxpy_(&Nl2, &alpha, &wrk4[0][0], &inc, &wrk3[0][0], &inc); //I freed wrk4 and saved the lalplacian in wrk3

      //      PetscRealView(4,&wrk2[0][0],PETSC_VIEWER_STDOUT_WORLD);
      //      PetscRealView(4,&wrk3[0][0],PETSC_VIEWER_STDOUT_WORLD);


      for (jx = 0; jx < appctx->param.N; jx++) {
        for (jy = 0; jy < appctx->param.N; jy++) {
          indx = ix * (appctx->param.N - 1) + jx;
          indy = iy * (appctx->param.N - 1) + jy;
          outl[indy][indx].u += appctx->param.mu * (wrk2[jy][jx]);
          outl[indy][indx].v += appctx->param.mu * (wrk3[jy][jx]);
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc,&ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, ujloc, &uj);CHKERRQ(ierr);

  ierr = VecSet(out, 0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);

  ierr = VecScale(out, -1);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(out, out, appctx->SEMop.mass);CHKERRQ(ierr);

  /*  printf("RHSJacobian\n");
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
  VecView(in,PETSC_VIEWER_STDOUT_WORLD);VecView(uloc,PETSC_VIEWER_STDOUT_WORLD);VecView(outloc,PETSC_VIEWER_STDOUT_WORLD);VecView(out,PETSC_VIEWER_STDOUT_WORLD);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);  */

  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  PetscDestroyEl2d(&ulb, appctx);
  PetscDestroyEl2d(&vlb, appctx);
  PetscDestroyEl2d(&ujb, appctx);
  PetscDestroyEl2d(&vjb, appctx);
  PetscDestroyEl2d(&wrk1, appctx);
  PetscDestroyEl2d(&wrk2, appctx);
  PetscDestroyEl2d(&wrk3, appctx);
  PetscDestroyEl2d(&wrk4, appctx);
  PetscDestroyEl2d(&wrk5, appctx);
  PetscDestroyEl2d(&wrk6, appctx);
  PetscDestroyEl2d(&wrk7, appctx);

  VecDestroy(&uloc);
  VecDestroy(&outloc);
  VecDestroy(&ujloc);
  //VecView(in,0);
  //  VecView(out,0);
  PetscFunctionReturn(0);
}

/*  Keeps a current copy of the Jacobian base vector, needed for the shell matrix to apply the Jacobian and its transpose */
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec globalin, Mat A, Mat B, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx *)ctx;

  PetscFunctionBegin;
  ierr = VecCopy(globalin, appctx->dat.pass_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
PetscErrorCode MonitorError(Tao tao, void *ctx)
{
  AppCtx         *appctx = (AppCtx *)ctx;
  Vec            temp;
  PetscReal      nrm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(appctx->dat.ic, &temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp, -1, appctx->dat.ic, appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp, temp, temp);CHKERRQ(ierr);
  ierr = VecDot(temp, appctx->SEMop.mass, &nrm);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  nrm = PetscSqrtReal(nrm);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "    Error of PDE continuum optimization solution %g\n", (double)nrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/
/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -tao_monitor  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 

   test:
     suffix: cn
     requires: !single
     args: -tao_monitor -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 

TEST*/
