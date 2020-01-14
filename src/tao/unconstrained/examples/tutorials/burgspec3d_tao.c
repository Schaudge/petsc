
static char help[] = "Solves a 3d Burgers PDE constrained optimization algorithm with TSAdjoint and TAO\n\n";


/* ------------------------------------------------------------------------

   The operators are discretized with the spectral element method on GLL points

  ------------------------------------------------------------------------- */

#include <petsctao.h>
#include <petscts.h>
#include <petscdt.h>
#include <petscdraw.h>
#include <petscdmda.h>
#include <stdio.h>
#include <petscblaslapack.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct
{
  PetscInt  n;         /* number of nodes */
  PetscReal *nodes;    /* GLL nodes */
  PetscReal *weights;  /* GLL weights */
  PetscReal **stiff;
  PetscReal **mass;
  PetscReal **grad;
} PetscGLL;

typedef struct
{
  PetscInt  N;               /* grid points per elements*/
  PetscInt  Ex;              /* number of elements */
  PetscInt  Ey;              /* number of elements */
  PetscInt  Ez;              /* number of elements */
  PetscInt  steps;           /* number of timesteps */
  PetscReal Tend;            /* endtime */
  PetscReal mu;              /* viscosity */
  PetscReal Lx;              /* total length of domain */
  PetscReal Ly;              /* total length of domain */
  PetscReal Lz;              /* total length of domain */
  PetscReal Lex;
  PetscReal Ley;
  PetscReal Lez;
  PetscInt  lenx;
  PetscInt  leny;
  PetscInt  lenz;
  PetscReal Tadj;
  PetscReal Tinit;
} PetscParam;

typedef struct
{
  PetscScalar u, v, w;
} Field;

typedef struct
{
  Vec obj;  /* desired end state */
  Vec grad;
  Vec ic;
  Vec curr_sol;
  Vec pass_sol;
  Vec pass_sol_local;
  Vec true_solution; /* actual initial conditions for the final solution */
} PetscData;

typedef struct
{
  Vec      mass;  /* mass matrix for total integration */
  PetscGLL gll;
} PetscSEMOperators;

typedef struct
{
  DM                da;                  /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
  Mat               H_shell,A_full;     /* matrix free operator for Jacobian, AIJ sparse representation of H_shell */
  PetscBool         formexplicitmatrix;  /* matrix is stored in A_full; for comparison only  */
  PetscScalar       ***tenswrk1, ***tenswrk2;
  PetscScalar       ***ulb;
  PetscScalar       ***vlb;
  PetscScalar       ***wlb;
  PetscScalar       ***ujb;
  PetscScalar       ***vjb;
  PetscScalar       ***wjb;
  PetscScalar       ***wrk1;
  PetscScalar       ***wrk2;
  PetscScalar       ***wrk3;
  PetscScalar       ***wrk4;
  PetscScalar       ***wrk5;
  PetscScalar       ***wrk6;
  PetscScalar       ***wrk7;
  PetscScalar       ***wrk8;
  PetscScalar       ***wrk9;
  PetscScalar       ***wrk10;
  PetscScalar       ***wrk11;
  PetscScalar       ***wrk12;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
extern PetscErrorCode InitialConditions(PetscReal, Vec, AppCtx *);
extern PetscErrorCode ComputeObjective(PetscReal, Vec, AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode MyMatMult(Mat, Vec, Vec);
extern PetscErrorCode MyMatMultTransp(Mat, Vec, Vec);
extern PetscErrorCode PetscAllocateEl3d(PetscScalar ****, AppCtx *);
extern PetscErrorCode PetscDestroyEl3d(PetscScalar ****, AppCtx *);
extern PetscErrorCode InitializeSpectral(AppCtx *);

int main(int argc, char **argv)
{
  AppCtx         appctx; /* user-defined application context */
  Vec            u; /* approximate solution vector */
  Tao		 tao;
  PetscErrorCode ierr;
  PetscInt       m, nn,mp,np,pp;
  Vec            global;
  Mat            H_shell;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr)
    return ierr;

  /*initialize parameters */
  appctx.param.N = 6;     /* order of the spectral element */
  appctx.param.Ex = 3;     /* number of elements */
  appctx.param.Ey = 3;     /* number of elements */
  appctx.param.Ez = 4;     /* number of elements */
  appctx.param.Lx = 4.0;   /* length of the domain */
  appctx.param.Ly = 4.0;   /* length of the domain */
  appctx.param.Lz = 4.0;   /* length of the domain */
  appctx.param.mu = 0.005; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend = 0.1;
  appctx.param.Tadj = 0.2;
  appctx.param.Tinit = 0;
  appctx.formexplicitmatrix = PETSC_FALSE;

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &appctx.param.N, NULL);   CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ex", &appctx.param.Ex, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ey", &appctx.param.Ey, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ez", &appctx.param.Ez, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Tend", &appctx.param.Tend, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Tadj", &appctx.param.Tadj, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-mu", &appctx.param.mu, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-formexplicitmatrix", &appctx.formexplicitmatrix, NULL);CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx / appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly / appctx.param.Ey;
  appctx.param.Lez = appctx.param.Lz / appctx.param.Ez;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscMalloc2(appctx.param.N, &appctx.SEMop.gll.nodes, appctx.param.N, &appctx.SEMop.gll.weights);   CHKERRQ(ierr);
  ierr = PetscDTGaussLobattoLegendreQuadrature(appctx.param.N, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights);
  CHKERRQ(ierr);
  appctx.SEMop.gll.n = appctx.param.N;
  appctx.param.lenx = appctx.param.Ex * (appctx.param.N - 1);
  appctx.param.leny = appctx.param.Ey * (appctx.param.N - 1);
  appctx.param.lenz = appctx.param.Ez * (appctx.param.N - 1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, appctx.param.lenx, appctx.param.leny,
                      appctx.param.lenz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, NULL, &appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);   CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);  CHKERRQ(ierr);
  ierr = DMDAGetInfo(appctx.da,NULL,NULL,NULL,NULL,&mp,&np,&pp,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (appctx.param.Ex % mp) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Elements in x direction %D must be divisible by processors in x direction %D",appctx.param.Ex,mp);
  if (appctx.param.Ey % np) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Elements in y direction %D must be divisible by processors in y direction %D",appctx.param.Ey,np);
  if (appctx.param.Ez % pp) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Elements in z direction %D must be divisible by processors in z direction %D",appctx.param.Ez,pp);
  ierr = DMDASetFieldName(appctx.da, 0, "u");  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 1, "v");  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 2, "w");  CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights, &appctx.SEMop.gll.stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights,  &appctx.SEMop.gll.grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights,  &appctx.SEMop.gll.mass);CHKERRQ(ierr);


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
  ierr = DMCreateLocalVector(appctx.da,&appctx.dat.pass_sol_local);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &appctx.dat.obj); CHKERRQ(ierr);
  ierr = InitializeSpectral(&appctx);  CHKERRQ(ierr);
  ierr = DMGetCoordinates(appctx.da, &global);CHKERRQ(ierr);

#if defined(foo)  
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "meshout.m", &viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global, "grid");CHKERRQ(ierr);
  ierr = VecView(global, viewfile); CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx.param.N, "N");CHKERRQ(ierr);
  //ierr = PetscScalarView(1,appctx.param.N, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass, "mass");CHKERRQ(ierr);
  ierr = VecView(appctx.SEMop.mass, viewfile); CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile); CHKERRQ(ierr);
#endif

  /* allocate work space needed by tensor products */
  ierr = PetscAllocateEl3d(&appctx.tenswrk1, &appctx);CHKERRQ(ierr);
  ierr = PetscAllocateEl3d(&appctx.tenswrk2, &appctx);CHKERRQ(ierr);
  PetscAllocateEl3d(&appctx.ulb, &appctx);
  PetscAllocateEl3d(&appctx.vlb, &appctx);
  PetscAllocateEl3d(&appctx.wlb, &appctx);
  PetscAllocateEl3d(&appctx.ujb, &appctx);
  PetscAllocateEl3d(&appctx.vjb, &appctx);
  PetscAllocateEl3d(&appctx.wjb, &appctx);
  PetscAllocateEl3d(&appctx.wrk1, &appctx);
  PetscAllocateEl3d(&appctx.wrk2, &appctx);
  PetscAllocateEl3d(&appctx.wrk3, &appctx);
  PetscAllocateEl3d(&appctx.wrk4, &appctx);
  PetscAllocateEl3d(&appctx.wrk5, &appctx);
  PetscAllocateEl3d(&appctx.wrk6, &appctx);
  PetscAllocateEl3d(&appctx.wrk7, &appctx);
  PetscAllocateEl3d(&appctx.wrk8, &appctx);
  PetscAllocateEl3d(&appctx.wrk9, &appctx);
  PetscAllocateEl3d(&appctx.wrk10, &appctx);
  PetscAllocateEl3d(&appctx.wrk11, &appctx);
  PetscAllocateEl3d(&appctx.wrk12, &appctx);  
 
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
  ierr = TSSetSaveTrajectory(appctx.ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);CHKERRQ(ierr);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx.ts, &appctx.initial_dt);CHKERRQ(ierr);

  ierr = InitialConditions(appctx.param.Tinit, appctx.dat.ic, &appctx);CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic,appctx.dat.curr_sol);CHKERRQ(ierr);
  /* Create matrix-free matrices for applying Jacobian of RHS function */
  ierr = VecGetLocalSize(u, &m);CHKERRQ(ierr);
  ierr = VecGetSize(u, &nn);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD, m, m, nn, nn, &appctx, &H_shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H_shell, MATOP_MULT, (void (*)(void))MyMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H_shell, MATOP_MULT_TRANSPOSE, (void (*)(void))MyMatMultTransp);CHKERRQ(ierr);
  appctx.H_shell = H_shell;
  if (!appctx.formexplicitmatrix) {
    ierr = TSSetRHSJacobian(appctx.ts, H_shell, H_shell, RHSJacobian, &appctx);CHKERRQ(ierr);
    appctx.A_full = NULL;
  } else {
    PetscInt xm,ym,zm,Nd;
    ierr = DMDAGetCorners(appctx.da,NULL,NULL,NULL,&xm,&ym,&zm);CHKERRQ(ierr);
    Nd   = nn;
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,m,m,nn,nn,Nd,NULL,Nd,NULL,&appctx.A_full);CHKERRQ(ierr);
    ierr = MatSetOption(appctx.A_full,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(appctx.ts, appctx.A_full, appctx.A_full, RHSJacobian, &appctx);CHKERRQ(ierr);
  }
  ierr = TSSetRHSFunction(appctx.ts, NULL, RHSFunction, &appctx);CHKERRQ(ierr);

  ierr = VecCopy(appctx.dat.ic, appctx.dat.pass_sol);  CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx.da,appctx.dat.pass_sol,INSERT_VALUES,appctx.dat.pass_sol_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx.da,appctx.dat.pass_sol,INSERT_VALUES,appctx.dat.pass_sol_local);CHKERRQ(ierr);

  /* Set Objective and Initial conditions for the problem and compute Objective function (evolution of true_solution to final time */
  
  ierr = ComputeObjective(appctx.param.Tadj, appctx.dat.obj, &appctx); CHKERRQ(ierr);
  /* Create TAO solver and set desired solution method  */
  ierr = TaoCreate(PETSC_COMM_WORLD, &tao); CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)tao,"TS",(PetscObject)appctx.ts);CHKERRQ(ierr);
  ierr = TaoSetType(tao, TAOBLMVM); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao, appctx.dat.ic); CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation  */
  ierr = TaoSetObjectiveAndGradientRoutine(tao, FormFunctionGradient, (void *)&appctx);CHKERRQ(ierr);
  /* Check for any TAO command line options  */
  ierr = TaoSetTolerances(tao, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
  //TaoSetMaximumIterations(tao,1);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr); 
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);
  ierr = VecDestroy(&appctx.dat.pass_sol);
  ierr = VecDestroy(&appctx.dat.pass_sol_local);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);
  ierr = MatDestroy(&H_shell);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A_full);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = PetscFree2(appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights);CHKERRQ(ierr);

  ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights, &appctx.SEMop.gll.stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights, &appctx.SEMop.gll.grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx.SEMop.gll.n, appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights, &appctx.SEMop.gll.mass);CHKERRQ(ierr);

  ierr = PetscDestroyEl3d(&appctx.tenswrk1, &appctx);CHKERRQ(ierr);
  ierr = PetscDestroyEl3d(&appctx.tenswrk2, &appctx);CHKERRQ(ierr);
  PetscDestroyEl3d(&appctx.ulb, &appctx);
  PetscDestroyEl3d(&appctx.vlb, &appctx);
  PetscDestroyEl3d(&appctx.wlb, &appctx);
  PetscDestroyEl3d(&appctx.ujb, &appctx);
  PetscDestroyEl3d(&appctx.vjb, &appctx);
  PetscDestroyEl3d(&appctx.wjb, &appctx);
  PetscDestroyEl3d(&appctx.wrk1, &appctx);
  PetscDestroyEl3d(&appctx.wrk2, &appctx);
  PetscDestroyEl3d(&appctx.wrk3, &appctx);
  PetscDestroyEl3d(&appctx.wrk4, &appctx);
  PetscDestroyEl3d(&appctx.wrk5, &appctx);
  PetscDestroyEl3d(&appctx.wrk6, &appctx);
  PetscDestroyEl3d(&appctx.wrk7, &appctx);
  PetscDestroyEl3d(&appctx.wrk8, &appctx);
  PetscDestroyEl3d(&appctx.wrk9, &appctx);
  PetscDestroyEl3d(&appctx.wrk10, &appctx);
  PetscDestroyEl3d(&appctx.wrk11, &appctx);
  PetscDestroyEl3d(&appctx.wrk12, &appctx);

  ierr = PetscFinalize();
  return ierr;
}

/*
Initialize Spectral grid and mass matrix
*/
PetscErrorCode InitializeSpectral(AppCtx *appctx)
{
  PetscErrorCode ierr;
  DM             cda;
  PetscInt       xs, xm, ys, ym, zs, zm, ix, iy, iz, jx, jy, jz;
  PetscInt       indx, indy, indz;
  PetscReal      x, y, z;
  Field          ***bmass;
  DMDACoor3d     ***coors;
  Vec            global, loc;
      
  PetscFunctionBegin;
  
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
 ierr = DMDAGetCorners(appctx->da, &xs, &ys, &zs, &xm, &ym, &zm);   CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid */
  xs = xs / (appctx->param.N - 1);
  xm = xm / (appctx->param.N - 1);
  ys = ys / (appctx->param.N - 1);
  ym = ym / (appctx->param.N - 1);
  zs = zs / (appctx->param.N - 1);
  zm = zm / (appctx->param.N - 1);

  VecSet(appctx->SEMop.mass, 0.0);

  DMCreateLocalVector(appctx->da, &loc);
  ierr = DMDAVecGetArray(appctx->da, loc, &bmass);   CHKERRQ(ierr);

  /*
     Build mass over entire mesh (multi-elemental) 

  */

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (jx = 0; jx < appctx->param.N; jx++)
    {
      for (iy = ys; iy < ys + ym; iy++)
      {
        for (jy = 0; jy < appctx->param.N; jy++)
        {
          for (iz = zs; iz < zs + zm; iz++)
          {
            for (jz = 0; jz < appctx->param.N; jz++)
            {
              x = (appctx->param.Lex / 2.0) * (appctx->SEMop.gll.nodes[jx] + 1.0) + appctx->param.Lex * ix;
              y = (appctx->param.Ley / 2.0) * (appctx->SEMop.gll.nodes[jy] + 1.0) + appctx->param.Ley * iy;
              z = (appctx->param.Lez / 2.0) * (appctx->SEMop.gll.nodes[jz] + 1.0) + appctx->param.Lez * iz;
              indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;
              bmass[indz][indy][indx].u += appctx->SEMop.gll.weights[jx] * appctx->SEMop.gll.weights[jy] * appctx->SEMop.gll.weights[jz] *
                                           0.125 * appctx->param.Lez * appctx->param.Ley * appctx->param.Lex;
              bmass[indz][indy][indx].v += appctx->SEMop.gll.weights[jx] * appctx->SEMop.gll.weights[jy] * appctx->SEMop.gll.weights[jz] *
                                           0.125 * appctx->param.Lez * appctx->param.Ley * appctx->param.Lex;
              bmass[indz][indy][indx].w += appctx->SEMop.gll.weights[jx] * appctx->SEMop.gll.weights[jy] * appctx->SEMop.gll.weights[jz] *
                                           0.125 * appctx->param.Lez * appctx->param.Ley * appctx->param.Lex;
            }
          }
        }
      }
    }
  }

  DMDAVecRestoreArray(appctx->da, loc, &bmass);
  CHKERRQ(ierr);
  DMLocalToGlobalBegin(appctx->da, loc, ADD_VALUES, appctx->SEMop.mass);
  DMLocalToGlobalEnd(appctx->da, loc, ADD_VALUES, appctx->SEMop.mass);

  DMDASetUniformCoordinates(appctx->da, 0.0, appctx->param.Lx, 0.0, appctx->param.Ly, 0.0, appctx->param.Lz);
  DMGetCoordinateDM(appctx->da, &cda);

  DMGetCoordinates(appctx->da, &global);
  VecSet(global, 0.0);
  DMDAVecGetArray(cda, global, &coors);

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (jx = 0; jx < appctx->param.N - 1; jx++)
    {
      for (iy = ys; iy < ys + ym; iy++)
      {
        for (jy = 0; jy < appctx->param.N - 1; jy++)
        {
          for (iz = zs; iz < zs + zm; iz++)
          {
            for (jz = 0; jz < appctx->param.N - 1; jz++)
            {
              x = (appctx->param.Lex / 2.0) * (appctx->SEMop.gll.nodes[jx] + 1.0) + appctx->param.Lex * ix - 2.0;
              y = (appctx->param.Ley / 2.0) * (appctx->SEMop.gll.nodes[jy] + 1.0) + appctx->param.Ley * iy - 2.0;
              z = (appctx->param.Lez / 2.0) * (appctx->SEMop.gll.nodes[jz] + 1.0) + appctx->param.Lez * iz - 2.0;
              indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;
              coors[indz][indy][indx].x = x;
              coors[indz][indy][indx].y = y;
              coors[indz][indy][indx].z = z;
            }
          }
        }
      }
    }
  }
  DMDAVecRestoreArray(cda, global, &coors);
  ierr = VecDestroy(&loc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscPointWiseMult(PetscInt Nl, const PetscScalar *A, const PetscScalar *B, PetscScalar *out)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < Nl; i++) {
    out[i] = A[i] * B[i];
  }
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()


   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(PetscReal t, Vec u, AppCtx *appctx)
{
  Field            ***s;
  PetscErrorCode   ierr;
  PetscInt         i, j, k, xs, ys, zs, xm, ym, zm;
  DM               cda;
  Vec              global;
  const DMDACoor3d ***coors;

  PetscFunctionBegin;
  ierr = DMDAVecGetArray(appctx->da, u, &s);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(appctx->da, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(appctx->da, &global);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda, global, &coors);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        s[k][j][i].u = PetscExpScalar(-appctx->param.mu * t) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].v = PetscExpScalar(-appctx->param.mu * t) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].w = PetscExpScalar(-appctx->param.mu * t) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, u, &s);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda, global, &coors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
   Sets the desired profile for the final end time; 

   Input Parameters:
   t - final time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ComputeObjective(PetscReal t, Vec obj, AppCtx *appctx)
{
  Field           ***s;
  PetscErrorCode  ierr;
  PetscInt        i, j, k, xs, ys, zs, xm, ym, zm;
  DM              cda;
  Vec             global;
  const DMDACoor3d ***coors;

  PetscFunctionBegin;
  ierr = DMDAVecGetArray(appctx->da, obj, &s);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(appctx->da, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(appctx->da, &global);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cda, global, &coors);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        s[k][j][i].u = PetscExpScalar(-appctx->param.mu * t) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].v = PetscExpScalar(-appctx->param.mu * t) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].w = PetscExpScalar(-appctx->param.mu * t) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, obj, &s);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cda, global, &coors);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscTens3dSEM(PetscScalar ***A, PetscScalar ***B, PetscScalar ***C, PetscScalar ****ulb, PetscScalar ***out, PetscScalar *alphavec, AppCtx *appctx)
{
  const PetscBLASInt Nl = (PetscBLASInt)appctx->param.N, Nl2 = Nl * Nl;
  PetscInt           jx;
  PetscScalar        *temp1, *temp2;
  PetscScalar        ***wrk1 = appctx->tenswrk1, ***wrk2  = appctx->tenswrk2, ***wrk3 = out;
  const PetscReal    beta = 0;

  PetscFunctionBegin;
  BLASgemm_("T", "N", &Nl, &Nl2, &Nl, &alphavec[0], A[0][0], &Nl, ulb[0][0][0], &Nl, &beta, &wrk1[0][0][0], &Nl);
  for (jx = 0; jx < Nl; jx++) {
    temp1 = &wrk1[0][0][0] + jx * Nl2;
    temp2 = &wrk2[0][0][0] + jx * Nl2;
    BLASgemm_("N", "N", &Nl, &Nl, &Nl, &alphavec[1], temp1, &Nl, B[0][0], &Nl, &beta, temp2, &Nl);
  }
  BLASgemm_("N", "N", &Nl2, &Nl, &Nl, &alphavec[2], &wrk2[0][0][0], &Nl2, C[0][0], &Nl, &beta, &wrk3[0][0][0], &Nl2);
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscTens3dSEMTranspose(PetscScalar ***A, PetscScalar ***B, PetscScalar ***C, PetscScalar ****ulb, PetscScalar ***out, PetscScalar *alphavec, AppCtx *appctx)
{
  const PetscBLASInt Nl = (PetscBLASInt)appctx->param.N, Nl2 = Nl * Nl;
  PetscInt           jx;
  PetscScalar        *temp1, *temp2;
  PetscScalar        ***wrk1 = appctx->tenswrk1, ***wrk2  = appctx->tenswrk2, ***wrk3 = out;
  const PetscReal    beta = 0;

  PetscFunctionBegin;
  BLASgemm_("N", "N", &Nl, &Nl2, &Nl, &alphavec[0], A[0][0], &Nl, ulb[0][0][0], &Nl, &beta, &wrk1[0][0][0], &Nl);
  for (jx = 0; jx < Nl; jx++) {
    temp1 = &wrk1[0][0][0] + jx * Nl2;
    temp2 = &wrk2[0][0][0] + jx * Nl2;
    BLASgemm_("N", "T", &Nl, &Nl, &Nl, &alphavec[1], temp1, &Nl, B[0][0], &Nl, &beta, temp2, &Nl);
  }
  BLASgemm_("N", "T", &Nl2, &Nl, &Nl, &alphavec[2], &wrk2[0][0][0], &Nl2, C[0][0], &Nl, &beta, &wrk3[0][0][0], &Nl2);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscAllocateEl3d(PetscScalar ****AA, AppCtx *appctx)
{
  PetscScalar    ***A, **B, *C;
  PetscErrorCode ierr;
  PetscInt       Nl = appctx->param.N, Nl2 = Nl*Nl, Nl3 = Nl2*Nl;
  PetscInt       ix, iy;

  PetscFunctionBegin;
  ierr = PetscMalloc1(Nl, &A);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl2, &B);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl3, &C);CHKERRQ(ierr);

  for (ix = 0; ix < Nl; ix++) {
    A[ix] = B + ix * Nl;
  }
  for (ix = 0; ix < Nl; ix++) {
    for (iy = 0; iy < Nl; iy++) {
      A[ix][iy] = C + ix * Nl * Nl + iy * Nl;
    }
  }
  *AA = A;
  PetscFunctionReturn(0);
}
PetscErrorCode PetscDestroyEl3d(PetscScalar ****AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0][0]);CHKERRQ(ierr);
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx *)ctx;
  PetscScalar    ***wrk3, ***wrk1, ***wrk2, ***wrk4, ***wrk5, ***wrk6, ***wrk7;
  PetscScalar    ***wrk8, ***wrk9, ***wrk10, ***wrk11;
  PetscScalar    **stiff=appctx->SEMop.gll.stiff;
  PetscScalar    **mass=appctx->SEMop.gll.mass;
  PetscScalar    **grad=appctx->SEMop.gll.grad;
  PetscScalar    ***ulb, ***vlb, ***wlb;
  const Field    ***ul;
  Field          ***outl;
  PetscInt       ix, iy, iz, jx, jy, jz, indx, indy, indz;
  PetscInt       xs, xm, ys, ym, zs, zm;
  Vec            uloc, outloc;
  PetscScalar    alpha;
  PetscReal      alphavec[3];
  PetscBLASInt   inc = 1,Nl = appctx->param.N, Nl3 = Nl*Nl*Nl;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(appctx->da, &uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx->da, globalin, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da, globalin, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da, uloc, &ul); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(appctx->da, &outloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, outloc, &outl); CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);
  zs = zs / (Nl - 1);
  zm = zm / (Nl - 1);

  /*
     Initialize work arrays
  */
  PetscAllocateEl3d(&ulb, appctx);
  PetscAllocateEl3d(&vlb, appctx);
  PetscAllocateEl3d(&wlb, appctx);
  PetscAllocateEl3d(&wrk1, appctx);
  PetscAllocateEl3d(&wrk2, appctx);
  PetscAllocateEl3d(&wrk3, appctx);
  PetscAllocateEl3d(&wrk4, appctx);
  PetscAllocateEl3d(&wrk5, appctx);
  PetscAllocateEl3d(&wrk6, appctx);
  PetscAllocateEl3d(&wrk7, appctx);
  PetscAllocateEl3d(&wrk8, appctx);
  PetscAllocateEl3d(&wrk9, appctx);
  PetscAllocateEl3d(&wrk10, appctx);
  PetscAllocateEl3d(&wrk11, appctx);

  for (ix = xs; ix < xs + xm; ix++) {
    for (iy = ys; iy < ys + ym; iy++) {
      for (iz = zs; iz < zs + zm; iz++) {
        for (jx = 0; jx < Nl; jx++) {
          for (jy = 0; jy < Nl; jy++) {
            for (jz = 0; jz < Nl; jz++) {
              indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;
              ulb[jz][jy][jx] = ul[indz][indy][indx].u;
              vlb[jz][jy][jx] = ul[indz][indy][indx].v;
              wlb[jz][jy][jx] = ul[indz][indy][indx].w;
            }
          }
        }

        alpha=1.0;
        //here the stifness matrix in 3d
        //the term (B x B x K_zz)u=W1 (u_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &ulb, wrk1, alphavec, appctx);
        //the term (B x K_yy x B)u=W1 (u_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &ulb, wrk2, alphavec, appctx);
        //the term (K_xx x B x B)u=W1 (u_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &ulb, wrk3, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
        BLASaxpy_(&Nl3, &alpha, &wrk2[0][0][0], &inc, &wrk1[0][0][0], &inc); //I freed wrk2 and saved the laplacian in wrk1

        //the term (B x B x K_zz)v=W2 (v_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &vlb, wrk2, alphavec, appctx);
        //the term (B x K_yy x B)v=W2 (v_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &vlb, wrk3, alphavec, appctx);
        //the term (K_xx x B x B)v=W2 (v_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &vlb, wrk4, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2

        //the term (B x B x K_zz)w=W3 (w_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &wlb, wrk3, alphavec, appctx);
        //the term (B x K_yy x B)w=W3 (w_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &wlb, wrk4, alphavec, appctx);
        //the term (K_xx x B x B)w=W3 (w_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &wlb, wrk5, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk5[0][0][0], &inc, &wrk4[0][0][0], &inc); //I freed wrk5 and saved the laplacian in wrk4
        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        // I save w1, w2, w3

        //now the gradient operator for u
         //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ulb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ulb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ulb, wrk6, alphavec, appctx);

        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk7[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk8[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk9[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for v
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vlb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vlb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vlb, wrk6, alphavec, appctx);        

        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk8[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk9[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //now the gradient operator for w
         //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wlb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wlb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wlb, wrk6, alphavec, appctx);

        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk11[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk10[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk9[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        /* I saved w7 w8 w9 */
        for (jx = 0; jx < appctx->param.N; jx++) {
          for (jy = 0; jy < appctx->param.N; jy++) {
            for (jz = 0; jz < appctx->param.N; jz++) {
              indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;
              outl[indz][indy][indx].u += appctx->param.mu *wrk1[jz][jy][jx]+wrk7[jz][jy][jx];
              outl[indz][indy][indx].v += appctx->param.mu *wrk2[jz][jy][jx]+wrk8[jz][jy][jx];
              outl[indz][indy][indx].w += appctx->param.mu *wrk3[jz][jy][jx]+wrk9[jz][jy][jx];
            }
          }
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);

  ierr = VecSet(globalout, 0.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, globalout);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, globalout);CHKERRQ(ierr);
  ierr = VecScale(globalout, -1.0);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(globalout, globalout, appctx->SEMop.mass);CHKERRQ(ierr);

  PetscDestroyEl3d(&ulb, appctx);
  PetscDestroyEl3d(&vlb, appctx);
  PetscDestroyEl3d(&wlb, appctx);
  PetscDestroyEl3d(&wrk1, appctx);
  PetscDestroyEl3d(&wrk2, appctx);
  PetscDestroyEl3d(&wrk3, appctx);
  PetscDestroyEl3d(&wrk4, appctx);
  PetscDestroyEl3d(&wrk5, appctx);
  PetscDestroyEl3d(&wrk6, appctx);
  PetscDestroyEl3d(&wrk7, appctx);
  PetscDestroyEl3d(&wrk8, appctx);
  PetscDestroyEl3d(&wrk9, appctx);
  PetscDestroyEl3d(&wrk10, appctx);
  PetscDestroyEl3d(&wrk11, appctx);

  ierr = VecDestroy(&outloc);CHKERRQ(ierr);
  ierr = VecDestroy(&uloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMult"
PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
{
  PetscErrorCode ierr;
  AppCtx         *appctx;
  PetscScalar ***wrk3, ***wrk1, ***wrk2, ***wrk4, ***wrk5, ***wrk6, ***wrk7;
  PetscScalar ***wrk8, ***wrk9, ***wrk10, ***wrk11, ***wrk12;
  PetscScalar ***ulb, ***vlb, ***wlb;
  PetscScalar ***ujb, ***vjb, ***wjb;
  const Field ***ul, ***uj;
  Field ***outl;
  PetscInt ix, iy, iz, jx, jy, jz, indx, indy, indz;
  PetscInt xs, xm, ys, ym, zs, zm, Nl, Nl3;
  Vec uloc, outloc;
  PetscScalar alpha;
  PetscReal alphavec[3];
  PetscInt inc;
  PetscScalar **stiff;
  PetscScalar **mass;
  PetscScalar **grad;
  
  PetscFunctionBegin;
  ierr  = MatShellGetContext(H, &appctx);CHKERRQ(ierr);
  stiff = appctx->SEMop.gll.stiff;
  mass  = appctx->SEMop.gll.mass;
  grad  = appctx->SEMop.gll.grad;
  
  DMGetLocalVector(appctx->da, &uloc);
  DMGlobalToLocalBegin(appctx->da, in, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, in, INSERT_VALUES, uloc);
  DMDAVecGetArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);

  /*  uj contains the base Jacobian vector (the point the Jacobian is evaluated) as a local array */
  DMDAVecGetArrayRead(appctx->da, appctx->dat.pass_sol_local, &uj);CHKERRQ(ierr);

  /* outl contains the output vector as a local array */
  ierr = DMGetLocalVector(appctx->da, &outloc);CHKERRQ(ierr);
  ierr = VecSet(outloc,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);

  Nl = appctx->param.N;
  Nl3 = appctx->param.N * appctx->param.N * appctx->param.N;

  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);
  zs = zs / (Nl - 1);
  zm = zm / (Nl - 1);

  inc = 1;
  /*
     Initialize work arrays
  */
  PetscAllocateEl3d(&ulb, appctx);
  PetscAllocateEl3d(&vlb, appctx);
  PetscAllocateEl3d(&wlb, appctx);
  PetscAllocateEl3d(&ujb, appctx);
  PetscAllocateEl3d(&vjb, appctx);
  PetscAllocateEl3d(&wjb, appctx);
  PetscAllocateEl3d(&wrk1, appctx);
  PetscAllocateEl3d(&wrk2, appctx);
  PetscAllocateEl3d(&wrk3, appctx);
  PetscAllocateEl3d(&wrk4, appctx);
  PetscAllocateEl3d(&wrk5, appctx);
  PetscAllocateEl3d(&wrk6, appctx);
  PetscAllocateEl3d(&wrk7, appctx);
  PetscAllocateEl3d(&wrk8, appctx);
  PetscAllocateEl3d(&wrk9, appctx);
  PetscAllocateEl3d(&wrk10, appctx);
  PetscAllocateEl3d(&wrk11, appctx);
  PetscAllocateEl3d(&wrk12, appctx);

  for (ix = xs; ix < xs + xm; ix++)
  {
    for (iy = ys; iy < ys + ym; iy++)
    {
      for (iz = zs; iz < zs + zm; iz++)

      {
        for (jx = 0; jx < Nl; jx++)
        {
          for (jy = 0; jy < Nl; jy++)
          {
            for (jz = 0; jz < Nl; jz++)
            {
	            indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;

              ujb[jz][jy][jx] = uj[indz][indy][indx].u;
              vjb[jz][jy][jx] = uj[indz][indy][indx].v;
              wjb[jz][jy][jx] = uj[indz][indy][indx].w;

              ulb[jz][jy][jx] = ul[indz][indy][indx].u;
              vlb[jz][jy][jx] = ul[indz][indy][indx].v;
              wlb[jz][jy][jx] = ul[indz][indy][indx].w;
	          }
          }
        }

        alpha=1.0;
        //here the stifness matrix in 3d
        //the term (B x B x K_zz)u=W1 (u_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &ulb, wrk1, alphavec, appctx);
        //the term (B x K_yy x B)u=W1 (u_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &ulb, wrk2, alphavec, appctx);
        //the term (K_xx x B x B)u=W1 (u_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &ulb, wrk3, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
        BLASaxpy_(&Nl3, &alpha, &wrk2[0][0][0], &inc, &wrk1[0][0][0], &inc); //I freed wrk2 and saved the laplacian in wrk1

      	//the term (B x B x K_zz)v=W2 (v_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &vlb, wrk2, alphavec, appctx);
        //the term (B x K_yy x B)v=W2 (v_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &vlb, wrk3, alphavec, appctx);
        //the term (K_xx x B x B)v=W2 (v_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &vlb, wrk4, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
 
        //the term (B x B x K_zz)w=W3 (w_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &wlb, wrk3, alphavec, appctx);
        //the term (B x K_yy x B)w=W3 (w_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &wlb, wrk4, alphavec, appctx);
        //the term (K_xx x B x B)w=W3 (w_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &wlb, wrk5, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk5[0][0][0], &inc, &wrk4[0][0][0], &inc); //I freed wrk5 and saved the laplacian in wrk4
        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        // I save w1, w2, w3
        
        //now the gradient operator for u
        //compute u.*(D w_u), i.e. ujb.*(D ulb)
         //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ulb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ulb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ulb, wrk6, alphavec, appctx);
        
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk7[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk8[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk9[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //compute w.*(D u)  i.e. ulb.*(D ujb)   
        //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ujb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ujb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ujb, wrk6, alphavec, appctx);
        
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk8[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk9[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //sum  u.*(D w_u)[w7]+w.*(D u)[w8]
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for v
        //compute u.*(D w_v), i.e. ujb.*(D vlb)
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vlb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vlb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vlb, wrk6, alphavec, appctx);        
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk8[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk9[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk10[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //compute w.*(D v), i.e. ulb.*(D vjb)
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vjb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vjb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vjb, wrk6, alphavec, appctx);        
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk9[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk10[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk11[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8


        //sum  u.*(D w_v)[w8]+w.*(D v)[w9]
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for w
        //compute u.*(D w_w), i.e. ujb.*(D wlb)
         //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wlb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wlb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wlb, wrk6, alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk11[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk10[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk9[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        //compute w.*(D w), i.e. ulb.*(D wjb)
        //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wjb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wjb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wjb, wrk6, alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk12[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk11[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk12[0][0][0], &inc, &wrk11[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        //sum  u.*(D w_v)[w9]+w.*(D v)[w10]
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //I saved w7 w8 w9
                      
        for (jx = 0; jx < appctx->param.N; jx++)
         {
          for (jy = 0; jy < appctx->param.N; jy++)
          {
            for (jz = 0; jz < appctx->param.N; jz++)
            {
              indx = ix * (appctx->param.N - 1) + jx;
              indy = iy * (appctx->param.N - 1) + jy;
              indz = iz * (appctx->param.N - 1) + jz;

              outl[indz][indy][indx].u += appctx->param.mu *wrk1[jz][jy][jx]+wrk7[jz][jy][jx];
              outl[indz][indy][indx].v += appctx->param.mu *wrk2[jz][jy][jx]+wrk8[jz][jy][jx];
              outl[indz][indy][indx].w += appctx->param.mu *wrk3[jz][jy][jx]+wrk9[jz][jy][jx];
            }
          }
        }
      } 
    }
  }

  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, appctx->dat.pass_sol_local, &uj);CHKERRQ(ierr);

  ierr = VecSet(out,0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);

  ierr = VecScale(out, -1);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(out, out, appctx->SEMop.mass);CHKERRQ(ierr);

  /*  printf("RHSJacobian\n");
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
  VecView(in,PETSC_VIEWER_STDOUT_WORLD);VecView(uloc,PETSC_VIEWER_STDOUT_WORLD);VecView(outloc,PETSC_VIEWER_STDOUT_WORLD);VecView(out,PETSC_VIEWER_STDOUT_WORLD);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);  */


  DMRestoreLocalVector(appctx->da, &uloc);
  DMRestoreLocalVector(appctx->da, &outloc);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMultTransp"
PetscErrorCode MyMatMultTransp(Mat H, Vec in, Vec out)
{
  AppCtx            *appctx;
  PetscErrorCode    ierr;
  PetscScalar       ***wrk3, ***wrk1, ***wrk2, ***wrk4, ***wrk5, ***wrk6, ***wrk7;
  PetscScalar       ***wrk8, ***wrk9, ***wrk10, ***wrk11, ***wrk12;
  PetscScalar       ***ulb, ***vlb, ***wlb;
  PetscScalar       ***ujb, ***vjb, ***wjb;
  const Field       ***ul, ***uj;
  Field             ***outl;
  PetscInt          ix, iy, iz, jx, jy, jz, indx, indy, indz;
  PetscInt          xs, xm, ys, ym, zs, zm, Nl, Nl3;
  Vec               uloc, outloc, incopy;
  PetscReal         alphavec[3];
  PetscScalar       **stiff;
  PetscScalar       **mass;
  PetscScalar       **grad;
  const PetscInt    inc = 1;
  const PetscScalar alpha=1.0;

  PetscFunctionBegin;
  ierr  = MatShellGetContext(H, &appctx);CHKERRQ(ierr);
  stiff = appctx->SEMop.gll.stiff;
  mass  = appctx->SEMop.gll.mass;
  grad  = appctx->SEMop.gll.grad;
  ulb  = appctx->ulb;
  vlb  = appctx->vlb;
  wlb  = appctx->wlb;
  ujb  = appctx->ujb;
  vjb  = appctx->vjb;
  wjb  = appctx->wjb;
  wrk1  = appctx->wrk1;
  wrk2  = appctx->wrk2;
  wrk3  = appctx->wrk3;
  wrk4  = appctx->wrk4;
  wrk5  = appctx->wrk5;
  wrk6  = appctx->wrk6;
  wrk7  = appctx->wrk7;
  wrk8  = appctx->wrk8;
  wrk9  = appctx->wrk9;
  wrk10  = appctx->wrk10;
  wrk11  = appctx->wrk11;
  wrk12  = appctx->wrk12;

  ierr = DMGetGlobalVector(appctx->da, &incopy);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(incopy, in, appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = VecScale(incopy, -1);CHKERRQ(ierr);

  /* ul contains local array of input, the vector the transpose is applied to */
  ierr = DMGetLocalVector(appctx->da, &uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(appctx->da, incopy, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da, incopy, INSERT_VALUES, uloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da, uloc, &ul);CHKERRQ(ierr);CHKERRQ(ierr);

  /* uj contains local array of Jacobian base vector, the location the Jacobian is evaluated at */
  ierr = DMDAVecGetArrayRead(appctx->da, appctx->dat.pass_sol_local, &uj);CHKERRQ(ierr);CHKERRQ(ierr);

  /* outl contains local array of output vector (the transpose product) */
  ierr = DMGetLocalVector(appctx->da, &outloc);CHKERRQ(ierr);
  ierr = VecSet(outloc,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, outloc, &outl);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  Nl = appctx->param.N;
  Nl3 = appctx->param.N * appctx->param.N * appctx->param.N;
  xs = xs / (Nl - 1);
  xm = xm / (Nl - 1);
  ys = ys / (Nl - 1);
  ym = ym / (Nl - 1);
  zs = zs / (Nl - 1);
  zm = zm / (Nl - 1);

  /*
     Initialize work arrays
  */


  for (ix = xs; ix < xs + xm; ix++) {
    for (iy = ys; iy < ys + ym; iy++) {
      for (iz = zs; iz < zs + zm; iz++) {
        for (jx = 0; jx < Nl; jx++) {
          for (jy = 0; jy < Nl; jy++) {
            for (jz = 0; jz < Nl; jz++) {
              indx = ix * (Nl - 1) + jx;
              indy = iy * (Nl - 1) + jy;
              indz = iz * (Nl - 1) + jz;

              ujb[jz][jy][jx] = uj[indz][indy][indx].u;
              vjb[jz][jy][jx] = uj[indz][indy][indx].v;
              wjb[jz][jy][jx] = uj[indz][indy][indx].w;

              ulb[jz][jy][jx] = ul[indz][indy][indx].u;
              vlb[jz][jy][jx] = ul[indz][indy][indx].v;
              wlb[jz][jy][jx] = ul[indz][indy][indx].w;
            }
          }
        }

        //here the stifness matrix in 3d
        //the term (B x B x K_zz)u=W1 (u_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEMTranspose(&mass, &mass, &stiff, &ulb, wrk1, alphavec, appctx);
        //the term (B x K_yy x B)u=W1 (u_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &stiff, &mass, &ulb, wrk2, alphavec, appctx);
        //the term (K_xx x B x B)u=W1 (u_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&stiff, &mass, &mass, &ulb, wrk3, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
        BLASaxpy_(&Nl3, &alpha, &wrk2[0][0][0], &inc, &wrk1[0][0][0], &inc); //I freed wrk2 and saved the laplacian in wrk1

        	//the term (B x B x K_zz)v=W2 (v_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEMTranspose(&mass, &mass, &stiff, &vlb, wrk2, alphavec, appctx);
        //the term (B x K_yy x B)v=W2 (v_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &stiff, &mass, &vlb, wrk3, alphavec, appctx);
        //the term (K_xx x B x B)v=W2 (v_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&stiff, &mass, &mass, &vlb, wrk4, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
 
        //the term (B x B x K_zz)w=W3 (w_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEMTranspose(&mass, &mass, &stiff, &wlb, wrk3, alphavec, appctx);
        //the term (B x K_yy x B)w=W3 (w_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &stiff, &mass, &wlb, wrk4, alphavec, appctx);
        //the term (K_xx x B x B)w=W3 (w_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&stiff, &mass, &mass, &wlb, wrk5, alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk5[0][0][0], &inc, &wrk4[0][0][0], &inc); //I freed wrk5 and saved the laplacian in wrk4
        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        // I save w1, w2, w3

        //now the gradient operator for u
        //compute D^T(u .* w_u), i.e. D^T.*(ujb.* ulb)
        PetscPointWiseMult(Nl3, &ulb[0][0][0], &ujb[0][0][0], &wrk4[0][0][0]);
         //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&grad, &mass, &mass, &wrk4, wrk7, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y) 
        PetscPointWiseMult(Nl3, &ulb[0][0][0], &vjb[0][0][0], &wrk4[0][0][0]);       
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &grad, &mass, &wrk4, wrk8, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)                
        PetscPointWiseMult(Nl3, &ulb[0][0][0], &wjb[0][0][0], &wrk4[0][0][0]);     
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEMTranspose(&mass, &mass, &grad, &wrk4, wrk9, alphavec, appctx);
        
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //compute w.*(D_x u)  i.e. ulb.*(D_x ujb)   
        //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ujb, wrk4, alphavec, appctx);
        //the term (D_x x B x B)v=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vjb, wrk5, alphavec, appctx);
        //the term (D_x x B x B)w=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wjb, wrk6, alphavec, appctx);
        
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk8[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk9[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //sum  u.*(D w_u)[w7]+w.*(D u)[w8]
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for v
        //compute D^T(u .* w_u), i.e. D^T.*(ujb.* vlb)
        PetscPointWiseMult(Nl3, &vlb[0][0][0], &ujb[0][0][0], &wrk4[0][0][0]);
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&grad, &mass, &mass, &wrk4, wrk8, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)   
        PetscPointWiseMult(Nl3, &vlb[0][0][0], &vjb[0][0][0], &wrk4[0][0][0]);
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &grad, &mass, &wrk4, wrk9, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)    
        PetscPointWiseMult(Nl3, &vlb[0][0][0], &wjb[0][0][0], &wrk4[0][0][0]);
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEMTranspose(&mass, &mass, &grad, &wrk4, wrk10, alphavec, appctx);        
 
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //compute w.*(D v), i.e. ulb.*(D_y vjb)
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ujb, wrk4, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vjb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wjb, wrk6, alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk9[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk10[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk11[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //sum  u.*(D w_v)[w8]+w.*(D v)[w9]
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for w
         //compute D^T(u .* w_u), i.e. D^T.*(ujb.* vlb)
        PetscPointWiseMult(Nl3, &wlb[0][0][0], &ujb[0][0][0], &wrk4[0][0][0]);
         //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&grad, &mass, &mass, &wrk4, wrk9, alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)    
        PetscPointWiseMult(Nl3, &wlb[0][0][0], &vjb[0][0][0], &wrk4[0][0][0]);
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEMTranspose(&mass, &grad, &mass, &wrk4, wrk10, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)  
        PetscPointWiseMult(Nl3, &wlb[0][0][0], &wjb[0][0][0], &wrk4[0][0][0]);
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEMTranspose(&mass, &mass, &grad, &wrk4, wrk11, alphavec, appctx);
 
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        //compute w.*(D w), i.e. ulb.*(D wjb)
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ujb, wrk4, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vjb, wrk5, alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wjb, wrk6, alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk12[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk11[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk12[0][0][0], &inc, &wrk11[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        //sum  u.*(D w_v)[w9]+w.*(D v)[w10]
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //I saved w7 w8 w9
        for (jx = 0; jx < Nl; jx++) {
          for (jy = 0; jy < Nl; jy++) {
            for (jz = 0; jz < Nl; jz++) {
              indx = ix * (Nl - 1) + jx;
              indy = iy * (Nl - 1) + jy;
              indz = iz * (Nl - 1) + jz;
              outl[indz][indy][indx].u += appctx->param.mu *wrk1[jz][jy][jx]+wrk7[jz][jy][jx];
              outl[indz][indy][indx].v += appctx->param.mu *wrk2[jz][jy][jx]+wrk8[jz][jy][jx];
              outl[indz][indy][indx].w += appctx->param.mu *wrk3[jz][jy][jx]+wrk9[jz][jy][jx];
            }
          }
        }
      }
    }
  }
  /*                          pointwise   axpy             tens                                             outl */
  ierr = PetscLogFlops(xm*ym*zm*(18*Nl3 + 2*21*Nl3 + 27*(2*Nl*Nl*Nl*Nl + 2*Nl*Nl*Nl*Nl + 2*Nl*Nl*Nl*Nl) + 3*Nl*Nl*Nl));

  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc,&ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, appctx->dat.pass_sol_local, &uj);CHKERRQ(ierr);

  ierr = VecSet(out,0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, out);CHKERRQ(ierr);

  /*  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
  VecView(in,PETSC_VIEWER_STDOUT_WORLD);VecView(outloc,PETSC_VIEWER_STDOUT_WORLD);VecView(out,PETSC_VIEWER_STDOUT_WORLD);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);   */


  ierr = DMRestoreGlobalVector(appctx->da, &incopy);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da, &uloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(appctx->da, &outloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec globalin, Mat A, Mat B, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx *)ctx;

  PetscFunctionBegin;

  /* save current Jacobian base vector that defines where it is applied */
  ierr = VecCopy(globalin, appctx->dat.pass_sol); CHKERRQ(ierr);
  /* save local copy of Jacobian base vector so do not need to do the GlobalToLocal() every time in the MatMult or MatMultTranspose */
  ierr = DMGlobalToLocalBegin(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, appctx->dat.pass_sol_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(appctx->da, appctx->dat.pass_sol, INSERT_VALUES, appctx->dat.pass_sol_local);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->H_shell, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->H_shell, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (appctx->formexplicitmatrix) {
    ierr = MatConvert(appctx->H_shell,MATAIJ, MAT_REUSE_MATRIX,&appctx->A_full);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   IC   - the input vector
   ctx - optional user-defined context, as set when calling TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient

   Notes:

          The forward equation is
              M u_t = F(U)
          which is converted to
                u_t = M^{-1} F(u)
          in the user code since TS has no direct way of providing a mass matrix. The Jacobian of this is
                 M^{-1} J
          where J is the Jacobian of F. Now the adjoint equation is
                M v_t = J^T v
          but TSAdjoint does not solve this since it can only solve the transposed system for the 
          Jacobian the user provided. Hence TSAdjoint solves
                 w_t = J^T M^{-1} w  (where w = M v)
          since there is no way to indicate the mass matrix as a seperate entitity to TS. Thus one
          must be careful in initializing the "adjoint equation" and using the result. This is
          why
              G = -2 M(u(T) - u_d)
          below (instead of -2(u(T) - u_d) and why the result is
              G = G/appctx->SEMop.mass (that is G = M^{-1}w)
          below (instead of just the result of the "adjoint solve").


*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec IC, PetscReal *f, Vec G, void *ctx)
{
  AppCtx             *appctx = (AppCtx *)ctx; /* user-defined application context */
  PetscErrorCode     ierr;
  Vec                temp, bsol, adj;
  PetscInt           its;
  PetscReal          ff, gnorm, cnorm, xdiff, errex;
  TaoConvergedReason reason;

  PetscFunctionBegin;
  ierr = TSSetTime(appctx->ts, 0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(appctx->ts, 0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx->ts, appctx->initial_dt);CHKERRQ(ierr);
  ierr = VecCopy(IC, appctx->dat.curr_sol);CHKERRQ(ierr);

  ierr = TSSolve(appctx->ts, appctx->dat.curr_sol);CHKERRQ(ierr);
  
  /*
  Store current solution for comparison
  */
  ierr = VecDuplicate(appctx->dat.curr_sol, &bsol);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.curr_sol, bsol);CHKERRQ(ierr);
  ierr = VecWAXPY(G, -1, appctx->dat.curr_sol, appctx->dat.obj);CHKERRQ(ierr);

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = VecDuplicate(G, &temp);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp, G, G);CHKERRQ(ierr);
  ierr = VecDot(temp, appctx->SEMop.mass, f);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);

  /* local error evaluation; TODO: remove this since Monitor displays the same information ? */
  ierr = VecDuplicate(appctx->dat.ic, &temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp, -1, appctx->dat.ic, appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp, temp, temp);CHKERRQ(ierr);
  ierr = VecDot(temp, appctx->SEMop.mass, &errex);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  errex = PetscSqrtReal(errex);
  
  /*
     Compute initial conditions for the adjoint integration. See Notes above
  */
  ierr = VecScale(G, -2);CHKERRQ(ierr);
  ierr = VecPointwiseMult(G, G, appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = TSSetCostGradients(appctx->ts, 1, &G, NULL);CHKERRQ(ierr);

  ierr = VecDuplicate(G, &adj);CHKERRQ(ierr);
  ierr = VecCopy(G, adj);CHKERRQ(ierr);

  /* solve gthe adjoint system for the gradient */
  ierr = TSAdjointSolve(appctx->ts);CHKERRQ(ierr);
  //ierr = VecPointwiseDivide(G, G, appctx->SEMop.mass);CHKERRQ(ierr);

  ierr = TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason);CHKERRQ(ierr);

#if defined(foo)  
  //counter++; // this was for storing the error accross line searches, we don't use it anymore
  //  PetscPrintf(PETSC_COMM_WORLD, "iteration=%D\t cost function (TAO)=%g, cost function (L2 %g), ic error %g\n", its, (double)ff, *f, errex);
  PetscSNPrintf(filename, sizeof(filename), "PDEadjoint/optimize%02d.m", its);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewfile);   CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
  PetscSNPrintf(data, sizeof(data), "TAO(%D)=%g; L2(%D)= %g ; Err(%D)=%g\n", its + 1, (double)ff, its + 1, *f, its + 1, errex);
  PetscViewerASCIIPrintf(viewfile, data);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.obj, "obj");
  ierr = VecView(appctx->dat.obj, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G, "Init_adj"); CHKERRQ(ierr);
  ierr = VecView(G, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)adj, "adj"); CHKERRQ(ierr);
  ierr = VecView(adj, viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)IC, "Init_ts"); CHKERRQ(ierr);
  ierr = VecView(IC, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)bsol, "fwd"); CHKERRQ(ierr);
  ierr = VecView(bsol, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.curr_sol, "Curr_sol"); CHKERRQ(ierr);
  ierr = VecView(appctx->dat.curr_sol, viewfile);  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.true_solution, "exact"); CHKERRQ(ierr);
  ierr = VecView(appctx->dat.true_solution, viewfile);  CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);  CHKERRQ(ierr);
#endif
  
  VecDestroy(&bsol);
  VecDestroy(&adj);
  /*   PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
  VecView(IC,0);
  VecView(G,0);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD); */
  PetscFunctionReturn(0);
}

/*TEST

   build:
     requires: !complex

   test:
     args: -tao_view -ts_trajectory_type memory -tao_monitor -tao_gttol 1.e-3 -Ex 2 -Ey 2 -Ez 2 -N 3 -Tend .8 -Tadj 1.2 -tao_converged_reason -tao_max_it 30  -ts_adapt_type none

   test:
     suffix: p
     nsize: 2
     timeoutfactor: 3
     args: -tao_view -ts_trajectory_type memory -tao_monitor -tao_gttol 1.e-3 -Ex 2 -Ey 2 -Ez 2 -N 3 -Tend .8 -Tadj 1.2 -tao_converged_reason -tao_max_it 30 -ts_adapt_type none

   test:
     suffix: fd
     requires: !single
     timeoutfactor: 10
     args: -tao_view -ts_trajectory_type memory -tao_monitor -tao_gttol 1.e-1 -Ex 2 -Ey 2 -Ez 2 -N 3 -Tend .2 -Tadj .3 -tao_converged_reason -tao_max_it 30 -ts_adapt_type none -tao_test_gradient

   test:
     suffix: rhs_fd
     requires: !single
     timeoutfactor: 8
     args: -tao_view -ts_trajectory_type memory -tao_monitor -tao_gttol 1.e-2 -Ex 2 -Ey 2 -Ez 2 -N 3 -Tend .4 -Tadj .7 -tao_converged_reason -tao_max_it 30  -ts_adapt_type none -ts_rhs_jacobian_test_mult_transpose

   test:
     suffix: cn_p
     nsize: 3
     timeoutfactor: 3
     requires: !single
     args: -tao_view -ts_trajectory_type memory -tao_monitor  -Ex 2 -Ey 3 -Ez 3 -N 6 -Tend .8 -Tadj 1.0 -tao_converged_reason -tao_max_it 30 -tao_gttol 1.e-3 -ts_type cn -pc_type none

   test:
     suffix: cn_fd
     timeoutfactor: 8
     requires: !single
     args: -tao_view -ts_trajectory_type memory -tao_monitor  -Ex 2 -Ey 2 -Ez 2 -N 3 -Tend .1 -Tadj .12 -tao_converged_reason -tao_max_it 30 -tao_gttol 1.e-1 -ts_type cn -pc_type none  -tao_test_gradient

TEST*/

