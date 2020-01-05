
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
  PetscInt n;         /* number of nodes */
  PetscReal *nodes;   /* GLL nodes */
  PetscReal *weights; /* GLL weights */
} PetscGLL;

typedef struct
{
  PetscInt N;                /* grid points per elements*/
  PetscInt Ex;               /* number of elements */
  PetscInt Ey;               /* number of elements */
  PetscInt Ez;               /* number of elements */
  PetscReal tol_L2, tol_max; /* error norms */
  PetscInt steps;            /* number of timesteps */
  PetscReal Tend;            /* endtime */
  PetscReal mu;              /* viscosity */
  PetscReal Lx;              /* total length of domain */
  PetscReal Ly;              /* total length of domain */
  PetscReal Lz;              /* total length of domain */
  PetscReal Lex;
  PetscReal Ley;
  PetscReal Lez;
  PetscInt lenx;
  PetscInt leny;
  PetscInt lenz;
  PetscReal Tinit;
} PetscParam;

typedef struct
{
  PetscScalar u, v, w; /* wind speed */
} Field;

typedef struct
{
  Vec grad;
  Vec ic;
  Vec curr_sol;
  Vec pass_sol;
  Vec true_solution; /* actual initial conditions for the final solution */
} PetscData;

typedef struct
{
  Vec grid;  /* total grid */
  Vec mass;  /* mass matrix for total integration */
  Mat stiff; /* stifness matrix */
  Mat keptstiff;
  Mat grad;
  Mat opadd;
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
extern PetscErrorCode InitialConditions(PetscReal, Vec, AppCtx *);
extern PetscErrorCode TrueSolution(PetscReal, Vec, AppCtx *);
extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode MyMatMult(Mat, Vec, Vec);
extern PetscErrorCode PetscAllocateEl3d(PetscScalar ****, AppCtx *);
extern PetscErrorCode PetscDestroyEl3d(PetscScalar ****, AppCtx *);
extern PetscErrorCode PetscPointWiseMult(PetscInt, const PetscScalar *, const PetscScalar *, PetscScalar *);
extern PetscErrorCode PetscTens3dSEM(PetscScalar ***, PetscScalar ***, PetscScalar ***, PetscScalar ****, PetscScalar ***, PetscScalar **,AppCtx *appctx);
extern PetscErrorCode InitializeSpectral(AppCtx *);

int main(int argc, char **argv)
{
  AppCtx         appctx; /* user-defined application context */
  Vec            u; /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt m, nn;
  Vec global;
  PetscViewer viewfile;
  Mat H_shell;

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
  appctx.param.Tend = 2.0;
  appctx.param.Tinit = 0;

  ierr = PetscOptionsGetInt(NULL, NULL, "-N", &appctx.param.N, NULL);   CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ex", &appctx.param.Ex, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ey", &appctx.param.Ey, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ez", &appctx.param.Ey, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Tend", &appctx.param.Tend, NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-mu", &appctx.param.mu, NULL);  CHKERRQ(ierr);
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
                      appctx.param.lenz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, NULL, &appctx.da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);   CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 0, "u");  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 1, "v");  CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da, 2, "w");  CHKERRQ(ierr);

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
  ierr = VecView(global, viewfile); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass, "mass");CHKERRQ(ierr);
  ierr = VecView(appctx.SEMop.mass, viewfile); CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);   CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da, &appctx.SEMop.stiff); CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da, &appctx.SEMop.grad);  CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da, &appctx.SEMop.opadd); CHKERRQ(ierr);
  */
 
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


  ierr = InitialConditions(appctx.param.Tinit, appctx.dat.ic, &appctx);CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic,appctx.dat.curr_sol);CHKERRQ(ierr);
  /* Create matrix-free matrices for applying Jacobian of RHS function */
  ierr = VecGetLocalSize(u, &m);CHKERRQ(ierr);
  ierr = VecGetSize(u, &nn);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD, m, m, nn, nn, &appctx, &H_shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H_shell, MATOP_MULT, (void (*)(void))MyMatMult);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts, H_shell, H_shell, RHSJacobian, &appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts, NULL, RHSFunction, &appctx);CHKERRQ(ierr);
  //ierr = TSSetRHSJacobian(appctx.ts,H_shell,H_shell,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
 
  ierr = VecCopy(appctx.dat.ic, appctx.dat.pass_sol);  CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic, appctx.dat.curr_sol);  CHKERRQ(ierr);
  //VecSet(appctx.dat.pass_sol,1.0);
  //Vec ref;
  //ierr = VecDuplicate(appctx.dat.ic, &ref);   CHKERRQ(ierr);
  //VecSet(ref,0);
  //RHSFunction(appctx.ts, 0.0, appctx.dat.ic, ref, &appctx);
  ierr = TSSolve(appctx.ts, appctx.dat.curr_sol);

  /*
  Vec wrk_vec, jac, vec_jac, vec_rhs, temp, vec_trans;
  Field ***s;
  PetscScalar vareps;
  PetscInt i;
  PetscInt its = 0;
  char var[15];

  ierr = VecDuplicate(appctx.dat.ic, &wrk_vec);   CHKERRQ(ierr);
  //ierr = VecDuplicate(appctx.dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic, &vec_jac);  CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic, &vec_rhs);  CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic, &vec_trans);  CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic, &ref);  CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic, &jac);  CHKERRQ(ierr);

  ierr = VecCopy(appctx.dat.ic, appctx.dat.pass_sol);
  CHKERRQ(ierr);
  //VecSet(appctx.dat.pass_sol,1.0);

  RHSFunction(appctx.ts, 0.0, appctx.dat.ic, ref, &appctx);
 
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
  //ierr = VecDuplicate(appctx.dat.ic,&uu);CHKERRQ(ierr);
  //ierr = VecCopy(appctx.dat.ic,uu);CHKERRQ(ierr);
  //MatView(H_shell,0);
  
   //ierr = VecDuplicate(appctx.dat.ic,&appctx.dat.curr_sol);CHKERRQ(ierr);
   //ierr = VecCopy(appctx.dat.ic,appctx.dat.curr_sol);CHKERRQ(ierr);
   //ierr = TSSolve(appctx.ts,appctx.dat.curr_sol);CHKERRQ(ierr);
*/

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"sol2d.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.curr_sol,"sol");
    ierr = VecView(appctx.dat.curr_sol,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"ic");
    ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
   
/* attach the null space to the matrix, this is not needed for periodic BCs as here */

  /*
  MatNullSpace nsp;

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(H_shell,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,H_shell,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  */

  /* The solution to the continous PDE optimization problem evaluted at the discrete grid points */
  /* In the limit as one refines the mesh one hopes the TAO solution converges to this value */
  ierr = TrueSolution(appctx.param.Tend, appctx.dat.true_solution, &appctx);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);
  ierr = VecDestroy(&appctx.dat.pass_sol);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);
  ierr = MatDestroy(&H_shell);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = PetscFree2(appctx.SEMop.gll.nodes, appctx.SEMop.gll.weights);CHKERRQ(ierr); 
  PetscViewerDestroy(&viewfile);
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
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


PetscErrorCode PetscPointWiseMult(PetscInt Nl, const PetscScalar *A, const PetscScalar *B, PetscScalar *out)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < Nl; i++) 
  {
    out[i] = A[i] * B[i];
  }
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

                       The routine TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(PetscReal tt, Vec u, AppCtx *appctx)
{ Field ***s;
  PetscErrorCode ierr;
  PetscInt i, j, k;
  DM cda;
  Vec global;
  DMDACoor3d ***coors;

  PetscFunctionBegin; 
  ierr = DMDAVecGetArray(appctx->da, u, &s);   CHKERRQ(ierr);

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

  for (i = 0; i < appctx->param.lenx; i++)
  {
    for (j = 0; j < appctx->param.leny; j++)
    {
      for (k = 0; k < appctx->param.lenz; k++)
      {
        s[k][j][i].u = PetscExpScalar(-appctx->param.mu * tt) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].v = PetscExpScalar(-appctx->param.mu * tt) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].w = PetscExpScalar(-appctx->param.mu * tt) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, u, &s);   CHKERRQ(ierr);
  DMDAVecRestoreArray(cda, global, &coors);
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
PetscErrorCode TrueSolution(PetscReal tt, Vec u, AppCtx *appctx)
{
  Field ***s;
  PetscErrorCode ierr;
  PetscInt i, j, k;
  DM cda;
  Vec global;
  DMDACoor3d ***coors;

  PetscFunctionBegin;
  ierr = DMDAVecGetArray(appctx->da, u, &s);   CHKERRQ(ierr);

  DMGetCoordinateDM(appctx->da, &cda);
  DMGetCoordinates(appctx->da, &global);
  DMDAVecGetArray(cda, global, &coors);

  for (i = 0; i < appctx->param.lenx; i++)
  {
    for (j = 0; j < appctx->param.leny; j++)
    {
      for (k = 0; k < appctx->param.lenz; k++)
      {
        s[k][j][i].u = PetscExpScalar(-appctx->param.mu * tt) * (PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].v = PetscExpScalar(-appctx->param.mu * tt) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
        s[k][j][i].w = PetscExpScalar(-appctx->param.mu * tt) * (PetscSinScalar(0.5 * PETSC_PI * coors[k][j][i].x) + PetscCosScalar(0.5 * PETSC_PI * coors[k][j][i].y)) / 10.0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(appctx->da, u, &s); CHKERRQ(ierr);
  DMDAVecRestoreArray(cda, global, &coors);
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------- */

PetscErrorCode PetscTens3dSEM(PetscScalar ***A, PetscScalar ***B, PetscScalar ***C, PetscScalar ****ulb, PetscScalar ***out, PetscScalar **alphavec, AppCtx *appctx)
{
  PetscInt Nl, Nl2;
  PetscInt jx;
  PetscScalar *temp1, *temp2;
  PetscScalar ***wrk1, ***wrk2, ***wrk3 = out;
  PetscReal beta;
 
  PetscFunctionBegin;
  Nl = appctx->param.N;
  Nl2 = Nl * Nl;
  
  beta=0.0;
  
  PetscAllocateEl3d(&wrk1, appctx);
  PetscAllocateEl3d(&wrk2, appctx);

  BLASgemm_("T", "N", &Nl, &Nl2, &Nl, alphavec[0], A[0][0], &Nl, ulb[0][0][0], &Nl, &beta, &wrk1[0][0][0], &Nl);
  for (jx = 0; jx < Nl; jx++)
  {
    temp1 = &wrk1[0][0][0] + jx * Nl2;
    temp2 = &wrk2[0][0][0] + jx * Nl2;

    BLASgemm_("N", "N", &Nl, &Nl, &Nl, alphavec[0]+1, temp1, &Nl, B[0][0], &Nl, &beta, temp2, &Nl);
  }

  BLASgemm_("N", "N", &Nl2, &Nl, &Nl, alphavec[0]+2, &wrk2[0][0][0], &Nl2, C[0][0], &Nl, &beta, &wrk3[0][0][0], &Nl2);

  PetscDestroyEl3d(&wrk1, appctx);
  PetscDestroyEl3d(&wrk2, appctx);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscAllocateEl3d(PetscScalar ****AA, AppCtx *appctx)
{
  PetscScalar ***A, **B, *C;
  PetscErrorCode ierr;
  PetscInt Nl, Nl2, Nl3;
  PetscInt ix, iy;

  PetscFunctionBegin;
  Nl = appctx->param.N;
  Nl2 = appctx->param.N * appctx->param.N;
  Nl3 = appctx->param.N * appctx->param.N * appctx->param.N;

  ierr = PetscMalloc1(Nl, &A);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl2, &B);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nl3, &C);CHKERRQ(ierr);

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
PetscErrorCode PetscDestroyEl3d(PetscScalar ****AA, AppCtx *appctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0][0]);   CHKERRQ(ierr);
  ierr = PetscFree((*AA)[0]);  CHKERRQ(ierr);
  ierr = PetscFree(*AA);  CHKERRQ(ierr);

  *AA = NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *appctx = (AppCtx *)ctx;
  PetscScalar ***wrk3, ***wrk1, ***wrk2, ***wrk4, ***wrk5, ***wrk6, ***wrk7;
  PetscScalar ***wrk8, ***wrk9, ***wrk10, ***wrk11;
  PetscScalar **stiff, **mass, **grad;
  PetscScalar ***ulb, ***vlb, ***wlb;
  const Field ***ul;
  Field ***outl;
  PetscInt ix, iy, iz, jx, jy, jz, indx, indy, indz;
  PetscInt xs, xm, ys, ym, zs, zm, Nl, Nl3;
  Vec uloc, outloc;
  PetscScalar alpha;
  PetscReal *alphavec;
  PetscInt inc;
  PetscViewer viewfile;
  static int its = 0;
  char var[12];

  PetscFunctionBegin;

  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  /* unwrap local vector for the input solution */
  /* globalin, the global array
     uloc, the local array
     ul, the pointer to uloc*/

  DMCreateLocalVector(appctx->da, &uloc);

  DMGlobalToLocalBegin(appctx->da, globalin, INSERT_VALUES, uloc);
  DMGlobalToLocalEnd(appctx->da, globalin, INSERT_VALUES, uloc);

  ierr = DMDAVecGetArrayRead(appctx->da, uloc, &ul); CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da, &outloc);

  ierr = DMDAVecGetArray(appctx->da, outloc, &outl); CHKERRQ(ierr);
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

  ierr = PetscMalloc1(3, &alphavec);
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
             
              ulb[jz][jy][jx] = ul[indz][indy][indx].u;
              vlb[jz][jy][jx] = ul[indz][indy][indx].v;
              wlb[jz][jy][jx] = ul[indz][indy][indx].w;
	            //ulb[jx][jy][jz] = (double)rand()/RAND_MAX*2.0-1.0;
            }
          }
        }

        alpha=1.0;
        //here the stifness matrix in 3d
        //the term (B x B x K_zz)u=W1 (u_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &ulb, wrk1, &alphavec, appctx);
        //the term (B x K_yy x B)u=W1 (u_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &ulb, wrk2, &alphavec, appctx);
        //the term (K_xx x B x B)u=W1 (u_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &ulb, wrk3, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
        BLASaxpy_(&Nl3, &alpha, &wrk2[0][0][0], &inc, &wrk1[0][0][0], &inc); //I freed wrk2 and saved the laplacian in wrk1

        //the term (B x B x K_zz)v=W2 (v_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &vlb, wrk2, &alphavec, appctx);
        //the term (B x K_yy x B)v=W2 (v_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &vlb, wrk3, &alphavec, appctx);
        //the term (K_xx x B x B)v=W2 (v_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &vlb, wrk4, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
 
        //the term (B x B x K_zz)w=W3 (w_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &wlb, wrk3, &alphavec, appctx);
        //the term (B x K_yy x B)w=W3 (w_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &wlb, wrk4, &alphavec, appctx);
        //the term (K_xx x B x B)w=W3 (w_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &wlb, wrk5, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk5[0][0][0], &inc, &wrk4[0][0][0], &inc); //I freed wrk5 and saved the laplacian in wrk4
        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        // I save w1, w2, w3
               
        //now the gradient operator for u
         //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ulb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ulb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ulb, wrk6, &alphavec, appctx);
        
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk7[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk8[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk9[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //now the gradient operator for v
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vlb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vlb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vlb, wrk6, &alphavec, appctx);        
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk8[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk9[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk10[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //now the gradient operator for w
         //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wlb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wlb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wlb, wrk6, &alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ulb[0][0][0], &wrk11[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vlb[0][0][0], &wrk10[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wlb[0][0][0], &wrk9[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

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

VecSet(globalout, 0.0);
DMLocalToGlobalBegin(appctx->da, outloc, ADD_VALUES, globalout);
DMLocalToGlobalEnd(appctx->da, outloc, ADD_VALUES, globalout);

VecScale(globalout, -1.0);

ierr = VecPointwiseDivide(globalout, globalout, appctx->SEMop.mass);
CHKERRQ(ierr);

ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
ierr = PetscGaussLobattoLegendreElementAdvectionDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
ierr = PetscGaussLobattoLegendreElementMassDestroy(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

/*
its = 1;
ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "rhsB.m", &viewfile); CHKERRQ(ierr);
ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
PetscSNPrintf(var, sizeof(var), "inr(:,%d)", its); 
ierr = PetscObjectSetName((PetscObject)globalin, var);
ierr = VecView(globalin, viewfile); CHKERRQ(ierr);
PetscSNPrintf(var, sizeof(var), "outr(:,%d)", its);
ierr = PetscObjectSetName((PetscObject)globalout, var);
ierr = VecView(globalout, viewfile); CHKERRQ(ierr);
ierr = PetscViewerPopFormat(viewfile);
*/

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
ierr = PetscFree(alphavec);CHKERRQ(ierr);
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
  PetscScalar **stiff, **mass, **grad;
  PetscScalar ***ulb, ***vlb, ***wlb;
  PetscScalar ***ujb, ***vjb, ***wjb;
  const Field ***ul, ***uj;
  Field ***outl;
  PetscInt ix, iy, iz, jx, jy, jz, indx, indy, indz;
  PetscInt xs, xm, ys, ym, zs, zm, Nl, Nl3;
  Vec uloc, outloc, ujloc;
  PetscScalar alpha;
  PetscReal *alphavec;
  PetscInt inc;
  PetscViewer viewfile;
  static int its = 0;
  char var[12];

  PetscFunctionBegin;
  ierr = MatShellGetContext(H, &appctx);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementLaplacianCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &stiff);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementAdvectionCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &grad);CHKERRQ(ierr);
  ierr = PetscGaussLobattoLegendreElementMassCreate(appctx->SEMop.gll.n, appctx->SEMop.gll.nodes, appctx->SEMop.gll.weights, &mass);CHKERRQ(ierr);

  /* unwrap local vector for the input solution */
  /* globalin, the global array
     uloc, the local array
     ul, the pointer to uloc*/

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

  ierr = PetscMalloc1(3, &alphavec);CHKERRQ(ierr);
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

              ulb[jz][jy][jx] = ul[indz][indy][indz].u;
              vlb[jz][jy][jx] = ul[indz][indy][indx].v;
              wlb[jz][jy][jx] = ul[indz][indy][indx].w;
	          }
          }
        }

        alpha=1.0;
        //here the stifness matrix in 3d
        //the term (B x B x K_zz)u=W1 (u_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &ulb, wrk1, &alphavec, appctx);
        //the term (B x K_yy x B)u=W1 (u_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &ulb, wrk2, &alphavec, appctx);
        //the term (K_xx x B x B)u=W1 (u_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &ulb, wrk3, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
        BLASaxpy_(&Nl3, &alpha, &wrk2[0][0][0], &inc, &wrk1[0][0][0], &inc); //I freed wrk2 and saved the laplacian in wrk1

      	//the term (B x B x K_zz)v=W2 (v_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &vlb, wrk2, &alphavec, appctx);
        //the term (B x K_yy x B)v=W2 (v_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &vlb, wrk3, &alphavec, appctx);
        //the term (K_xx x B x B)v=W2 (v_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &vlb, wrk4, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        BLASaxpy_(&Nl3, &alpha, &wrk3[0][0][0], &inc, &wrk2[0][0][0], &inc); //I freed wrk3 and saved the laplacian in wrk2
 
        //the term (B x B x K_zz)w=W3 (w_zz)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=2. / appctx->param.Lez;
        PetscTens3dSEM(&mass, &mass, &stiff, &wlb, wrk3, &alphavec, appctx);
        //the term (B x K_yy x B)w=W3 (w_yy)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=2./appctx->param.Ley;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &stiff, &mass, &wlb, wrk4, &alphavec, appctx);
        //the term (K_xx x B x B)w=W3 (w_xx)         
        alphavec[0]=2./appctx->param.Lex;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&stiff, &mass, &mass, &wlb, wrk5, &alphavec, appctx);

        BLASaxpy_(&Nl3, &alpha, &wrk5[0][0][0], &inc, &wrk4[0][0][0], &inc); //I freed wrk5 and saved the laplacian in wrk4
        BLASaxpy_(&Nl3, &alpha, &wrk4[0][0][0], &inc, &wrk3[0][0][0], &inc); //I freed wrk4 and saved the laplacian in wrk3
        // I save w1, w2, w3

        //now the gradient operator for u
        //compute u.*(D w_u), i.e. ujb.*(D ulb)
         //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ulb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ulb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ulb, wrk6, &alphavec, appctx);
        
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk7[0][0][0]); //u.*u_x goes in w7, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk8[0][0][0]); //v.*u_y goes in w8, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk9[0][0][0]); //w.*u_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8
        BLASaxpy_(&Nl3, &alpha, &wrk8[0][0][0], &inc, &wrk7[0][0][0], &inc); //I freed wrk8 and saved the grad in wrk7

        //compute w.*(D u)  i.e. ulb.*(D ujb)   
        //the term (D_x x B x B)u=W4 (u_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &ujb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (u_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &ujb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (u_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &ujb, wrk6, &alphavec, appctx);
        
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
        PetscTens3dSEM(&grad, &mass, &mass, &vlb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vlb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vlb, wrk6, &alphavec, appctx);        
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk8[0][0][0]); //u.*v_x goes in w10, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk9[0][0][0]); //v.*v_y goes in w9, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk10[0][0][0]);//w.*v_z goes in w8, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the grad in wrk9
        BLASaxpy_(&Nl3, &alpha, &wrk9[0][0][0], &inc, &wrk8[0][0][0], &inc); //I freed wrk9 and saved the grad in wrk8

        //compute w.*(D v), i.e. ulb.*(D vjb)
        //the term (D_x x B x B)u=W4 (v_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &vjb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (v_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &vjb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (v_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &vjb, wrk6, &alphavec, appctx);        
 
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
        PetscTens3dSEM(&grad, &mass, &mass, &wlb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wlb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wlb, wrk6, &alphavec, appctx);
 
        PetscPointWiseMult(Nl3, &wrk4[0][0][0], &ujb[0][0][0], &wrk11[0][0][0]); //u.*w_x goes in w11, w4 free
        PetscPointWiseMult(Nl3, &wrk5[0][0][0], &vjb[0][0][0], &wrk10[0][0][0]); //v.*w_y goes in w10, w5 free
        PetscPointWiseMult(Nl3, &wrk6[0][0][0], &wjb[0][0][0], &wrk9[0][0][0]);  //w.*w_z goes in w9, w6 free
        //sum up all contributions
        BLASaxpy_(&Nl3, &alpha, &wrk11[0][0][0], &inc, &wrk10[0][0][0], &inc); //I freed wrk11 and saved the laplacian in wrk10
        BLASaxpy_(&Nl3, &alpha, &wrk10[0][0][0], &inc, &wrk9[0][0][0], &inc); //I freed wrk10 and saved the laplacian in wrk9

        //compute w.*(D w), i.e. ulb.*(D wjb)
        //the term (D_x x B x B)u=W4 (w_x)         
        alphavec[0]=1.0;  alphavec[1]=appctx->param.Ley/2.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&grad, &mass, &mass, &wjb, wrk4, &alphavec, appctx);
        //the term (B x D_y x B)u=W4 (w_y)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=1.0;  alphavec[2]=appctx->param.Lez/2.0;
        PetscTens3dSEM(&mass, &grad, &mass, &wjb, wrk5, &alphavec, appctx);
        //the term (B x B x D_z)u=W4 (w_z)         
        alphavec[0]=appctx->param.Lex / 2.0;  alphavec[1]=appctx->param.Ley / 2.0;  alphavec[2]=1.0;
        PetscTens3dSEM(&mass, &mass, &grad, &wjb, wrk6, &alphavec, appctx);
 
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

  ierr = DMDAVecRestoreArray(appctx->da, outloc, &outl);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, uloc,&ul);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da, ujloc, &uj);CHKERRQ(ierr);

  ierr = VecSet(out,0);CHKERRQ(ierr);
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
/*
  its = 1;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "rhsJac.m", &viewfile); CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
  PetscSNPrintf(var, sizeof(var), "inr(:,%d)", its); 
  ierr = PetscObjectSetName((PetscObject)in, var);
  ierr = VecView(in, viewfile); CHKERRQ(ierr);
  PetscSNPrintf(var, sizeof(var), "outr(:,%d)", its);
  ierr = PetscObjectSetName((PetscObject)out, var);
  ierr = VecView(out, viewfile); CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);
*/
  PetscDestroyEl3d(&ulb, appctx);
  PetscDestroyEl3d(&vlb, appctx);
  PetscDestroyEl3d(&wlb, appctx);
  PetscDestroyEl3d(&ujb, appctx);
  PetscDestroyEl3d(&vjb, appctx);
  PetscDestroyEl3d(&wjb, appctx);
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
  PetscDestroyEl3d(&wrk12, appctx);

  VecDestroy(&uloc);
  VecDestroy(&outloc);
  VecDestroy(&ujloc);
  ierr = PetscFree(alphavec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec globalin, Mat A, Mat B, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *appctx = (AppCtx *)ctx;
  PetscFunctionBegin;

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  ierr = VecCopy(globalin, appctx->dat.pass_sol); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


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
