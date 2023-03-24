static const char help[] = "DGNetwork Conservation Law Test Function. \n\
Just Runs an Simulation with the specified Setup. \n\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

/*
  This only has shallow water physics for now
*/

/*
  Example:
  1. Run default SWE and view on X
     mpiexec -n 1 ./ex8 -view
  2. Requires: GLVis
     Run default SWE and view on GLVis
     mpiexec -n 1 ./ex8 -view -view_glvis -view_full_net -glvis_pause 1e-10
  3. Run on Parent-Daughter Network with P^4 DG Basis and linearized coupling
     mpiexec -n 1 ./ex8 -view -network 3 -order 4
  4. Requires: GLVis
     Run on Parent-Daughter Network with P^4 DG Basis and linearized coupling View GLVis
     mpiexec -n 1 ./ex8 -view -network 3 -view_glvis -view_full_net -glvis_pause 1e-10 -order 4

  5. Nonlinear Coupling: Run hydronetwork EPANet Network (network -2 uses hydrnetwork reader for networks)
  mpiexec -np 2 ./ex8 -f ../hydronetwork-2021/cases/brazosRiver.inp -ts_monitor -network -2 -ts_max_steps 2 -dx 1000

  6. Linear Coupling Run hydronetwork EPANet Network (network -2 uses hydrnetwork reader for networks)
  mpiexec -np 2 ./ex8 -f ../hydronetwork-2021/cases/brazosRiver.inp -ts_monitor -network -2 -ts_max_steps 2 -dx 1000 -lincouple 


  A Note for the NetRP solvers, The ksp for linear solvers can be accessed by 
    -netrp_ksp 
  and the snes for nonlinear solvers can be accessed
    -netrp_snes
  and snes ksp by 
    -netrp_snes_ksp 

  By default SNES and KSP use PCLU only for the solve, and the symbolic factorization is resused between calls. 
*/

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  PetscCall(DGNetworkMonitorView(monitor, x));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDGNetworkMonitor_GLVis(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis(monitor, x));
  PetscFunctionReturn(0);
}

PetscErrorCode TSDGNetworkMonitor_GLVis_NET(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis_NET(monitor, x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  char                   physname[256] = "shallow", outputfile[256];
  PetscFunctionList      physics       = 0;
  MPI_Comm               comm;
  TS                     ts;
  DGNetwork              dgnet;
  PetscInt               maxorder = 2, systemsize, rhsversion = 2;
  PetscReal              maxtime, norm;
  PetscMPIInt            size, rank;
  PetscBool              limit = PETSC_TRUE, view3d = PETSC_FALSE, viewglvis = PETSC_FALSE, glvismode = PETSC_FALSE, viewfullnet = PETSC_FALSE, savefinal = PETSC_FALSE;
  DGNetworkMonitor       monitor = NULL;
  DGNetworkMonitor_Glvis monitor_gl;
  PetscViewer            vecbinary;
  Vec                    Xtest;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics, "shallow", PhysicsCreate_Shallow));

  PetscCall(PetscCalloc1(1, &dgnet)); /* Replace with proper dgnet creation function */
  /* Set default values */
  dgnet->comm           = comm;
  dgnet->cfl            = 0.9;
  dgnet->networktype    = 6;
  dgnet->hratio         = 1;
  maxtime               = 2.0;
  dgnet->Mx             = 10;
  dgnet->initial        = 1;
  dgnet->ndaughters     = 2;
  dgnet->length         = 10.0;
  dgnet->view           = PETSC_FALSE;
  dgnet->jumptol        = 0.5;
  dgnet->diagnosticlow  = 0.5;
  dgnet->diagnosticup   = 1e-4;
  dgnet->linearcoupling = PETSC_FALSE;
  dgnet->M              = 50;
  dgnet->edgethickness  = 1;
  dgnet->dx             = 1000;

  /* Command Line Options */
  PetscOptionsBegin(comm, NULL, "DGNetwork solver options", "");
  PetscCall(PetscOptionsFList("-physics", "Name of physics model to use", "", physics, physname, physname, sizeof(physname), NULL));
  PetscCall(PetscOptionsInt("-initial", "Initial Condition (depends on the physics)", "", dgnet->initial, &dgnet->initial, NULL));
  PetscCall(PetscOptionsInt("-network", "Network topology to load, along with boundary condition information", "", dgnet->networktype, &dgnet->networktype, NULL));
  PetscCall(PetscOptionsReal("-cfl", "CFL number to time step at", "", dgnet->cfl, &dgnet->cfl, NULL));
  PetscCall(PetscOptionsReal("-length", "Length of Edges in the Network", "", dgnet->length, &dgnet->length, NULL));
  PetscCall(PetscOptionsInt("-Mx", "Smallest number of cells for an edge", "", dgnet->Mx, &dgnet->Mx, NULL));
  PetscCall(PetscOptionsInt("-ndaughters", "Number of daughter branches for network type 3", "", dgnet->ndaughters, &dgnet->ndaughters, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of the DG Basis", "", maxorder, &maxorder, NULL));
  PetscCall(PetscOptionsInt("-rhsversion", "Version of the RHS to use", "", rhsversion, &rhsversion, NULL));
  PetscCall(PetscOptionsBool("-savefinal", "View GLVis of Edge", "", savefinal, &savefinal, NULL));
  PetscCall(PetscOptionsBool("-view", "View the DG solution", "", dgnet->view, &dgnet->view, NULL));
  PetscCall(PetscOptionsBool("-uselimiter", "Use a limiter for the DG solution", "", limit, &limit, NULL));
  PetscCall(PetscOptionsReal("-jumptol", "Set jump tolerance for lame one-sided limiter", "", dgnet->jumptol, &dgnet->jumptol, NULL));
  PetscCall(PetscOptionsBool("-lincouple", "Use lax curve diagnostic for coupling", "", dgnet->linearcoupling, &dgnet->linearcoupling, NULL));
  PetscCall(PetscOptionsBool("-view_dump", "Dump the Glvis view or socket", "", glvismode, &glvismode, NULL));
  PetscCall(PetscOptionsBool("-view_3d", "View a 3d version of edge", "", view3d, &view3d, NULL));
  PetscCall(PetscOptionsBool("-view_glvis", "View GLVis of Edge", "", viewglvis, &viewglvis, NULL));
  PetscCall(PetscOptionsBool("-view_full_net", "View GLVis of Entire Network", "", viewfullnet, &viewfullnet, NULL));
  PetscCall(PetscOptionsReal("-dx", "Size of Cells in some cases", "", dgnet->dx, &dgnet->dx, NULL));
  PetscOptionsEnd();
  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics, physname, &r));
    if (!r) SETERRQ(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(dgnet));
  }
  PetscCall(PetscMalloc1(dgnet->physics.dof, &dgnet->physics.order)); /* should be constructed by physics */
  PetscCall(MakeOrder(dgnet->physics.dof, dgnet->physics.order, maxorder));

  /* Generate Network Data */
  PetscCall(DGNetworkCreate(dgnet, dgnet->networktype, dgnet->Mx));
  /* Create DMNetwork */
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &dgnet->network));
  /* Set Network Data into the DMNetwork (on proc[0]) */
  PetscCall(DGNetworkSetComponents(dgnet));
  /* Delete unneeded data in dgnet */
  PetscCall(DGNetworkCleanUp(dgnet));
  PetscCall(DGNetworkBuildTabulation(dgnet));
  PetscCall(DMNetworkDistribute(&dgnet->network, 0));

  /* Create Vectors */
  PetscCall(DGNetworkCreateVectors(dgnet));
  /* Set up component dynamic data structures */
  PetscCall(DGNetworkBuildDynamic(dgnet));
  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorCreate_Glvis(dgnet, &monitor_gl));
      if (viewfullnet) {
        PetscCall(DGNetworkMonitorAdd_Glvis_2D_NET(monitor_gl, "localhost", glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
      } else {
        if (view3d) {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis_3D(dgnet, monitor_gl, glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        } else {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis(dgnet, monitor_gl, glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        }
      }
    } else {
      PetscCall(DGNetworkMonitorCreate(dgnet, &monitor));
      PetscCall(DGNetworkAddMonitortoEdges(dgnet, monitor));
    }
  }
  /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to
   set all the physics parts at once) */

  PetscCall(RiemannSolverCreate(dgnet->comm, &dgnet->physics.rs));
  PetscCall(RiemannSolverSetApplicationContext(dgnet->physics.rs, dgnet->physics.user));
  PetscCall(RiemannSolverSetFromOptions(dgnet->physics.rs));
  PetscCall(RiemannSolverSetFluxEig(dgnet->physics.rs, dgnet->physics.fluxeig));
  PetscCall(RiemannSolverSetRoeAvgFunct(dgnet->physics.rs, dgnet->physics.roeavg));
  PetscCall(RiemannSolverSetRoeMatrixFunct(dgnet->physics.rs, dgnet->physics.roemat));
  PetscCall(RiemannSolverSetEigBasis(dgnet->physics.rs, dgnet->physics.eigbasis));
  PetscCall(RiemannSolverSetFlux(dgnet->physics.rs, 1, dgnet->physics.dof, dgnet->physics.flux2));
  PetscCall(RiemannSolverSetLaxCurve(dgnet->physics.rs, dgnet->physics.laxcurve));
  PetscCall(RiemannSolverSetUp(dgnet->physics.rs));

  /* Set up NetRS */
  PetscCall(DGNetworkAssignNetRS(dgnet));
  PetscCall(DGNetworkProject(dgnet, dgnet->X, 0.0));
  PetscCall(VecGetSize(dgnet->X, &systemsize));
  PetscCall(PetscPrintf(comm, "\nWe have %" PetscInt_FMT " Dofs\n\n", systemsize));
  /* Create a time-stepping object */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetApplicationContext(ts, dgnet));

  switch (rhsversion) {
  case 0:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS, dgnet));
    break;
  case 1:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_V2, dgnet));
    break;
  case 2:
  default:
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_V3, dgnet));
  }

  PetscCall(TSSetType(ts, TSSSP));
  PetscCall(TSSetMaxTime(ts, maxtime));
  PetscCall(TSSetMaxSteps(ts, 10));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, dgnet->cfl / dgnet->Mx / (2 * maxorder + 1)));
  PetscCall(TSSetFromOptions(ts)); /* Take runtime options */
  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      if (viewfullnet) {
        PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor_GLVis_NET, monitor_gl, NULL));
      } else {
        PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor_GLVis, monitor_gl, NULL));
      }
    } else {
      PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL));
    }
  }
  if (limit) {
    /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */
    PetscCall(DGNetlimiter(ts, 0, 0, &dgnet->X));
    PetscCall(TSSetPostStage(ts, DGNetlimiter));
  }
  PetscCall(TSSolve(ts, dgnet->X));

  PetscCall(VecDuplicate(dgnet->X, &Xtest));
  PetscCall(VecCopy(dgnet->X, Xtest));

  /* resolve with tested DGNetRHS */
  PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS, dgnet));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(DGNetworkProject(dgnet, dgnet->X, 0.0));

  PetscCall(TSSolve(ts, dgnet->X));

  PetscCall(VecAXPY(Xtest, -1, dgnet->X));

  PetscCall(VecNorm(Xtest, NORM_1, &norm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Difference between tested version 0 and version %" PetscInt_FMT " for DGNetRHS\n Norm: %e \n", rhsversion, norm));

  if (savefinal) {
    PetscCall(PetscSNPrintf(outputfile, 256, "ex8output_P%i_%i", maxorder, dgnet->Mx));
    PetscCall(PetscViewerBinaryOpen(comm, outputfile, FILE_MODE_WRITE, &vecbinary));
    PetscCall(VecView(dgnet->X, vecbinary));
    PetscCall(PetscViewerDestroy(&vecbinary));
  }

  /* Clean up */
  if (dgnet->view && size == 1) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorDestroy_Glvis(&monitor_gl));
    } else {
      PetscCall(DGNetworkMonitorDestroy(&monitor));
    }
  }
  PetscCall(RiemannSolverDestroy(&dgnet->physics.rs));
  PetscCall(PetscFree(dgnet->physics.order));
  PetscCall(DGNetworkDestroy(dgnet)); /* Destroy all data within the network and within dgnet */
  PetscCall(DMDestroy(&dgnet->network));
  PetscCall(PetscFree(dgnet));

  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
}
