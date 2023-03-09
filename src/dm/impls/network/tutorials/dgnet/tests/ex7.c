static const char help[] = "DGNET Coupling Test Example";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  PetscCall(DGNetworkMonitorView(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;
  for (i = 0; i < dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc, char *argv[])
{
  char              physname[256] = "shallow", errorestimator[256] = "lax";
  PetscFunctionList physics = 0, errest = 0;
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         dgnet;
  PetscInt          maxorder = 1;
  PetscReal         maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size, rank;
  PetscBool         limit   = PETSC_TRUE;
  DGNetworkMonitor  monitor = NULL;
  NRSErrorEstimator errorest;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics, "shallow", PhysicsCreate_Shallow));

  /* register error estimator functions */
  PetscCall(PetscFunctionListAdd(&errest, "roe", NetRSRoeErrorEstimate));
  PetscCall(PetscFunctionListAdd(&errest, "lax", NetRSLaxErrorEstimate));
  PetscCall(PetscFunctionListAdd(&errest, "taylor", NetRSTaylorErrorEstimate));

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
  dgnet->length         = 3.0;
  dgnet->view           = PETSC_FALSE;
  dgnet->jumptol        = 0.5;
  dgnet->laxcurve       = PETSC_FALSE;
  dgnet->diagnosticlow  = 0.5;
  dgnet->diagnosticup   = 1e-4;
  dgnet->adaptivecouple = PETSC_FALSE;
  dgnet->linearcoupling = PETSC_TRUE;
  /* Command Line Options */
  ierr = PetscOptionsBegin(comm, NULL, "DGNetwork solver options", "");
  CHKERRQ(ierr);
  PetscCall(PetscOptionsFList("-physics", "Name of physics model to use", "", physics, physname, physname, sizeof(physname), NULL));
  PetscCall(PetscOptionsFList("-errest", "", "", errest, errorestimator, errorestimator, sizeof(physname), NULL));
  PetscCall(PetscOptionsInt("-initial", "Initial Condition (depends on the physics)", "", dgnet->initial, &dgnet->initial, NULL));
  PetscCall(PetscOptionsInt("-network", "Network topology to load, along with boundary condition information", "", dgnet->networktype, &dgnet->networktype, NULL));
  PetscCall(PetscOptionsReal("-cfl", "CFL number to time step at", "", dgnet->cfl, &dgnet->cfl, NULL));
  PetscCall(PetscOptionsReal("-length", "Length of Edges in the Network", "", dgnet->length, &dgnet->length, NULL));
  PetscCall(PetscOptionsInt("-Mx", "Smallest number of cells for an edge", "", dgnet->Mx, &dgnet->Mx, NULL));
  PetscCall(PetscOptionsInt("-ndaughters", "Number of daughter branches for network type 3", "", dgnet->ndaughters, &dgnet->ndaughters, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of the DG Basis", "", maxorder, &maxorder, NULL));
  PetscCall(PetscOptionsBool("-view", "View the DG solution", "", dgnet->view, &dgnet->view, NULL));
  PetscCallQ(ierr);
  ierr = PetscOptionsBool("-uselimiter", "Use a limiter for the DG solution", "", limit, &limit, NULL);
  Q(PetscOptionsBool("-uselimiter", "Use a limiter for the DG solution", "", limit, &limit, NULL));
  PetscCall(PetscOptionsBool("-adaptivecouple", "Use adaptive Coupling for Netrs", "", dgnet->adaptivecouple, &dgnet->adaptivecouple, NULL));
  PetscCall(PetscOptionsBool("-lax", "Use lax curve diagnostic for coupling", "", dgnet->laxcurve, &dgnet->laxcurve, NULL));
  PetscCall(PetscOptionsReal("-jumptol", "Set jump tolerance for lame one-sided limiter", "", dgnet->jumptol, &dgnet->jumptol, NULL));
  PetscCall(PetscOptionsBool("-lincouple", "Use lax curve diagnostic for coupling", "", dgnet->linearcoupling, &dgnet->linearcoupling, NULL));

  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);
  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics, physname, &r));
    if (!r) SETERRQ1(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
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
  if (size == 1 && dgnet->view) {
    PetscCall(DGNetworkMonitorCreate(dgnet, &monitor));
    PetscCall(DGNetworkAddMonitortoEdges(dgnet, monitor));
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

  /* Create Vectors */
  PetscCall(DGNetworkCreateVectors(dgnet));
  /* Set up component dynamic data structures */
  PetscCall(DGNetworkBuildDynamic(dgnet));
  /* Set up NetRS */
  PetscCall(PetscFunctionListFind(errest, errorestimator, &errorest));
  PetscCall(DGNetworkAssignNetRS(dgnet, dgnet->physics.rs, errorest, 1));
  PetscCall(DGNetworkProject(dgnet, dgnet->X, 0.0));
  /* Create a time-stepping object */
  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetApplicationContext(ts, dgnet));
  PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_NETRSVERSION, dgnet));
  PetscCall(TSSetType(ts, TSSSP));
  PetscCall(TSSetMaxTime(ts, maxtime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, dgnet->cfl / dgnet->Mx / (2 * maxorder + 1)));
  PetscCall(TSSetFromOptions(ts)); /* Take runtime options */
  if (size == 1 && dgnet->view) { PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL)); }
  if (limit) {
    /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */
    PetscCall(DGNetlimiter(ts, 0, 0, &dgnet->X));
    PetscCall(TSSetPostStage(ts, DGNetlimiter));
  }
  /* Clean up the output directory (this output data should be redone */
  PetscCall(PetscRMTree("output"));
  PetscCall(PetscMkdir("output"));
  PetscCall(TSSolve(ts, dgnet->X));

  /* Clean up */
  if (dgnet->view && size == 1) { PetscCall(DGNetworkMonitorDestroy(&monitor)); }
  PetscCall(RiemannSolverDestroy(&dgnet->physics.rs));
  PetscCall(PetscFree(dgnet->physics.order));
  PetscCall(DGNetworkDestroyNetRS(dgnet));
  PetscCall(DGNetworkDestroy(dgnet)); /* Destroy all data within the network and within dgnet */
  PetscCall(DMDestroy(&dgnet->network));
  PetscCall(PetscFree(dgnet));

  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize(); CHKERRQ(ierr));
  return 0;
}
