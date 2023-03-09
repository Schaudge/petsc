static const char help[] = "Traffic Flow Convergence Test for the DGNetwork. \n\
Uses Method of characteristics to generate a true solution.\n\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include "../physics.h"

/*
  Example:
  1. 4 levels of convergence with P^4 DG basis and X viewing:
    mpiexec -n 1 ./ex4 -convergence 4 -order 4 -view -ts_ssp_type rk104

  2. requires: GLVis
     4 levels of convergence with P^4 DG basis and GLVis full network view:
    mpiexec -n 1 ./ex4 -convergence 4 -order 4 -view -ts_ssp_type rk104 -view_glvis -view_full_net -glvis_pause 1e-10
*/

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  PetscCall(DGNetworkMonitorView(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDGNetworkMonitor_GLVis(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDGNetworkMonitor_GLVis_NET(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  DGNetworkMonitor_Glvis monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  PetscCall(DGNetworkMonitorView_Glvis_NET(monitor, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  char                   physname[256] = "traffic";
  PetscFunctionList      physics       = 0;
  MPI_Comm               comm;
  TS                     ts;
  DGNetwork              dgnet;
  PetscInt               convergencelevel = 3, maxorder = 1, *order, n, i, j;
  PetscReal              maxtime, *norm;
  PetscErrorCode         ierr;
  PetscMPIInt            size, rank;
  DGNetworkMonitor       monitor    = NULL;
  DGNetworkMonitor_Glvis monitor_gl = NULL;
  Vec                    Xtrue;
  PetscBool              glvismode = PETSC_FALSE, view3d = PETSC_FALSE, viewglvis = PETSC_FALSE, viewfullnet = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMalloc1(1, &dgnet));
  PetscCall(PetscMemzero(dgnet, sizeof(*dgnet)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics, "traffic", PhysicsCreate_Traffic));

  /* Set default values */
  dgnet->comm        = comm;
  dgnet->cfl         = 0.5;
  dgnet->networktype = 6;
  maxtime            = 0.5;
  dgnet->Mx          = 10;
  dgnet->initial     = 1;
  dgnet->ymin        = 0;
  dgnet->ymax        = 2.0;
  dgnet->ndaughters  = 2;
  dgnet->length      = 5.0;
  dgnet->view        = PETSC_FALSE;

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm, NULL, "Finite Volume solver options", "");
  CHKERRQ(ierr);
  PetscCall(PetscOptionsReal("-cfl", "CFL number to time step at", "", dgnet->cfl, &dgnet->cfl, NULL));
  PetscCall(PetscOptionsInt("-convergence", "Test convergence on meshes 2^3 - 2^n", "", convergencelevel, &convergencelevel, NULL));
  PetscCall(PetscOptionsInt("-order", "Order of the DG Basis", "", maxorder, &maxorder, NULL));
  PetscCall(PetscOptionsBool("-view", "View the DG solution", "", dgnet->view, &dgnet->view, NULL));
  PetscCall(PetscOptionsBool("-view_dump", "Dump the Glvis view or socket", "", glvismode, &glvismode, NULL));
  PetscCall(PetscOptionsBool("-view_3d", "View a 3d version of edge", "", view3d, &view3d, NULL));
  PetscCall(PetscOptionsBool("-view_glvis", "View GLVis of Edge", "", viewglvis, &viewglvis, NULL));
  PetscCall(PetscOptionsBool("-view_full_net", "View GLVis of Entire Network", "", viewfullnet, &viewfullnet, NULL));
  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);

  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics, physname, &r));
    if (!r) SETERRQ1(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(dgnet));
  }

  PetscCall(PetscMalloc1(dgnet->physics.dof * convergencelevel, &norm));
  PetscCall(PetscMalloc1(dgnet->physics.dof, &order));
  PetscCall(MakeOrder(dgnet->physics.dof, order, maxorder));
  /*
  TODO : Build a better physics object with destructor (i.e. use PETSC DS? )
  */
  PetscCall((*dgnet->physics.destroy)(dgnet->physics.user));
  for (i = 0; i < dgnet->physics.dof; i++) { PetscCall(PetscFree(dgnet->physics.fieldname[i])); }
  for (n = 0; n < convergencelevel; ++n) {
    dgnet->Mx = PetscPowInt(2, n);
    {
      PetscErrorCode (*r)(DGNetwork);
      PetscCall(PetscFunctionListFind(physics, physname, &r));
      if (!r) SETERRQ1(PETSC_COMM_SELF, 1, "Physics '%s' not found", physname);
      /* Create the physics, will set the number of fields and their names */
      PetscCall((*r)(dgnet));
    }

    dgnet->physics.order = order;
    /* Generate Network Data */
    PetscCall(DGNetworkCreate(dgnet, dgnet->networktype, dgnet->Mx));
    /* Create DMNetwork */
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &dgnet->network));
    if (size == 1 && dgnet->view) {
      if (viewglvis) {
        PetscCall(DGNetworkMonitorCreate_Glvis(dgnet, &monitor_gl));
      } else {
        PetscCall(DGNetworkMonitorCreate(dgnet, &monitor));
      }
    }
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

    if (viewglvis) {
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
      PetscCall(DGNetworkAddMonitortoEdges(dgnet, monitor));
    }

    /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to
       set all the physics parts at once) */
    PetscCall(RiemannSolverCreate(dgnet->comm, &dgnet->physics.rs));
    PetscCall(RiemannSolverSetApplicationContext(dgnet->physics.rs, dgnet->physics.user));
    PetscCall(RiemannSolverSetFromOptions(dgnet->physics.rs));
    PetscCall(RiemannSolverSetFluxEig(dgnet->physics.rs, dgnet->physics.fluxeig));
    PetscCall(RiemannSolverSetFlux(dgnet->physics.rs, 1, dgnet->physics.dof, dgnet->physics.flux2));
    PetscCall(RiemannSolverSetUp(dgnet->physics.rs));

    PetscCall(DGNetworkAssignNetRS(dgnet, dgnet->physics.rs, NULL, -1)); /* No Error Estimation or adapativity */

    /* Create a time-stepping object */
    PetscCall(TSCreate(comm, &ts));
    PetscCall(TSSetDM(ts, dgnet->network));
    PetscCall(TSSetApplicationContext(ts, dgnet));
    PetscCall(TSSetRHSFunction(ts, NULL, DGNetRHS_NETRSVERSION2, dgnet));
    PetscCall(TSSetType(ts, TSSSP));
    PetscCall(TSSetMaxTime(ts, maxtime));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

    /* Compute initial conditions and starting time step */
    PetscCall(DGNetworkProject(dgnet, dgnet->X, 0.0));
    ierr = TSSetTimeStep(ts, dgnet->cfl / PetscPowReal(2.0, n) / (2 * maxorder + 1));
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
    /* Evolve the PDE network in time */
    PetscCall(TSSolve(ts, dgnet->X));
    /* Compute true solution and compute norm of the difference with computed solution*/
    PetscCall(VecDuplicate(dgnet->X, &Xtrue));
    PetscCall(DGNetworkProject(dgnet, Xtrue, maxtime));
    PetscCall(VecAXPY(Xtrue, -1, dgnet->X));
    PetscCall(DGNetworkNormL2(dgnet, Xtrue, norm + (dgnet->physics.dof * (n))));
    PetscCall(VecDestroy(&Xtrue));

    /* Clean up */
    if (dgnet->view && size == 1) {
      if (viewglvis) {
        ierr = DGNetworkMonitorDestroy_Glvis(&monitor_gl);
      } else {
        ierr = DGNetworkMonitorDestroy(&monitor);
      }
    }
    PetscCall(DGNetworkDestroyNetRS(dgnet));
    PetscCall(RiemannSolverDestroy(&dgnet->physics.rs));
    PetscCall(DGNetworkDestroy(dgnet)); /* Destroy all data within the network and within fvnet */
    PetscCall(DMDestroy(&dgnet->network));
    PetscCall(TSDestroy(&ts));
  }

  for (j = 0; j < dgnet->physics.dof; j++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "DGNET:  Convergence Table for Variable %i \n"
                       "DGNET: |---h---||---Error---||---Order---| \n",
                       j);
    CHKERRQ(ierr);
    for (i = 0; i < convergencelevel; i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "DGNET: |  %g  |  %g  |  %g  | \n", 1.0 / PetscPowReal(2.0, i), norm[dgnet->physics.dof * i + j], !i ? NAN : -PetscLog2Real(norm[dgnet->physics.dof * i + j] / norm[dgnet->physics.dof * (i - 1) + j]));
      CHKERRQ(ierr);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }

  PetscCall(PetscFree(norm));
  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(PetscFree(dgnet));
  PetscCall(PetscFree(order));
  PetscCall(PetscFinalize(); CHKERRQ(ierr));
  return 0;
}
