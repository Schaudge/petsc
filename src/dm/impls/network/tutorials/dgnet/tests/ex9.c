static const char help[] = "Tests the Coarse to Fine Projection";
/*

  TODO ADD EXAMPLE RUNS HERE 

  Contributed by: Aidan Hamilton <aidan@udel.edu>

*/

#include <petscdm.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
PetscErrorCode PhysicsDestroy_NoFree_Net(void *vctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PhysicsSample_1(void *vctx, PetscInt initial, PetscReal t, PetscReal x, PetscReal *u, PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Exact solutions not implemented for t > 0");
  switch (initial) {
  case 0:
    u[0] = 1;
    break;
  case 1:
    u[0] = x;
    break;
  case 2:
    u[0] = PetscPowReal(x, 2);
    break;
  case 3:
    u[0] = PetscPowReal(x, 3);
    break;
  case 4:
    u[0] = PetscSinReal(PETSC_PI * x);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "unknown initial condition");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PhysicsSample_2(void *vctx, PetscInt initial, PetscReal t, PetscReal x, PetscReal *u, PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Exact solutions not implemented for t > 0");
  switch (initial) {
  case 0:
    u[0] = 1;
    u[1] = 1;
    break;
  case 1:
    u[0] = x;
    u[1] = 1;
    break;
  case 2:
    u[0] = PetscPowReal(x, 2);
    u[1] = 1;
    break;
  case 3:
    u[0] = PetscPowReal(x, 3);
    u[1] = 1;
    break;
  case 4:
    u[0] = PetscSinReal(PETSC_PI * x);
    u[1] = 1;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "unknown initial condition");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode Physics_CreateDummy(DGNetwork dgnet, PetscInt dof, PetscInt *order)
{
  PetscInt i;

  PetscFunctionBeginUser;
  dgnet->physics.dof     = dof;
  dgnet->physics.order   = order;
  dgnet->physics.destroy = PhysicsDestroy_NoFree_Net;

  switch (dof) {
  case 1:
    dgnet->physics.samplenetwork = PhysicsSample_1;
    break;
  case 2:
    dgnet->physics.samplenetwork = PhysicsSample_2;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "No initial conditions available for dofs greater than 2");
  }
  for (i = 0; i < dof; i++) { PetscCall(PetscStrallocpy("TEST", &dgnet->physics.fieldname[i])); }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode OrderCopyTest(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;
  for (i = 0; i < dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode OrderTabTest(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;
  for (i = 0; i < dof; i++) order[i] = i;
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode OrderInterestingTest(PetscInt dof, PetscInt *order, PetscInt maxdegree)
{
  PetscInt i;
  for (i = 0; i < dof; i++) order[i] = PetscFloorReal(i / 2.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc, char *argv[])
{
  MPI_Comm         comm;
  DGNetwork        dgnet_fine, dgnet_coarse;
  PetscInt         i, *order_fine, *order_coarse, dof = 1, maxdegree_fine = 1, maxdegree_coarse = 1, mode = 0;
  PetscReal       *norm;
  PetscBool        view = PETSC_FALSE;
  PetscErrorCode   ierr;
  PetscMPIInt      size, rank;
  PetscViewer      viewer;
  DGNetworkMonitor monitor_fine = NULL, monitor_coarse = NULL, monitor_fine2 = NULL, monitor_fine3 = NULL;
  Vec              projection, error;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMalloc2(1, &dgnet_fine, 1, &dgnet_coarse));
  PetscCall(PetscMemzero(dgnet_fine, sizeof(*dgnet_fine)));
  PetscCall(PetscMemzero(dgnet_fine, sizeof(*dgnet_coarse)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Set default values */
  dgnet_fine->comm   = comm;
  dgnet_coarse->comm = comm;
  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
  dgnet_fine->networktype = 6;
  dgnet_fine->initial     = 1;
  dgnet_fine->Mx          = 20;
  dgnet_coarse->Mx        = 4;
  ierr                    = PetscOptionsBegin(comm, NULL, "DGNetwork options", "");
  CHKERRQ(ierr);
  PetscCall(PetscOptionsInt("-deg_fine", "Degree for Fine tabulation", "", maxdegree_fine, &maxdegree_fine, NULL));
  PetscCall(PetscOptionsInt("-deg_coarse", "Degree for Coarse tabulation", "", maxdegree_coarse, &maxdegree_coarse, NULL));
  PetscCall(PetscOptionsInt("-fields", "Number of Fields to Generate Tabulations for", "", dof, &dof, NULL));
  PetscCall(PetscOptionsInt("-mode", "Mode to compute the orders for each field (enter int in [0-2])", "", mode, &mode, NULL));
  PetscCall(PetscOptionsInt("-Mx_fine", "Smallest number of cells for an edge", "", dgnet_fine->Mx, &dgnet_fine->Mx, NULL));
  PetscCall(PetscOptionsInt("-Mx_coarse", "Smallest number of cells for an edge", "", dgnet_coarse->Mx, &dgnet_coarse->Mx, NULL));
  PetscCall(PetscOptionsBool("-view", "view the fields on the network", "", view, &view, NULL));
  PetscCall(PetscOptionsInt("-initial", "Initial Condition (depends on the physics)", "", dgnet_fine->initial, &dgnet_fine->initial, NULL));
  PetscCall(PetscOptionsInt("-network", "Network topology to load, along with boundary condition information", "", dgnet_fine->networktype, &dgnet_fine->networktype, NULL));
  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);

  dgnet_coarse->networktype = dgnet_fine->networktype;
  dgnet_coarse->initial     = dgnet_fine->initial;
  PetscCall(PetscMalloc2(dof, &order_fine, dof, &order_coarse));
  switch (mode) {
  default:
  case 0:
    OrderCopyTest(dof, order_fine, maxdegree_fine);
    OrderCopyTest(dof, order_coarse, maxdegree_coarse);
    break;
  case 1:
    OrderTabTest(dof, order_fine, maxdegree_fine);
    OrderTabTest(dof, order_coarse, maxdegree_coarse);
    break;
  case 2:
    OrderInterestingTest(dof, order_fine, maxdegree_fine);
    OrderInterestingTest(dof, order_coarse, maxdegree_coarse);
    break;
  }

  viewer = PETSC_VIEWER_STDOUT_(comm);

  PetscCall(Physics_CreateDummy(dgnet_fine, dof, order_fine));
  PetscCall(Physics_CreateDummy(dgnet_coarse, dof, order_coarse));

  PetscCall(DGNetworkCreate(dgnet_fine, dgnet_fine->networktype, dgnet_fine->Mx));
  PetscCall(DGNetworkCreate(dgnet_coarse, dgnet_coarse->networktype, dgnet_coarse->Mx));
  /* Create DMNetwork */
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &dgnet_fine->network));
  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD, &dgnet_coarse->network));

  if (size == 1 && view) {
    PetscCall(DGNetworkMonitorCreate(dgnet_fine, &monitor_fine));
    PetscCall(DGNetworkMonitorCreate(dgnet_fine, &monitor_fine2));
    PetscCall(DGNetworkMonitorCreate(dgnet_fine, &monitor_fine3));
    PetscCall(DGNetworkMonitorCreate(dgnet_coarse, &monitor_coarse));
  }
  /* Set Network Data into the DMNetwork (on proc[0]) */
  PetscCall(DGNetworkSetComponents(dgnet_fine));
  PetscCall(DGNetworkSetComponents(dgnet_coarse));

  /* Delete unneeded data in dgnet */
  PetscCall(DGNetworkCleanUp(dgnet_fine));
  PetscCall(DGNetworkCleanUp(dgnet_coarse));

  PetscCall(DGNetworkBuildTabulation(dgnet_fine));
  PetscCall(DGNetworkBuildTabulation(dgnet_coarse));

  PetscCall(DMNetworkDistribute(&dgnet_fine->network, 0));
  PetscCall(DMNetworkDistribute(&dgnet_coarse->network, 0));

  /* Create Vectors */
  PetscCall(DGNetworkCreateVectors(dgnet_fine));
  PetscCall(DGNetworkCreateVectors(dgnet_coarse));

  /* Set up component dynamic data structures */
  PetscCall(DGNetworkBuildDynamic(dgnet_fine));
  PetscCall(DGNetworkBuildDynamic(dgnet_coarse));

  if (view) {
    PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine, monitor_fine));
    PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine, monitor_fine2));
    PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine, monitor_fine3));

    PetscCall(DGNetworkAddMonitortoEdges(dgnet_coarse, monitor_coarse));
  }

  /* Project the initial conditions on the fine and coarse grids */
  PetscCall(DGNetworkProject(dgnet_fine, dgnet_fine->X, 0.0));
  PetscCall(DGNetworkProject(dgnet_coarse, dgnet_coarse->X, 0.0));
  PetscCall(VecDuplicate(dgnet_fine->X, &projection));
  PetscCall(VecDuplicate(dgnet_fine->X, &error));

  if (size == 1 && view) {
    PetscCall(DGNetworkMonitorView(monitor_fine, dgnet_fine->X));
    PetscCall(DGNetworkMonitorView(monitor_coarse, dgnet_coarse->X));
  }
  /* Project from the coarse grid to the fine grid */
  PetscCall(DGNetworkProject_Coarse_To_Fine(dgnet_fine, dgnet_coarse, dgnet_coarse->X, projection));
  /* Compute Error in the projection */
  PetscCall(VecWAXPY(error, -1, dgnet_fine->X, projection));
  if (size == 1 && view) {
    PetscCall(DGNetworkMonitorView(monitor_fine2, projection));
    PetscCall(DGNetworkMonitorView(monitor_fine3, error));
  }
  /*compute norm of the error */
  PetscCall(PetscMalloc1(dof, &norm));
  PetscCall(DGNetworkNormL2(dgnet_fine, error, norm));
  for (i = 0; i < dof; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "DGNET Error in Projection:   %g \n", norm[i]);
    CHKERRQ(ierr);
  }
  PetscCall(VecDestroy(&projection));
  PetscCall(VecDestroy(&error));
  /* Clean up */
  if (view && size == 1) {
    PetscCall(DGNetworkMonitorDestroy(&monitor_fine));
    PetscCall(DGNetworkMonitorDestroy(&monitor_coarse));
    PetscCall(DGNetworkMonitorDestroy(&monitor_fine2));
    PetscCall(DGNetworkMonitorDestroy(&monitor_fine3));
  }

  PetscCall(DGNetworkDestroy(dgnet_fine));   /* Destroy all data within the network and within dgnet */
  PetscCall(DGNetworkDestroy(dgnet_coarse)); /* Destroy all data within the network and within dgnet */
  PetscCall(DMDestroy(&dgnet_fine->network));
  PetscCall(DMDestroy(&dgnet_coarse->network));
  PetscCall(PetscFree2(dgnet_coarse, dgnet_fine));
  PetscCall(PetscFree2(order_coarse, order_fine));
  PetscCall(PetscFinalize());
  return 0;
}
