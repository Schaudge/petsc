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
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsSample_1(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = 1;
      break; 
    case 1:
      u[0] = x; 
      break;
    case 2: 
      u[0] = PetscPowReal(x,2);
      break;
    case 3: 
      u[0] = PetscPowReal(x,3); 
      break;
    case 4: 
      u[0] = PetscSinReal(PETSC_PI*x);
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsSample_2(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
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
      u[0] = PetscPowReal(x,2);
      u[1] = 1; 
      break;
    case 3: 
      u[0] = PetscPowReal(x,3); 
      u[1] = 1;
      break;
    case 4: 
      u[0] = PetscSinReal(PETSC_PI*x); 
      u[1] = 1; 
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode Physics_CreateDummy(DGNetwork dgnet,PetscInt dof, PetscInt *order)
{
  PetscInt i;
  PetscErrorCode ierr; 

  PetscFunctionBeginUser;
  dgnet->physics.dof            = dof;
  dgnet->physics.order          = order; 
  dgnet->physics.destroy        = PhysicsDestroy_NoFree_Net;

  switch(dof) {
    case 1:
        dgnet->physics.samplenetwork = PhysicsSample_1;
        break;
    case 2: 
        dgnet->physics.samplenetwork = PhysicsSample_2; 
        break; 
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"No initial conditions available for dofs greater than 2");
  }
  for(i=0;i<dof;i++) {
    ierr = PetscStrallocpy("TEST",&dgnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode OrderCopyTest(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
static PetscErrorCode OrderTabTest(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = i;
  PetscFunctionReturn(0);
}
static PetscErrorCode OrderInterestingTest(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = PetscFloorReal(i/2.0); 
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  MPI_Comm          comm;
  DGNetwork         dgnet_fine,dgnet_coarse;
  PetscInt          i,*order_fine,*order_coarse,dof=1,maxdegree_fine=1,maxdegree_coarse=1,mode=0;
  PetscReal         *norm;
  PetscBool         view = PETSC_FALSE;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscViewer       viewer; 
  DGNetworkMonitor  monitor_fine=NULL,monitor_coarse=NULL,monitor_fine2=NULL,monitor_fine3=NULL;
  Vec               projection,error; 

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc2(1,&dgnet_fine,1,&dgnet_coarse);CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet_fine,sizeof(*dgnet_fine));CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet_fine,sizeof(*dgnet_coarse));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Set default values */
  dgnet_fine->comm           = comm;
  dgnet_coarse->comm         = comm;
  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
  dgnet_fine->networktype = 6;
  dgnet_fine->initial     = 1;
  dgnet_fine->Mx          = 20; 
  dgnet_coarse->Mx        = 4; 
  ierr = PetscOptionsBegin(comm,NULL,"DGNetwork options","");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-deg_fine","Degree for Fine tabulation","",maxdegree_fine,&maxdegree_fine,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-deg_coarse","Degree for Coarse tabulation","",maxdegree_coarse,&maxdegree_coarse,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-fields","Number of Fields to Generate Tabulations for","",dof,&dof,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mode","Mode to compute the orders for each field (enter int in [0-2])","",mode,&mode,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Mx_fine","Smallest number of cells for an edge","",dgnet_fine->Mx,&dgnet_fine->Mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Mx_coarse","Smallest number of cells for an edge","",dgnet_coarse->Mx,&dgnet_coarse->Mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view","view the fields on the network","",view,&view,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",dgnet_fine->initial,&dgnet_fine->initial,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",dgnet_fine->networktype,&dgnet_fine->networktype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  dgnet_coarse->networktype = dgnet_fine->networktype;
  dgnet_coarse->initial = dgnet_fine->initial;
  ierr = PetscMalloc2(dof,&order_fine,dof,&order_coarse);CHKERRQ(ierr);
  switch (mode)
  {
  default: 
  case 0:
    OrderCopyTest(dof,order_fine,maxdegree_fine);
    OrderCopyTest(dof,order_coarse,maxdegree_coarse);
    break;
  case 1: 
    OrderTabTest(dof,order_fine,maxdegree_fine);
    OrderTabTest(dof,order_coarse,maxdegree_coarse);
    break; 
  case 2:
    OrderInterestingTest(dof,order_fine,maxdegree_fine);
    OrderInterestingTest(dof,order_coarse,maxdegree_coarse);
    break;
  }


  viewer = PETSC_VIEWER_STDOUT_(comm);
  
    ierr = Physics_CreateDummy(dgnet_fine,dof,order_fine);CHKERRQ(ierr);
    ierr = Physics_CreateDummy(dgnet_coarse,dof,order_coarse);CHKERRQ(ierr);

    ierr = DGNetworkCreate(dgnet_fine,dgnet_fine->networktype,dgnet_fine->Mx);CHKERRQ(ierr);
    ierr = DGNetworkCreate(dgnet_coarse,dgnet_coarse->networktype,dgnet_coarse->Mx);CHKERRQ(ierr);
    /* Create DMNetwork */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dgnet_fine->network);CHKERRQ(ierr);
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dgnet_coarse->network);CHKERRQ(ierr);
    
    if (size == 1 && view) {
        ierr = DGNetworkMonitorCreate(dgnet_fine,&monitor_fine);CHKERRQ(ierr);
        ierr = DGNetworkMonitorCreate(dgnet_fine,&monitor_fine2);CHKERRQ(ierr);
        ierr = DGNetworkMonitorCreate(dgnet_fine,&monitor_fine3);CHKERRQ(ierr);
        ierr = DGNetworkMonitorCreate(dgnet_coarse,&monitor_coarse);CHKERRQ(ierr);
    }
    /* Set Network Data into the DMNetwork (on proc[0]) */
    ierr = DGNetworkSetComponents(dgnet_fine);CHKERRQ(ierr);
    ierr = DGNetworkSetComponents(dgnet_coarse);CHKERRQ(ierr);

    /* Delete unneeded data in dgnet */
    ierr = DGNetworkCleanUp(dgnet_fine);CHKERRQ(ierr);
    ierr = DGNetworkCleanUp(dgnet_coarse);CHKERRQ(ierr);

    ierr = DGNetworkBuildTabulation(dgnet_fine);CHKERRQ(ierr);
    ierr = DGNetworkBuildTabulation(dgnet_coarse);CHKERRQ(ierr);

    ierr = DMNetworkDistribute(&dgnet_fine->network,0);CHKERRQ(ierr);
    ierr = DMNetworkDistribute(&dgnet_coarse->network,0);CHKERRQ(ierr);

    /* Create Vectors */
    ierr = DGNetworkCreateVectors(dgnet_fine);CHKERRQ(ierr);
    ierr = DGNetworkCreateVectors(dgnet_coarse);CHKERRQ(ierr);

    /* Set up component dynamic data structures */
    ierr = DGNetworkBuildDynamic(dgnet_fine);CHKERRQ(ierr);
    ierr = DGNetworkBuildDynamic(dgnet_coarse);CHKERRQ(ierr);

    if (view) {
        ierr = DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine);CHKERRQ(ierr);
        ierr = DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine2);CHKERRQ(ierr);
        ierr = DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine3);CHKERRQ(ierr);

        ierr = DGNetworkAddMonitortoEdges(dgnet_coarse,monitor_coarse);CHKERRQ(ierr);
    }

    /* Project the initial conditions on the fine and coarse grids */
    ierr = DGNetworkProject(dgnet_fine,dgnet_fine->X,0.0);CHKERRQ(ierr);
    ierr = DGNetworkProject(dgnet_coarse,dgnet_coarse->X,0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(dgnet_fine->X,&projection);CHKERRQ(ierr);
    ierr = VecDuplicate(dgnet_fine->X,&error);CHKERRQ(ierr);

    if (size == 1 && view) {
        ierr = DGNetworkMonitorView(monitor_fine,dgnet_fine->X);CHKERRQ(ierr);
        ierr = DGNetworkMonitorView(monitor_coarse,dgnet_coarse->X);CHKERRQ(ierr);
    }
    /* Project from the coarse grid to the fine grid */
    ierr = DGNetworkProject_Coarse_To_Fine(dgnet_fine,dgnet_coarse,dgnet_coarse->X,projection);CHKERRQ(ierr);
    /* Compute Error in the projection */
    ierr = VecWAXPY(error,-1,dgnet_fine->X,projection);CHKERRQ(ierr);
    if (size == 1 && view) {
        ierr = DGNetworkMonitorView(monitor_fine2,projection);CHKERRQ(ierr);
        ierr = DGNetworkMonitorView(monitor_fine3,error);CHKERRQ(ierr);
    }
    /*compute norm of the error */
    ierr = PetscMalloc1(dof,&norm);CHKERRQ(ierr);
    ierr = DGNetworkNormL2(dgnet_fine,error,norm);CHKERRQ(ierr);
    for (i=0;i<dof;i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
            "DGNET Error in Projection:   %g \n",norm[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&projection);CHKERRQ(ierr);
    ierr = VecDestroy(&error);CHKERRQ(ierr);
        /* Clean up */
    if (view && size==1) {
        ierr = DGNetworkMonitorDestroy(&monitor_fine);CHKERRQ(ierr);
        ierr = DGNetworkMonitorDestroy(&monitor_coarse);CHKERRQ(ierr);
        ierr = DGNetworkMonitorDestroy(&monitor_fine2);CHKERRQ(ierr);
        ierr = DGNetworkMonitorDestroy(&monitor_fine3);CHKERRQ(ierr);
    }

    ierr = DGNetworkDestroy(dgnet_fine);CHKERRQ(ierr); /* Destroy all data within the network and within dgnet */
    ierr = DGNetworkDestroy(dgnet_coarse);CHKERRQ(ierr); /* Destroy all data within the network and within dgnet */
    ierr = DMDestroy(&dgnet_fine->network);CHKERRQ(ierr);
    ierr = DMDestroy(&dgnet_coarse->network);CHKERRQ(ierr);
  ierr = PetscFree2(dgnet_coarse,dgnet_fine);CHKERRQ(ierr);
  ierr = PetscFree2(order_coarse,order_fine);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
