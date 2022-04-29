static const char help[] = "Loads previously computed DGNet Solutions on Coarse and Fine Grids and then Computes their difference";
/*

  TODO ADD EXAMPLE RUNS HERE 

  Contributed by: Aidan Hamilton <aidan@udel.edu>

*/

#include <petscdm.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include "../physics.h"

static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i;

  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  MPI_Comm          comm;
  char              physname[256] = "shallow", coarsefile[256];
  PetscFunctionList physics = 0;
  DGNetwork         dgnet_fine,dgnet_coarse;
  PetscInt          i,*order_fine,*order_coarse,dof=1,maxdegree_fine=1,maxdegree_coarse=1,mode=0;
  PetscReal         *norm;
  PetscBool         view = PETSC_FALSE;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscViewer       viewer,coarseload,fineload;
  DGNetworkMonitor  monitor_fine=NULL,monitor_coarse=NULL,monitor_fine2=NULL,monitor_fine3=NULL;
  Vec               projection,error,coarse,fine;

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc2(1,&dgnet_fine,1,&dgnet_coarse);CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet_fine,sizeof(*dgnet_fine));CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet_fine,sizeof(*dgnet_coarse));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);


ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);

  /* Set default values */
  dgnet_fine->comm           = comm;
  dgnet_coarse->comm         = comm;
  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
   /* Set default values */
  dgnet_fine->comm           = comm;
  dgnet_fine->cfl            = 0.9;
  dgnet_fine->networktype    = 3;
  dgnet_fine->hratio         = 1;
  dgnet_fine->Mx             = 10;
  dgnet_fine->initial        = 1;
  dgnet_fine->ndaughters     = 2;
  dgnet_fine->length         = 10.0;
  dgnet_fine->view           = PETSC_FALSE;
  dgnet_fine->jumptol        = 0.5;
  dgnet_fine->diagnosticlow  = 0.5;
  dgnet_fine->diagnosticup   = 1e-4; 
  dgnet_fine->linearcoupling   = PETSC_FALSE;
  dgnet_fine->M                = 50;
  dgnet_fine->hratio           = 1; 
  dgnet_coarse->hratio         = dgnet_fine->hratio;


  dgnet_coarse->Mx             = 10;

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



  dgnet_coarse->comm           = dgnet_fine->comm;
  dgnet_coarse->cfl            =  dgnet_coarse->cfl ;
  dgnet_coarse->networktype    =  dgnet_coarse->networktype;
  dgnet_coarse->hratio         = dgnet_coarse->hratio;
  dgnet_coarse->initial        = dgnet_fine->initial;
  dgnet_coarse->ndaughters     = dgnet_fine->ndaughters;
  dgnet_coarse->length         = dgnet_fine->length;
  dgnet_coarse->view           = dgnet_fine->view;
  dgnet_coarse->networktype    = dgnet_fine->networktype;
  dgnet_coarse->initial = dgnet_fine->initial;
  ierr = PetscMalloc2(dof,&order_fine,dof,&order_coarse);CHKERRQ(ierr);

    /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(dgnet_fine);CHKERRQ(ierr);
    ierr = (*r)(dgnet_coarse);CHKERRQ(ierr);
  }
    ierr = PetscMalloc2(dgnet_fine->physics.dof,&dgnet_fine->physics.order,dgnet_coarse->physics.dof,&dgnet_coarse->physics.order);CHKERRQ(ierr); /* should be constructed by physics */

  ierr = MakeOrder(dgnet_fine->physics.dof,dgnet_fine->physics.order,maxdegree_fine);CHKERRQ(ierr);
  ierr = MakeOrder(dgnet_coarse->physics.dof,dgnet_coarse->physics.order,maxdegree_coarse);CHKERRQ(ierr);

    viewer = PETSC_VIEWER_STDOUT_(comm);

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

    /* Load Fine Vector and Coarse Vector */
    ierr = VecCreate(comm,&coarse);CHKERRQ(ierr);
    ierr = VecCreate(comm,&fine);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"ex8output_true",FILE_MODE_READ,&fineload);CHKERRQ(ierr);
    ierr = PetscSNPrintf(coarsefile,256,"ex8output_P%i_%i",maxdegree_coarse,dgnet_coarse->Mx);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,coarsefile,FILE_MODE_READ,&coarseload);CHKERRQ(ierr);
    
    ierr = VecLoad(fine,fineload);CHKERRQ(ierr);
    ierr = VecLoad(coarse,coarseload);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fineload);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&coarseload);CHKERRQ(ierr);

    ierr = VecDuplicate(dgnet_fine->X,&projection);CHKERRQ(ierr);
    ierr = VecDuplicate(dgnet_fine->X,&error);CHKERRQ(ierr);

    if (size == 1 && view) {
        ierr = DGNetworkMonitorView(monitor_fine,fine);CHKERRQ(ierr);
        ierr = DGNetworkMonitorView(monitor_coarse,coarse);CHKERRQ(ierr);
    }
    /* Project from the coarse grid to the fine grid */
    ierr = DGNetworkProject_Coarse_To_Fine(dgnet_fine,dgnet_coarse,coarse,projection);CHKERRQ(ierr);
    /* Compute Error in the projection */
    ierr = VecWAXPY(error,-1,fine,projection);CHKERRQ(ierr);
    if (size == 1 && view) {
        ierr = DGNetworkMonitorView(monitor_fine2,projection);CHKERRQ(ierr);
        ierr = DGNetworkMonitorView(monitor_fine3,error);CHKERRQ(ierr);
    }
    /*compute norm of the error */
    ierr = PetscMalloc1(dof,&norm);CHKERRQ(ierr);
    ierr = DGNetworkNormL2(dgnet_fine,error,norm);CHKERRQ(ierr);
    for (i=0;i<dof;i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
            "DGNET Error in Projection:  %g \n",norm[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&projection);CHKERRQ(ierr);
    ierr = VecDestroy(&error);CHKERRQ(ierr);
    ierr = VecDestroy(&fine);CHKERRQ(ierr);
    ierr = VecDestroy(&coarse);CHKERRQ(ierr);
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
