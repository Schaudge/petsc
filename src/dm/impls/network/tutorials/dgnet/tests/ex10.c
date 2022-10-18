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

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMalloc2(1,&dgnet_fine,1,&dgnet_coarse));
  PetscCall(PetscMemzero(dgnet_fine,sizeof(*dgnet_fine)));
  PetscCall(PetscMemzero(dgnet_fine,sizeof(*dgnet_coarse)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));


PetscCall(PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow));

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
    PetscCall(PetscOptionsInt("-deg_fine","Degree for Fine tabulation","",maxdegree_fine,&maxdegree_fine,NULL));
    PetscCall(PetscOptionsInt("-deg_coarse","Degree for Coarse tabulation","",maxdegree_coarse,&maxdegree_coarse,NULL));
    PetscCall(PetscOptionsInt("-fields","Number of Fields to Generate Tabulations for","",dof,&dof,NULL));
    PetscCall(PetscOptionsInt("-mode","Mode to compute the orders for each field (enter int in [0-2])","",mode,&mode,NULL));
    PetscCall(PetscOptionsInt("-Mx_fine","Smallest number of cells for an edge","",dgnet_fine->Mx,&dgnet_fine->Mx,NULL));
    PetscCall(PetscOptionsInt("-Mx_coarse","Smallest number of cells for an edge","",dgnet_coarse->Mx,&dgnet_coarse->Mx,NULL));
    PetscCall(PetscOptionsBool("-view","view the fields on the network","",view,&view,NULL));
    PetscCall(PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",dgnet_fine->initial,&dgnet_fine->initial,NULL));
    PetscCall(PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",dgnet_fine->networktype,&dgnet_fine->networktype,NULL));
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
  PetscCall(PetscMalloc2(dof,&order_fine,dof,&order_coarse));

    /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    PetscCall(PetscFunctionListFind(physics,physname,&r));
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(dgnet_fine));
    PetscCall((*r)(dgnet_coarse));
  }
    PetscCall(PetscMalloc2(dgnet_fine->physics.dof,&dgnet_fine->physics.order,dgnet_coarse->physics.dof,&dgnet_coarse->physics.order)); /* should be constructed by physics */

  PetscCall(MakeOrder(dgnet_fine->physics.dof,dgnet_fine->physics.order,maxdegree_fine));
  PetscCall(MakeOrder(dgnet_coarse->physics.dof,dgnet_coarse->physics.order,maxdegree_coarse));

    viewer = PETSC_VIEWER_STDOUT_(comm);

    PetscCall(DGNetworkCreate(dgnet_fine,dgnet_fine->networktype,dgnet_fine->Mx));
    PetscCall(DGNetworkCreate(dgnet_coarse,dgnet_coarse->networktype,dgnet_coarse->Mx));
    /* Create DMNetwork */
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dgnet_fine->network));
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dgnet_coarse->network));
    
    if (size == 1 && view) {
        PetscCall(DGNetworkMonitorCreate(dgnet_fine,&monitor_fine));
        PetscCall(DGNetworkMonitorCreate(dgnet_fine,&monitor_fine2));
        PetscCall(DGNetworkMonitorCreate(dgnet_fine,&monitor_fine3));
        PetscCall(DGNetworkMonitorCreate(dgnet_coarse,&monitor_coarse));
    }
    /* Set Network Data into the DMNetwork (on proc[0]) */
    PetscCall(DGNetworkSetComponents(dgnet_fine));
    PetscCall(DGNetworkSetComponents(dgnet_coarse));

    /* Delete unneeded data in dgnet */
    PetscCall(DGNetworkCleanUp(dgnet_fine));
    PetscCall(DGNetworkCleanUp(dgnet_coarse));

    PetscCall(DGNetworkBuildTabulation(dgnet_fine));
    PetscCall(DGNetworkBuildTabulation(dgnet_coarse));

    PetscCall(DMNetworkDistribute(&dgnet_fine->network,0));
    PetscCall(DMNetworkDistribute(&dgnet_coarse->network,0));

    /* Create Vectors */
    PetscCall(DGNetworkCreateVectors(dgnet_fine));
    PetscCall(DGNetworkCreateVectors(dgnet_coarse));

    /* Set up component dynamic data structures */
    PetscCall(DGNetworkBuildDynamic(dgnet_fine));
    PetscCall(DGNetworkBuildDynamic(dgnet_coarse));

    if (view) {
        PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine));
        PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine2));
        PetscCall(DGNetworkAddMonitortoEdges(dgnet_fine,monitor_fine3));
        PetscCall(DGNetworkAddMonitortoEdges(dgnet_coarse,monitor_coarse));
    }

    /* Load Fine Vector and Coarse Vector */
    PetscCall(VecCreate(comm,&coarse));
    PetscCall(VecCreate(comm,&fine));
    PetscCall(PetscViewerBinaryOpen(comm,"ex8output_true",FILE_MODE_READ,&fineload));
    PetscCall(PetscSNPrintf(coarsefile,256,"ex8output_P%i_%i",maxdegree_coarse,dgnet_coarse->Mx));
    PetscCall(PetscViewerBinaryOpen(comm,coarsefile,FILE_MODE_READ,&coarseload));
    
    PetscCall(VecLoad(fine,fineload));
    PetscCall(VecLoad(coarse,coarseload));
    PetscCall(PetscViewerDestroy(&fineload));
    PetscCall(PetscViewerDestroy(&coarseload));

    PetscCall(VecDuplicate(dgnet_fine->X,&projection));
    PetscCall(VecDuplicate(dgnet_fine->X,&error));

    if (size == 1 && view) {
        PetscCall(DGNetworkMonitorView(monitor_fine,fine));
        PetscCall(DGNetworkMonitorView(monitor_coarse,coarse));
    }
    /* Project from the coarse grid to the fine grid */
    PetscCall(DGNetworkProject_Coarse_To_Fine(dgnet_fine,dgnet_coarse,coarse,projection));
    /* Compute Error in the projection */
    PetscCall(VecWAXPY(error,-1,fine,projection));
    if (size == 1 && view) {
        PetscCall(DGNetworkMonitorView(monitor_fine2,projection));
        PetscCall(DGNetworkMonitorView(monitor_fine3,error));
    }
    /*compute norm of the error */
    PetscCall(PetscMalloc1(dof,&norm));
    PetscCall(DGNetworkNormL2(dgnet_fine,error,norm));
    for (i=0;i<dof;i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
            "DGNET Error in Projection:  %g \n",norm[i]);CHKERRQ(ierr);
    }
    PetscCall(VecDestroy(&projection));
    PetscCall(VecDestroy(&error));
    PetscCall(VecDestroy(&fine));
    PetscCall(VecDestroy(&coarse));
        /* Clean up */
    if (view && size==1) {
        PetscCall(DGNetworkMonitorDestroy(&monitor_fine));
        PetscCall(DGNetworkMonitorDestroy(&monitor_coarse));
        PetscCall(DGNetworkMonitorDestroy(&monitor_fine2));
        PetscCall(DGNetworkMonitorDestroy(&monitor_fine3));
    }

    PetscCall(DGNetworkDestroy(dgnet_fine)); /* Destroy all data within the network and within dgnet */
    PetscCall(DGNetworkDestroy(dgnet_coarse)); /* Destroy all data within the network and within dgnet */
    PetscCall(DMDestroy(&dgnet_fine->network));
    PetscCall(DMDestroy(&dgnet_coarse->network));
  PetscCall(PetscFree2(dgnet_coarse,dgnet_fine));
  PetscCall(PetscFree2(order_coarse,order_fine));
  PetscCall(PetscFinalize());
  return 0;
}
