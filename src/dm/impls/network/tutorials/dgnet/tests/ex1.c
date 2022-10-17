static const char help[] = "Tests The DGNet Tabulation Generation";
/*

  TODO ADD EXAMPLE RUNS HERE 

  Contributed by: Aidan Hamilton <aidan@udel.edu>

*/

#include <petscdm.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"

static PetscErrorCode Physics_CreateDummy(DGNetwork dgnet,PetscInt dof, PetscInt *order)
{
  PetscFunctionBeginUser;
  dgnet->physics.dof            = dof;
  dgnet->physics.order          = order; 
  dgnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
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
  DGNetwork         dgnet;
  PetscInt          *order,dof=1,maxdegree = 2,mode=0; 
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscViewer       viewer; 

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMalloc1(1,&dgnet));
  PetscCall(PetscMemzero(dgnet,sizeof(*dgnet)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Set default values */
  dgnet->comm           = comm;

  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
  ierr = PetscOptionsBegin(comm,NULL,"DGNetwork options","");CHKERRQ(ierr);
    PetscCall(PetscOptionsInt("-deg","Degree for tabulation","",maxdegree,&maxdegree,NULL));
    PetscCall(PetscOptionsInt("-fields","Number of Fields to Generate Tabulations for","",dof,&dof,NULL));
    PetscCall(PetscOptionsInt("-mode","Mode to compute the orders for each field (enter int in [0-2])","",mode,&mode,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscCall(PetscMalloc1(dof,&order));
  switch (mode)
  {
  case 0:
    OrderCopyTest(dof,order,maxdegree);
    break;
  case 1: 
    OrderTabTest(dof,order,maxdegree);
    break; 
  case 2:
    OrderInterestingTest(dof,order,maxdegree);
    break; 
  default:
    OrderCopyTest(dof,order,maxdegree);
    break;
  }
  PetscCall(Physics_CreateDummy(dgnet,dof,order));
  PetscCall(DGNetworkBuildTabulation(dgnet));
  viewer = PETSC_VIEWER_STDOUT_(comm);
  PetscCall(ViewDiscretizationObjects(dgnet,viewer));

  /* Clean up */
  PetscCall(DGNetworkDestroyTabulation(dgnet)); /* Destroy all data within the network and within dgnet */
  PetscCall(PetscFree(dgnet));
  PetscCall(PetscFree(order));
  PetscCall(PetscFinalize());
  return 0;
}
