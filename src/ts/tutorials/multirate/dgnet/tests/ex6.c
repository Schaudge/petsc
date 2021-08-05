static const char help[] = "Test of DMPlexAdd_Disconnected";
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
static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  MPI_Comm          comm;
  DGNetwork         dgnet;
  PetscInt          *order,dof=1,maxdegree = 2,networktype = 0,dx=10,numdm; 
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscViewer       viewer; 
  DM                dmsum,*dmlist; 
  PetscSection      stratumoff; 

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc1(1,&dgnet);CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet,sizeof(*dgnet));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Set default values */
  dgnet->comm           = comm;

  /* Command Line Options
   
   TODO Create DGNetwork commandline options (same as any other petsc object)
  
   */
  dgnet->ndaughters = 2;
  ierr = PetscOptionsBegin(comm,NULL,"DGNetwork options","");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-network","Select from preselected graphs to build the network (0-6)","",networktype,&networktype,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dx","Number of elements on each edge","",dx,&dx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc1(dof,&order);CHKERRQ(ierr);
  MakeOrder(dof,order,maxdegree);
  ierr = Physics_CreateDummy(dgnet,dof,order);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(comm);
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dgnet->network);CHKERRQ(ierr); 
  /* dgnet creation needs to be reworked as this is silly */
  ierr = DGNetworkCreate(dgnet,networktype,dx);CHKERRQ(ierr);
  ierr = DGNetworkSetComponents(dgnet);CHKERRQ(ierr);
  ierr = DGNetworkCleanUp(dgnet);CHKERRQ(ierr);
  ierr = DGNetworkViewEdgeDMs(dgnet,viewer);CHKERRQ(ierr);
  ierr = DGNetworkCreateNetworkDMPlex_3D(dgnet,NULL,0,&dmsum,&stratumoff,&dmlist,&numdm);CHKERRQ(ierr);
  ierr = PetscSectionView(stratumoff,viewer);CHKERRQ(ierr);
  ierr = DMView(dmsum,viewer);CHKERRQ(ierr);

  /* Clean up */
  ierr = PetscFree(dmlist);
  ierr = DGNetworkDestroy(dgnet);CHKERRQ(ierr);
  ierr = DMDestroy(&dgnet->network);CHKERRQ(ierr);
  ierr = DMDestroy(&dmsum);
  ierr = PetscSectionDestroy(&stratumoff);CHKERRQ(ierr);
  ierr = PetscFree(dgnet);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
TEST*/
