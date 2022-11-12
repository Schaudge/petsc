#include <petsc/private/netrsimpl.h> 
/*@C
  NetRSCreate - This function creates an empty NetRS. The type of solver 
    can then be set with NetRSSetType().

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. rs   - The NetRS

  Level: beginner

.seealso: NetRSSetType(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode  NetRSCreate(MPI_Comm comm, NetRS *rs)
{
  NetRS  r;

  PetscFunctionBegin;
  PetscValidPointer(rs,1);
  *rs = NULL;
  PetscCall(NetRSInitializePackage());
  PetscCall(PetscHeaderCreate(r, NETRS_CLASSID, "NetRS", "Network Riemann Solver", "NetRS", comm, NetRSDestroy, NetRSView));
  PetscCall(PetscHMapNetRPICreate(&r->netrphmap));
  PetscCall(PetscHSetICreate(&r->vertexdegrees_total)); 
  PetscCall(DMLabelCreate(comm,"NetRSLabel",&r->subgraphs)); 
  PetscCall(DMLabelCreate(comm,"NetRS_DM_InternalTopological",&r->VertexDeg_shared)); 

  /* Add default behavior for NetRS here */
  r->setupcalled = PETSC_FALSE; 
  r->setupvectorspace = PETSC_FALSE; 
  *rs = r;
  PetscFunctionReturn(0);
}
