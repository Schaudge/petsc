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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(rs,1);
  *rs = NULL;
  ierr = NetRSInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(r, NETRS_CLASSID, "NetRS", "Network Riemann Solver", "NetRS", comm, NetRSDestroy, NetRSView);CHKERRQ(ierr);
  /* Add default behavior for NetRS here */
  r->numfields = -1; 
  r->numedges  = -1;
  r->rs        = PETSC_NULL;
  *rs = r;
  PetscFunctionReturn(0);
}