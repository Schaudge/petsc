#include <petsc/private/localnetrpimpl.h> /*I "petscnetrp.h" I*/
/*@C
  NetRPCreate - This function creates an empty NetRP. The type of solver 
    can then be set with NetRSSetType().

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. rp   - The NetRP

  Level: beginner

.seealso: NetRSSetType(), NetRSSetUp(), NetRSDestroy()
@*/
PetscErrorCode NetRPCreate(MPI_Comm comm, NetRP *rp)
{
  NetRP r;

  PetscFunctionBegin;
  PetscValidPointer(rp, 1);
  *rp = NULL;
  PetscCall(NetRPInitializePackage());
  PetscCall(PetscHeaderCreate(r, NETRP_CLASSID, "NetRP", "Local Network Riemann Problem", "NetRP", comm, NetRPDestroy, NetRPView));
  PetscCall(PetscHMapICreate(&r->hmap));
  PetscCall(PetscHMapIJCreate(&r->dirhmap));

  /* Add default behavior for NetRP here */
  r->flux      = PETSC_NULLPTR;
  r->cachetype = UndirectedVDeg; /* Assume UndirectedVDeg unless told otherwise */
  /* Assume worst case situation */
  r->solvetype         = Other;
  r->physicsgenerality = Specific;
  r->numfields         = -1;
  *rp                  = r;
  PetscFunctionReturn(PETSC_SUCCESS);
}
