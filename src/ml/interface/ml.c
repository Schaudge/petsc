#include <petsc/private/mlimpl.h> /*I "petscml.h"  I*/
#include <petsctao.h>

/*@
  MLCreate - Creates a ML context.

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. ml - the new `ML` context


  Level: beginner

.seealso: [](ch_ml), `PetscML`, `MLDestroy()`
@*/
PetscErrorCode MLCreate(MPI_Comm comm, ML *ml)
{
  PetscFunctionBegin;
  PetscAssertPointer(ml, 2);
  *ml = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MLDestroy - Destroys the ML Object
  with `MLCreate()`.

  Collective

  Input Parameter:
. ml - the `ML` context

  Level: beginner

.seealso: [](ch_ml), `ML`, `MLCreate()`
@*/
PetscErrorCode MLDestroy(ML *ml)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ml));
  PetscFunctionReturn(PETSC_SUCCESS);
}
