
#include <petscfn.h>

PetscErrorCode PetscFnCreate_Shell(PetscFn fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)fn,PETSCFNSHELL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
