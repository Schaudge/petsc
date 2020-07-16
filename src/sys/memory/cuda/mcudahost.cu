#include <petscsys.h>             /*I   "petscsys.h"   I*/

/*@C
   PetscMallocSetCUDAHost - Set PetscMalloc to use CUDAHostMalloc
     Switch the current malloc and free routines to the CUDA malloc and free routines

   Not Collective

   Level: developer

   Notes:
     This provides a way to use the CUDA malloc and free routines temporarily. One
     can switch back to the previous choice by calling PetscMallocResetCUDAHost().

.seealso: PetscMallocResetCUDAHost()
@*/
PetscErrorCode PetscMallocSetCUDAHost(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPushMallocType(PETSC_MALLOC_CUDA_HOST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocResetCUDAHost - Reset the changes made by PetscMallocSetCUDAHost

   Not Collective

   Level: developer

.seealso: PetscMallocSetCUDAHost()
@*/
PetscErrorCode PetscMallocResetCUDAHost(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPopMallocType();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
