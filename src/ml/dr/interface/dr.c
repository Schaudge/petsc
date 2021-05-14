#include <petsc/private/mldrimpl.h>

PetscBool         MLDRRegisterAllCalled = PETSC_FALSE;
PetscFunctionList MLDRList              = NULL;

PetscErrorCode MLDRRegister(const char sname[],PetscErrorCode (*function)(MLDR))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MLDRInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MLDRList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MLDRCreate(MPI_Comm comm,MLDR *newmldr)
{
  MLDR           mldr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newmldr,2);
  *newmldr = NULL;
  ierr = MLDRInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(mldr,MLDR_CLASSID,"MLDR","Dimension Reduction","MLDR",comm,MLDRDestroy,MLDRView);CHKERRQ(ierr);

  // TODO: Finish setting the various fields of the MLDR private data structure to defaults, etc.
  mldr->setupcalled = PETSC_FALSE;
  mldr->data = NULL;
  mldr->svd = NULL;
  mldr->training = NULL;
  
  *newmldr = mldr;
  PetscFunctionReturn(0);
}

PetscErrorCode MLDRReset(MLDR mldr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mldr,MLDR_CLASSID,1);
  // TODO: Finish putting all of the Reset, Destroy, and free calls needed here!
  if (mldr->svd) {
    ierr = SVDReset(mldr->svd);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&mldr->training);CHKERRQ(ierr);
  mldr->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
