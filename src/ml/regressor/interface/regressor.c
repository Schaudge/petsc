#include <petsc/private/mlregressorimpl.h>

PetscBool         MLRegressorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList MLRegressorList              = NULL;

PetscClassId MLREGRESSOR_CLASSID;

PetscErrorCode MLRegressorRegister(const char sname[],PetscErrorCode (*function)(MLRegressor))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MLRegressorInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&MLRegressorList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorCreate(MPI_Comm comm,MLRegressor *newmlregressor)
{
  MLRegressor    mlregressor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newmlregressor,2);
  *newmlregressor = NULL;
  ierr = MLRegressorInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(mlregressor,MLREGRESSOR_CLASSID,"MLRegressor","Regressor","MLRegressor",comm,MLRegressorDestroy,MLRegressorView);CHKERRQ(ierr);

  // TODO: Finish setting the various fields of the MLRegressor private data structure to defaults, etc.
  mlregressor->setupcalled = PETSC_FALSE;
  mlregressor->data = NULL;
  mlregressor->training = NULL;
  mlregressor->target = NULL;
  
  *newmlregressor = mlregressor;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorReset(MLRegressor mlregressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  // TODO: Finish putting all of the Reset, Destroy, and free calls needed here!
  ierr = MatDestroy(&mlregressor->training);CHKERRQ(ierr);
  mlregressor->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   MLRegressorDestroy - Destroys the regressor context that was created
   with MLRegressorCreate().

   Collective on MLRegressor

   Input Parameter:
.  mlregressor - the MLRegressor context

   Level: beginner

.seealso: MLRegressorCreate(), MLRegressorSetUp(), MLRegressorReset(), MLRegressor
@*/
PetscErrorCode MLRegressorDestroy(MLRegressor *mlregressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mlregressor) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*mlregressor),MLREGRESSOR_CLASSID,1);
  if (--((PetscObject)(*mlregressor))->refct > 0) {*mlregressor = NULL; PetscFunctionReturn(0);}

  ierr = MLRegressorReset((*mlregressor));CHKERRQ(ierr);

  ierr = PetscHeaderDestroy(mlregressor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorView(MLRegressor mlregressor,PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: Complete this when I have a good idea of what bits of the MLRegressor should be shown!
  PetscFunctionReturn(0);
}
