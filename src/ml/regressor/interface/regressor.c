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
  if (mlregressor->ops->reset) {
    ierr = (*mlregressor->ops->reset)(mlregressor);CHKERRQ(ierr);
  }
  // TODO: Finish putting all of the Reset, Destroy, and free calls needed here!
  ierr = MatDestroy(&mlregressor->training);CHKERRQ(ierr);
  ierr = VecDestroy(&mlregressor->target);CHKERRQ(ierr);
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
  if ((*mlregressor)->ops->destroy) {ierr = (*(*mlregressor)->ops->destroy)(*mlregressor);CHKERRQ(ierr);}

  ierr = PetscHeaderDestroy(mlregressor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorSetType(MLRegressor mlregressor, MLRegressorType type)
{
  PetscErrorCode ierr,(*r)(MLRegressor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)mlregressor,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(MLRegressorList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)mlregressor),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested MLRegressor type %s",type);

  /* Destroy the previous private MLRegressor context */
  if (mlregressor->ops->destroy) {
    ierr               = (*(mlregressor)->ops->destroy)(mlregressor);CHKERRQ(ierr);
    mlregressor->ops->destroy = NULL;
  }

  /* Reinitialize function pointers in MLRegressorOps structure */
  mlregressor->ops->setup          = NULL;
  mlregressor->ops->setfromoptions = NULL;
  mlregressor->ops->settraining    = NULL;
  mlregressor->ops->fit            = NULL;
  mlregressor->ops->predict        = NULL;
  mlregressor->ops->destroy        = NULL;
  mlregressor->ops->reset          = NULL;
  mlregressor->ops->view           = NULL;

  /* Call the MLRegressorCreate_XXX routine for this particular regressor */
  mlregressor->setupcalled = PETSC_FALSE;
  ierr = (*r)(mlregressor);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)mlregressor,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorView(MLRegressor mlregressor,PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: Complete this when I have a good idea of what bits of the MLRegressor should be shown!
  PetscFunctionReturn(0);
}
