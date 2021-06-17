#include <petsc/private/mlregressorimpl.h>

PetscBool         MLRegressorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList MLRegressorList              = NULL;

PetscClassId MLREGRESSOR_CLASSID;

/* Logging support */
PetscLogEvent MLRegressor_SetUp, MLRegressor_Fit, MLRegressor_Predict;

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

/*@
   MLRegressorSetUp - Sets up the internal data structures for the later use
   of a regressor.

   Collective on MLRegressor

   Input Parameters:
.  mlregressor - the MLRegressor context

   Notes:
   For basic use of the MLRegressor solvers the user need not explicitly call
   MLRegressorSetUp(), since these actions will automatically occur during
   the call to MLRegressorFit().  However, if one wishes to control this
   phase separately, MLRegressorSetUp() should be called after MLRegressorCreate()
   and optional routines of the form MLRegressorSetXXX(), but before MLRegressorFit().

   Level: advanced

.seealso: MLRegressorCreate(), MLRegressorFit(), MLRegressorDestroy()
@*/
PetscErrorCode MLRegressorSetUp(MLRegressor mlregressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  if (mlregressor->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(MLRegressor_SetUp,mlregressor,0,0,0);CHKERRQ(ierr);

  if (!((PetscObject)mlregressor)->type_name) {
    ierr = MLRegressorSetType(mlregressor,MLREGRESSORLINEAR);CHKERRQ(ierr);
  }

  if (mlregressor->ops->setup) {
    ierr = (*mlregressor->ops->setup)(mlregressor);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(MLRegressor_SetUp,mlregressor,0,0,0);CHKERRQ(ierr);
  mlregressor->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOTE: I've decided to make this take X and y, like the Scikit-learn Fit routines do.
 * Am I overlooking some reason that X should be set in a seperate function call, a la KSPSetOperators()?. */
PetscErrorCode MLRegressorFit(MLRegressor mlregressor, Mat X, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,MAT_CLASSID,2);
  if (y) PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  if (X) {
    ierr                  = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    ierr                  = MatDestroy(&mlregressor->training);CHKERRQ(ierr);
    mlregressor->training = X;
  }
  if (y) {
    ierr                  = PetscObjectReference((PetscObject)y);CHKERRQ(ierr);
    ierr                  = VecDestroy(&mlregressor->target);CHKERRQ(ierr);
    mlregressor->target   = y;
  }
  ierr = (*mlregressor->ops->fit)(mlregressor);CHKERRQ(ierr);
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
