#include <petsc/private/regressorimpl.h>

PetscBool         PetscRegressorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscRegressorList              = NULL;

PetscClassId PETSCREGRESSOR_CLASSID;

/* Logging support */
PetscLogEvent PetscRegressor_SetUp, PetscRegressor_Fit, PetscRegressor_Predict;

/*@C
   PetscRegressorRegister - Adds a method to the PetscRegressor package.

   Not collective

   Input Parameters:
+  sname - name of a new user-defined regressor
-  function - routine to create method context

   Notes:
   PetscRegressorRegister() may be called multiple times to add several user-defined regressors.

   Sample usage:
.vb
   PetscRegressorRegister("my_regressor",MyRegressorCreate);
.ve

   Then, your regressor can be chosen with the procedural interface via
$     PetscRegressorSetType(regressor,"my_regressor")
   or at runtime via the option
$     -regressor_type my_regressor

   Level: advanced

.seealso: PetscRegressorRegisterAll()
@*/
PetscErrorCode PetscRegressorRegister(const char sname[],PetscErrorCode (*function)(PetscRegressor))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRegressorInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscRegressorList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorCreate - Creates a regressor object.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newregressor - the new regressor object

   Level: beginner

.seealso: PetscRegressorFit(), PetscRegressorPredict(), PetscRegressor
@*/
PetscErrorCode PetscRegressorCreate(MPI_Comm comm,PetscRegressor *newregressor)
{
  PetscRegressor    regressor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newregressor,2);
  *newregressor = NULL;
  ierr = PetscRegressorInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(regressor,PETSCREGRESSOR_CLASSID,"PetscRegressor","Regressor","PetscRegressor",comm,PetscRegressorDestroy,PetscRegressorView);CHKERRQ(ierr);

  // TODO: Finish setting the various fields of the PetscRegressor private data structure to defaults, etc.
  regressor->setupcalled = PETSC_FALSE;
  regressor->data = NULL;
  regressor->training = NULL;
  regressor->target = NULL;
  
  *newregressor = regressor;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorSetFromOptions - Sets PetscRegressor options from the options database.

   Collective on PetscRegressor

   Input Parameter:
.  regressor - the PetscRegressor context

   Options Database Keys:
.  -regressor_type <type> - the particular type of regressor to be used; see PetscRegressorType for complete list

   This routine must be called before PetscRegressorSetUp() (or PetscRegressorFit(), which calls
   the former) if the user is to be allowed to set the regressor type.

   Level: beginner

.seealso: PetscRegressor, PetscRegressorCreate()
@*/
PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor regressor)
{
  PetscErrorCode ierr;
  PetscBool flg;
  PetscRegressorType default_type = PETSCREGRESSORLINEAR;
  char type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)regressor);CHKERRQ(ierr);
  if (((PetscObject)regressor)->type_name) {
    default_type = ((PetscObject)regressor)->type_name;
  }
  /* Check for type from options */
  ierr = PetscOptionsFList("-regressor_type","PetscRegressor type","PetscRegressorSetType",PetscRegressorList,default_type,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRegressorSetType(regressor,type);CHKERRQ(ierr);
  } else if (!((PetscObject)regressor)->type_name) {
    ierr = PetscRegressorSetType(regressor,default_type);CHKERRQ(ierr);
  }
  /* TODO: Add code to handle other options that apply to all PetscRegressor types. */
  if (regressor->ops->setfromoptions) {
    ierr = (*regressor->ops->setfromoptions)(PetscOptionsObject,regressor);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorSetUp - Sets up the internal data structures for the later use of a regressor.

   Collective on PetscRegressor

   Input Parameters:
.  regressor - the PetscRegressor context

   Notes:
   For basic use of the PetscRegressor solvers the user need not to explicitly call
   PetscRegressorSetUp(), since these actions will automatically occur during
   the call to PetscRegressorFit().  However, if one wishes to control this
   phase separately, PetscRegressorSetUp() should be called after PetscRegressorCreate(),
   PetscRegressorSetUp(), and optional routines of the form PetscRegressorSetXXX(),
   but before PetscRegressorFit().

   Level: advanced

.seealso: PetscRegressorCreate(), PetscRegressorFit(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorSetUp(PetscRegressor regressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  if (regressor->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PetscRegressor_SetUp,regressor,0,0,0);CHKERRQ(ierr);

  if (!((PetscObject)regressor)->type_name) {
    ierr = PetscRegressorSetType(regressor,PETSCREGRESSORLINEAR);CHKERRQ(ierr);
  }

  if (regressor->ops->setup) {
    ierr = (*regressor->ops->setup)(regressor);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(PetscRegressor_SetUp,regressor,0,0,0);CHKERRQ(ierr);
  regressor->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOTE: I've decided to make this take X and y, like the Scikit-learn Fit routines do.
 * Am I overlooking some reason that X should be set in a separate function call, a la KSPSetOperators()?. */
/*@
   PetscRegressorFit - Fit, or train, a regressor from a training dataset

   Collective on PetscRegressor

   Input Parameters:
+  regressor - the regressor context
.  X - matrix of training data (of dimension [number of samples] x [number of features])
-  y - vector of target values from the training dataset

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorDestroy(), PetscRegressorPredict()
@*/
PetscErrorCode PetscRegressorFit(PetscRegressor regressor, Mat X, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,MAT_CLASSID,2);
  if (y) PetscValidHeaderSpecific(y,VEC_CLASSID,3);

  if (X) {
    ierr                  = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    ierr                  = MatDestroy(&regressor->training);CHKERRQ(ierr);
    regressor->training = X;
  }
  if (y) {
    ierr                  = PetscObjectReference((PetscObject)y);CHKERRQ(ierr);
    ierr                  = VecDestroy(&regressor->target);CHKERRQ(ierr);
    regressor->target   = y;
  }
  ierr = PetscRegressorSetUp(regressor);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(PetscRegressor_Fit,regressor,X,y,0);CHKERRQ(ierr);
  ierr = (*regressor->ops->fit)(regressor);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PetscRegressor_Fit,regressor,X,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorPredict - Compute predictions (that is, perform inference) using a fitted regression model.

   Collective on PetscRegressorPredict

   Input Parameters:
+  regressor - the regressor context (for which PetscRegressorFit() must have been called)
.  X - data matrix of unlabeled observations
-  y - vector of predicted labels

   Level: beginner

.seealso: PetscRegressorFit(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorPredict(PetscRegressor regressor, Mat X, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,MAT_CLASSID,2);
  if (y) PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  // TODO: Add code to check that the Fit step has been completed!
  ierr = PetscLogEventBegin(PetscRegressor_Predict,regressor,X,y,0);CHKERRQ(ierr);
  ierr = (*regressor->ops->predict)(regressor,X,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PetscRegressor_Predict,regressor,X,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorReset - Resets a PetscRegressor context to the setupcalled = 0 state and remoecs any allocated Vecs and Mats

   Collective on PetscRegressor

   Input Parameter:
.  regressor - context obtained from PetscRegressorCreate()

   Level: intermediate

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorFit(), PetscRegressorPredict(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorReset(PetscRegressor regressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  if (regressor->ops->reset) {
    ierr = (*regressor->ops->reset)(regressor);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&regressor->training);CHKERRQ(ierr);
  ierr = VecDestroy(&regressor->target);CHKERRQ(ierr);
  regressor->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorDestroy - Destroys the regressor context that was created with PetscRegressorCreate().

   Collective on PetscRegressor

   Input Parameter:
.  regressor - the PetscRegressor context

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorReset(), PetscRegressor
@*/
PetscErrorCode PetscRegressorDestroy(PetscRegressor *regressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*regressor) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*regressor),PETSCREGRESSOR_CLASSID,1);
  if (--((PetscObject)(*regressor))->refct > 0) {*regressor = NULL; PetscFunctionReturn(0);}

  ierr = PetscRegressorReset((*regressor));CHKERRQ(ierr);
  if ((*regressor)->ops->destroy) {ierr = (*(*regressor)->ops->destroy)(*regressor);CHKERRQ(ierr);}

  ierr = PetscHeaderDestroy(regressor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorSetType - Sets the type for the regressor.

   Collective on PetscRegressor

   Input Parameters:
+  regressor - the PetscRegressor context
-  type - a known regression method

   Options Database Key:
.  -regressor_type <type> - Sets the type of regressor; use -help for a list of available types

   Notes:
   See "include/petscregressor.h" for available methods (for instance)
.    PETSCREGRESSORLINEAR - Linear regression models (ordinary least squares as well as regularized variants)

   Normally, it is best to use the PetscRegressorSetFromOptions() command and then
   set the PetscRegressor type from the options database rather than by using
   this routine, as this provides maximum flexibility.
   The PetscRegressorSetType() routine is provided for those situations where it
   is necessary to set the nonlinear solver independently of the command
   line or options database.

   Level: intermediate

.seealso: PetscRegressorType
@*/
PetscErrorCode PetscRegressorSetType(PetscRegressor regressor, PetscRegressorType type)
{
  PetscErrorCode ierr,(*r)(PetscRegressor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)regressor,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PetscRegressorList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)regressor),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscRegressor type %s",type);

  /* Destroy the previous private PetscRegressor context */
  if (regressor->ops->destroy) {
    ierr               = (*(regressor)->ops->destroy)(regressor);CHKERRQ(ierr);
    regressor->ops->destroy = NULL;
  }

  /* Reinitialize function pointers in PetscRegressorOps structure */
  regressor->ops->setup          = NULL;
  regressor->ops->setfromoptions = NULL;
  regressor->ops->settraining    = NULL;
  regressor->ops->fit            = NULL;
  regressor->ops->predict        = NULL;
  regressor->ops->destroy        = NULL;
  regressor->ops->reset          = NULL;
  regressor->ops->view           = NULL;

  /* Call the PetscRegressorCreate_XXX routine for this particular regressor */
  regressor->setupcalled = PETSC_FALSE;
  ierr = (*r)(regressor);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)regressor,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorView(PetscRegressor regressor,PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: Complete this when I have a good idea of what bits of the PetscRegressor should be shown!
  PetscFunctionReturn(0);
}
