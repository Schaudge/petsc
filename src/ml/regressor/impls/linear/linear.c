#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscregressor.h" I*/

PetscErrorCode EvaluateResidual(Tao tao, Vec x, Vec f, void *ptr)
{
  PetscErrorCode ierr;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)ptr;

  PetscFunctionBegin;
  /* Evaluate f = A * x - b */
  ierr = MatMult(linear->X,x,f);CHKERRQ(ierr);
  ierr = VecAXPY(f,-1.0,linear->rhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ptr)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)ptr;

  PetscFunctionBegin;
  J = linear->X;
  Jpre = linear->X;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorSetUp_Linear(PetscRegressor regressor)
{
  //MPI_Comm comm;
  PetscErrorCode ierr;
  PetscInt M,N;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;
  KSP ksp;
  Tao tao;

  PetscFunctionBegin;

  ierr = MatGetSize(regressor->training,&M,&N);CHKERRQ(ierr);

  if (linear->fit_intercept) {
    /* If we are fitting the intercept, we need to make A a composite matrix using MATCENTERING to preserve sparsity.
     * Though there might be some cases we don't want to do this for, depending on what kind of matrix is passed in. (Probably bad idea for dense?)
     * We will also need to ensure that the right-hand side passed to the KSP is also mean-centered, since we
     * intend to compute the intercept separately from regression coefficients (that is, we will not be adding a
     * column of all 1s to our design matrix). */
    ierr = MatCreateCentering(PetscObjectComm((PetscObject)regressor),PETSC_DECIDE,M,&linear->C);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)regressor),&linear->X);CHKERRQ(ierr);
    ierr = MatSetSizes(linear->X,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(linear->X,MATCOMPOSITE);CHKERRQ(ierr);
    ierr = MatCompositeSetType(linear->X,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
    ierr = MatCompositeAddMat(linear->X,regressor->training);CHKERRQ(ierr);
    ierr = MatCompositeAddMat(linear->X,linear->C);CHKERRQ(ierr);
    ierr = VecDuplicate(regressor->target,&linear->rhs);CHKERRQ(ierr);
    ierr = MatMult(linear->C,regressor->target,linear->rhs);CHKERRQ(ierr);
  } else {
    /* When not fitting intercept, we assume that the input data are already centered.
     * TODO: Perhaps revisit exactly what options should exist around this. */
    linear->X = regressor->training;
    linear->rhs = regressor->target;
  }

  if (linear->coefficients) {ierr = VecDestroy(&linear->coefficients);CHKERRQ(ierr);}

  if (linear->use_ksp) {
    if (!linear->ksp) {
      ierr = PetscRegressorLinearGetKSP(regressor,&linear->ksp);CHKERRQ(ierr);
      // TODO: Figure out if I need to set operators for the KSP here or set the operator X.
      // I think maybe I can just do this stuff in the Fit() routine.
    }
    ksp = linear->ksp;

    ierr = MatCreateVecs(linear->X,&linear->coefficients,NULL);CHKERRQ(ierr);
    /* Set up the KSP to solve the least squares problem (without solving for intercept, as this is done separately) using KSPLSQR.
     * TODO: Add options to use other methods. */
    ierr = MatCreateNormal(linear->X,&linear->XtX);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,linear->X,linear->XtX);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);  // TODO: Does this have the right option prefixes set?
  } else { /* Use TAO */
    if (!linear->tao) {
      ierr = PetscRegressorLinearGetTao(regressor,&linear->tao);CHKERRQ(ierr);
    }
    tao = linear->tao;

    ierr = MatCreateVecs(linear->X,&linear->coefficients,&linear->residual);CHKERRQ(ierr);
    /* Set up the TAO object to solve the (regularized) least squares problem (without solving for intercept, which is done separately) using TAOBRGN. */
    ierr = TaoSetType(tao, TAOBRGN);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,linear->coefficients);CHKERRQ(ierr);
    // Note: Above will need to come the below when I rebase over a more recent PETSc:
    // ierr = TaoSetSolution(tao,linear->coefficients);CHKERRQ(ierr);
    ierr = TaoSetResidualRoutine(tao,linear->residual,EvaluateResidual,linear);CHKERRQ(ierr);
    ierr = TaoSetJacobianResidualRoutine(tao,linear->X,linear->X,EvaluateJacobian,linear);CHKERRQ(ierr);
    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorReset_Linear(PetscRegressor regressor)
{
  PetscErrorCode ierr;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  ierr = MatDestroy(&linear->X);CHKERRQ(ierr);
  ierr = MatDestroy(&linear->XtX);CHKERRQ(ierr);
  ierr = MatDestroy(&linear->C);CHKERRQ(ierr);
  ierr = KSPDestroy(&linear->ksp);CHKERRQ(ierr);
  ierr = TaoDestroy(&linear->tao);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->coefficients);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->residual);CHKERRQ(ierr);

  /* Reset options/parameters to the setupcalled = 0 state. */
  /* TODO: Add the reset code once the linear regressor is fleshed out enough to need resetting! */
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorDestroy_Linear(PetscRegressor regressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRegressorReset_Linear(regressor);CHKERRQ(ierr);
  ierr = PetscFree(regressor->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor regressor, PetscBool flg)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  linear->fit_intercept = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor regressor, PetscBool flg)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  linear->use_ksp = flg;
  PetscFunctionReturn(0);
}
PetscErrorCode PetscRegressorSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject, PetscRegressor regressor)
{
  PetscBool          flg,set;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscRegressor options for linear regressors");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-regressor_linear_fit_intercept","Calculate intercept for linear model","PetscRegressorLinearSetFitIntercept",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = PetscRegressorLinearSetFitIntercept(regressor,flg);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-regressor_linear_use_ksp","Use KSP instead of TAO for linear model fitting problem","PetscRegressorLinearSetFitIntercept",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = PetscRegressorLinearSetUseKSP(regressor,flg);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorView_Linear(PetscRegressor regressor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor regressor,KSP *ksp)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  PetscValidPointer(ksp,2);
  /* Analogous to how SNESGetKSP() operates, this routine should create the KSP if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!linear->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)regressor),&linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)linear->ksp,(PetscObject)regressor,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)regressor,(PetscObject)linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)linear->ksp,((PetscObject)regressor)->options);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorLinearGetTao(PetscRegressor regressor,Tao *tao)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  PetscValidPointer(tao,2);
  /* Analogous to how SNESGetKSP() operates, this routine should create the TAO if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!linear->tao) {
    ierr = TaoCreate(PetscObjectComm((PetscObject)regressor),&linear->tao);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)linear->tao,(PetscObject)regressor,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)regressor,(PetscObject)linear->tao);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)linear->tao,((PetscObject)regressor)->options);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor regressor, Vec *coefficients)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  PetscValidPointer(coefficients,2);
  *coefficients = linear->coefficients;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor regressor, PetscScalar *intercept)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor,PETSCREGRESSOR_CLASSID,1);
  PetscValidPointer(intercept,2);
  *intercept = linear->intercept;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorFit_Linear(PetscRegressor regressor)
{
  PetscErrorCode ierr;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;
  KSP ksp;
  PetscScalar target_mean,*column_means_global,*column_means_local,column_means_dot_coefficients;
  Vec column_means;
  PetscInt m,N,istart,i;

  PetscFunctionBegin;
  if (linear->use_ksp && !linear->ksp) {ierr = PetscRegressorLinearGetKSP(regressor,&linear->ksp);CHKERRQ(ierr);}
  ksp = linear->ksp;

  /* Solve the least-squares problem (previously set up in PetscRegressorSetUp_Linear()) without finding the intercept. */
  if (linear->use_ksp) {
    ierr = KSPSolve(ksp,linear->rhs,linear->coefficients);CHKERRQ(ierr);
  } else {
    ierr = TaoSolve(linear->tao);CHKERRQ(ierr);
  }

  /* Calculate the intercept. */
  if (linear->fit_intercept) {
    ierr = MatGetSize(regressor->training,NULL,&N);CHKERRQ(ierr);
    ierr = PetscMalloc1(N,&column_means_global);CHKERRQ(ierr);
    ierr = VecMean(regressor->target,&target_mean);CHKERRQ(ierr);
    /* We need the means of all columns of regressor->training, placed into a Vec compatible with linear->coefficients.
     * Note the potential scalability issue: MatGetColumnMeans() computes means of ALL colummns. */
    ierr = MatGetColumnMeans(regressor->training,column_means_global);CHKERRQ(ierr);
    /* TODO: Calculation of the Vec and matrix column means should probably go into the SetUp phase, and also be placed
     *       into a routine that is callable from outside of PetscRegressorFit_Linear(), because we'll want to do the same
     *       thing for other models, such as ridge and LASSO regression, and should avoid code duplication.
     *       What we are calling 'target_mean' and 'column_means' should be stashed in the base linear regressor struct,
     *       and perhaps renamed to make it clear they are offsets that should be applied (though the current naming
     *       makes sense since it makes it clear where these come from.) */
    ierr = VecDuplicate(linear->coefficients,&column_means);CHKERRQ(ierr);
    ierr = VecGetLocalSize(column_means,&m);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(column_means,&istart,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(column_means,&column_means_local);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      column_means_local[i] = column_means_global[istart+i];
    }
    ierr = VecRestoreArrayWrite(column_means,&column_means_local);CHKERRQ(ierr);
    ierr = VecDot(column_means,linear->coefficients,&column_means_dot_coefficients);CHKERRQ(ierr);
    ierr = VecDestroy(&column_means);CHKERRQ(ierr);
    linear->intercept = target_mean - column_means_dot_coefficients;
  } else {
    linear->intercept = 0.0;
  }

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscRegressorPredict_Linear(PetscRegressor regressor, Mat X, Vec y)
{
  PetscErrorCode ierr;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR*)regressor->data;

  PetscFunctionBegin;
  ierr = MatMult(X,linear->coefficients,y);CHKERRQ(ierr);
  ierr = VecShift(y,linear->intercept);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PETSCREGRESSORLINEAR - Linear regression model (ordinary least squares or regularized variants)

   Options Database:
+  -regressor_linear_fit_intercept - Calculate the intercept for the linear model
-  -regressor_linear_use_ksp - Use KSP instead of TAO for linear model fitting (non-regularized variants only)

   Notes:
   This is the default regressor in PetscRegressor

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressor, PetscRegressorSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor regressor)
{
  PETSCREGRESSOR_LINEAR *linear;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(regressor,&linear);CHKERRQ(ierr);
  regressor->data = (void*)linear;

  regressor->ops->setup          = PetscRegressorSetUp_Linear;
  regressor->ops->reset          = PetscRegressorReset_Linear;
  regressor->ops->destroy        = PetscRegressorDestroy_Linear;
  regressor->ops->setfromoptions = PetscRegressorSetFromOptions_Linear;
  regressor->ops->view           = PetscRegressorView_Linear;
  regressor->ops->fit            = PetscRegressorFit_Linear;
  regressor->ops->predict        = PetscRegressorPredict_Linear;

  linear->intercept = 0.0;
  linear->fit_intercept = PETSC_TRUE;  /* Defaulting to calculating the intercept is probably sensible, but TODO: add option to turn this off! */
  linear->use_ksp = PETSC_FALSE;  /* Do not default to using KSP for solving the model-fitting problem (use TAO instead). */
  PetscFunctionReturn(0);
}
