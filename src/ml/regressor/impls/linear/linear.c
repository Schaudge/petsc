#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscmlregressor.h" I*/

PetscErrorCode MLRegressorSetUp_Linear(MLRegressor mlregressor)
{
  //MPI_Comm comm;
  PetscErrorCode ierr;
  PetscInt M,N;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;
  KSP ksp;

  PetscFunctionBegin;
  if (!linear->ksp) {
    ierr = MLRegressorLinearGetKSP(mlregressor,&linear->ksp);CHKERRQ(ierr);
    // TODO: Figure out if I need to set operators for the KSP here or set the operator X.
    // I think maybe I can just do this stuff in the Fit() routine.
  }
  ksp = linear->ksp;

  ierr = MatGetSize(mlregressor->training,&M,&N);CHKERRQ(ierr);

  if (linear->fit_intercept) {
    /* If we are fitting the intercept, we need to make A a composite matrix using MATCENTERING to preserve sparsity.
     * Though there might be some cases we don't want to do this for, depending on what kind of matrix is passed in. (Probably bad idea for dense?)
     * We will also need to ensure that the right-hand side passed to the KSP is also mean-centered, since we
     * intend to compute the intercept separately from regression coefficients (that is, we will not be adding a
     * column of all 1s to our design matrix). */
    ierr = MatCreateCentering(PetscObjectComm((PetscObject)mlregressor),PETSC_DECIDE,M,&linear->C);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)mlregressor),&linear->X);CHKERRQ(ierr);
    ierr = MatSetSizes(linear->X,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
    ierr = MatSetType(linear->X,MATCOMPOSITE);CHKERRQ(ierr);
    ierr = MatCompositeSetType(linear->X,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
    ierr = MatCompositeAddMat(linear->X,mlregressor->training);CHKERRQ(ierr);
    ierr = MatCompositeAddMat(linear->X,linear->C);CHKERRQ(ierr);
    ierr = VecDuplicate(mlregressor->target,&linear->rhs);CHKERRQ(ierr);
    ierr = MatMult(linear->C,mlregressor->target,linear->rhs);CHKERRQ(ierr);
  } else {
    /* When not fitting intercept, we assume that the input data are already centered.
     * TODO: Perhaps revisit exactly what options should exist around this. */
    linear->X = mlregressor->training;
    linear->rhs = mlregressor->target;
  }

  if (linear->coefficients) {ierr = VecDestroy(&linear->coefficients);CHKERRQ(ierr);}
  ierr = MatCreateVecs(linear->X,&linear->coefficients,NULL);CHKERRQ(ierr);

  /* Set up the KSP to solve the least squares problem (without solving for intercept, as this is done separately) using KSPLSQR.
   * TODO: Add options to use other methods. */
  ierr = MatCreateNormal(linear->X,&linear->XtX);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,linear->X,linear->XtX);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);  // TODO: Does this have the right option prefixes set?
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorReset_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  ierr = MatDestroy(&linear->X);CHKERRQ(ierr);
  ierr = MatDestroy(&linear->XtX);CHKERRQ(ierr);
  ierr = MatDestroy(&linear->C);CHKERRQ(ierr);
  ierr = KSPDestroy(&linear->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->coefficients);CHKERRQ(ierr);
  ierr = VecDestroy(&linear->rhs);CHKERRQ(ierr);

  /* Reset options/parameters to the setupcalled = 0 state. */
  /* TODO: Add the reset code once the linear regressor is fleshed out enough to need resetting! */
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorDestroy_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MLRegressorReset_Linear(mlregressor);CHKERRQ(ierr);
  ierr = PetscFree(mlregressor->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject, MLRegressor mlregressor)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorView_Linear(MLRegressor mlregressor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorLinearGetKSP(MLRegressor mlregressor,KSP *ksp)
{
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  PetscValidPointer(ksp,2);
  /* Analogous to how SNESGetKSP() operates, this routine should create the KSP if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!linear->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)mlregressor),&linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)linear->ksp,(PetscObject)mlregressor,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mlregressor,(PetscObject)linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)linear->ksp,((PetscObject)mlregressor)->options);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MLRegressorLinearGetCoefficients(MLRegressor mlregressor, Vec *coefficients)
{
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  PetscValidPointer(coefficients,2);
  *coefficients = linear->coefficients;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MLRegressorLinearGetIntercept(MLRegressor mlregressor, PetscScalar *intercept)
{
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mlregressor,MLREGRESSOR_CLASSID,1);
  PetscValidPointer(intercept,2);
  *intercept = linear->intercept;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorFit_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;
  KSP ksp;
  PetscScalar target_mean,*column_means_global,*column_means_local,column_means_dot_coefficients;
  Vec column_means;
  PetscInt m,N,istart,i;

  PetscFunctionBegin;
  if (!linear->ksp) {ierr = MLRegressorLinearGetKSP(mlregressor,&linear->ksp);CHKERRQ(ierr);}
  ksp = linear->ksp;

  /* Solve the least-squares problem (previously set up in MLRegressorSetUp_Linear()) without finding the intercept. */
  ierr = KSPSolve(ksp,linear->rhs,linear->coefficients);CHKERRQ(ierr);

  /* Calculate the intercept. */
  if (linear->fit_intercept) {
    ierr = MatGetSize(mlregressor->training,NULL,&N);CHKERRQ(ierr);
    ierr = PetscMalloc1(N,&column_means_global);CHKERRQ(ierr);
    ierr = VecMean(mlregressor->target,&target_mean);CHKERRQ(ierr);
    /* We need the means of all columns of mlregressor->training, placed into a Vec compatible with linear->coefficients.
     * Note the potential scalability issue: MatGetColumnMeans() computes means of ALL colummns. */
    ierr = MatGetColumnMeans(mlregressor->training,column_means_global);CHKERRQ(ierr);
    /* TODO: Calculation of the Vec and matrix column means should probably go into the SetUp phase, and also be placed
     *       into a routine that is callable from outside of MLRegressorFit_Linear(), because we'll want to do the same
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

PETSC_EXTERN PetscErrorCode MLRegressorPredict_Linear(MLRegressor mlregressor, Mat X, Vec y)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  ierr = MatMult(X,linear->coefficients,y);CHKERRQ(ierr);
  ierr = VecShift(y,linear->intercept);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MLRegressorCreate_Linear(MLRegressor mlregressor)
{
  MLREGRESSOR_LINEAR *linear;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(mlregressor,&linear);CHKERRQ(ierr);
  mlregressor->data = (void*)linear;

  mlregressor->ops->setup          = MLRegressorSetUp_Linear;
  mlregressor->ops->reset          = MLRegressorReset_Linear;
  mlregressor->ops->destroy        = MLRegressorDestroy_Linear;
  mlregressor->ops->setfromoptions = MLRegressorSetFromOptions_Linear;
  mlregressor->ops->view           = MLRegressorView_Linear;
  mlregressor->ops->fit            = MLRegressorFit_Linear;
  mlregressor->ops->predict        = MLRegressorPredict_Linear;

  linear->intercept = 0.0;
  linear->fit_intercept = PETSC_TRUE;  /* Defaulting to calculating the intercept is probably sensible, but TODO: add option to turn this off! */
  PetscFunctionReturn(0);
}
