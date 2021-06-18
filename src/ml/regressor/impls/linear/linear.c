#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscmlregressor.h" I*/

PetscErrorCode MLRegressorSetUp_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  if (!linear->ksp) {
    ierr = MLRegressorLinearGetKSP(mlregressor,&linear->ksp);CHKERRQ(ierr);
    // TODO: Figure out if I need to set operators for the KSP here or set the operator X.
    // I think maybe I can just do this stuff in the Fit() routine.
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorReset_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  ierr = MatDestroy(&linear->X);CHKERRQ(ierr);
  ierr = KSPDestroy(&linear->ksp);CHKERRQ(ierr);

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

PetscErrorCode MLRegressorFit_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;
  KSP ksp;
  Mat A;  /* The operator we will pass to the KSP; this could be something like a composite matrix using MATCENTERING. */
  Mat AtA;

  PetscFunctionBegin;
  if (!linear->ksp) {ierr = MLRegressorLinearGetKSP(mlregressor,&linear->ksp);CHKERRQ(ierr);}
  ksp = linear->ksp;

  if (linear->fit_intercept) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Linear MLRegressor intercept fitting is not yet implemented!");
    /* TODO: If we are fitting the intercept, we probably need to make A a composite matrix using MATCENTERING. 
     * Though there might be some cases we don't want to do this for, depending on what kind of matrix is passed in. 
     * We will also need to ensure that the right-hand side passed to the KSP is also mean-centered, since we
     * intend to compute the intercept separately from regression coefficients (that is, we will not be adding a
     * column of all 1s to our design matrix). */
  } else {
    /* When not fitting intercept, we assume that the input data are already centered.
     * TODO: Perhaps revisit exactly what options should exist around this. */
    A = mlregressor->training;
  }

  /* Now use the KSP to solve the least squares problem using KSPLSQR.
   * TODO: Add options to use other methods. */
  ierr = MatCreateNormal(A,&AtA);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,AtA);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);  // TODO: Does this have the right option prefixes set?
  ierr = KSPSolve(ksp,mlregressor->target,linear->coefficients);CHKERRQ(ierr);

  /* Calculate the intercept. */
  if (linear->fit_intercept) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Linear MLRegressor intercept fitting is not yet implemented!");
    /* TODO: Write the code to calculate the intercept here! */
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
  linear->fit_intercept = PETSC_FALSE;  /* TODO: This should probably default to true; but using false for now so I can initially implement less! */
  PetscFunctionReturn(0);
}
