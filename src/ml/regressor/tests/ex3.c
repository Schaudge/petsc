static char help[] = "Tests basic creation and destruction of PetscRegressor objects.\n\n";

#include <petscregressor.h>

int main(int argc,char **args)
{
  PetscRegressor regressor;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt i, N=10;
  Mat X;
  Vec y,y_predicted,coefficients;
  PetscScalar intercept, mean;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Options for PetscRegressor ex3:", "");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "Dimension of the N x N data matrix", NULL, N, &N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y_predicted);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);

  /* Set up a training data matrix that is the identity.
   * We do this because this gives us a special case in which we can analytically determine what the regression
   * coefficients should be for ordinary least squares, LASSO (L1 regularized), and ridge (L2 regularized) regression.
   * See details in section 6.2 of James et al.'s An Introduction to Statistical Learning (ISLR), in the subsection
   * titled "A Simple Special Case for Ridge Regression and the Lasso".
   * Note that the coefficients we generate with ridge regression (-tao_brgn_regularization_type l2pure -tao_brgn_regularizer_weight <lambda>)
   * match those of the ISLR formula exactly. For LASSO it does not match the ISLR formula: where they use lambda/2, we need to use lambda.
   * It also doesn't match what Scikit-learn does; in that case their lambda is 1/n_samples of our lambda. Apparently everyone is scaling
   * their loss function by a different value, hence the need to change what "lambda" is. But it's clear that ISLR, Scikit-learn, and we
   * are basically doing the same thing otherwise. */
  if (!rank) {
    for (i=0; i<N; i++) {
      ierr = VecSetValue(y,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(X,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* Center the target vector we will train with. */
  ierr = VecMean(y,&mean);CHKERRQ(ierr);
  ierr = VecShift(y,-1.0 * mean);CHKERRQ(ierr);

  ierr = PetscRegressorCreate(PETSC_COMM_WORLD,&regressor);CHKERRQ(ierr);
  ierr = PetscRegressorSetType(regressor,PETSCREGRESSORLINEAR);CHKERRQ(ierr);
  ierr = PetscRegressorLinearSetFitIntercept(regressor,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscRegressorSetFromOptions(regressor);
  ierr = PetscRegressorFit(regressor,X,y);CHKERRQ(ierr);
  ierr = PetscRegressorPredict(regressor,X,y_predicted);CHKERRQ(ierr);
  ierr = PetscRegressorLinearGetIntercept(regressor,&intercept);CHKERRQ(ierr);
  ierr = PetscRegressorLinearGetCoefficients(regressor,&coefficients);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Training target vector is\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Predicted values are\n");CHKERRQ(ierr);
  ierr = VecView(y_predicted,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coefficients are\n");CHKERRQ(ierr);
  ierr = VecView(coefficients,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);

  ierr = PetscRegressorDestroy(&regressor);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
