static char help[] = "Tests basic creation and destruction of PetscRegressor objects.\n\n";

#include <petscregressor.h>

int main(int argc,char **args)
{
  PetscRegressor regressor;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  Mat X;
  Vec y,y_predicted,coefficients;
  PetscScalar intercept;
  /* y_array[] and X_array[] are NOT mean-centered; in ex1.c they are! */
  PetscScalar y_array[5] = {1.0,0.5,0,0.5,2};
  PetscScalar X_array[10] = {-1.00000,  1.00000,
                             -0.50000,  0.25000,
                              0.00000,  0.00000,
                              0.50000,  0.25000,
                              1.00000,  1.00000};
  PetscInt rows_ix[5] = {0, 1, 2, 3, 4};
  PetscInt cols_ix[2] = {0, 1};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,5);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y_predicted);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,PETSC_DECIDE,PETSC_DECIDE,5,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecSetValues(y,5,rows_ix,y_array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(X,5,rows_ix,2,cols_ix,X_array,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscRegressorCreate(PETSC_COMM_WORLD,&regressor);CHKERRQ(ierr);
  ierr = PetscRegressorSetType(regressor,PETSCREGRESSORLINEAR);CHKERRQ(ierr);
  ierr = PetscRegressorSetFromOptions(regressor);
  ierr = PetscRegressorFit(regressor,X,y);CHKERRQ(ierr);
  ierr = PetscRegressorPredict(regressor,X,y_predicted);CHKERRQ(ierr);
  ierr = PetscRegressorLinearGetIntercept(regressor,&intercept);CHKERRQ(ierr);
  ierr = PetscRegressorLinearGetCoefficients(regressor,&coefficients);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Intercept is %lf\n",intercept);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coefficients are\n");CHKERRQ(ierr);
  ierr = VecView(coefficients,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Predicted values are\n");CHKERRQ(ierr);
  ierr = VecView(y_predicted,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);

  ierr = PetscRegressorDestroy(&regressor);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
