/* Example inspired by the toy example in https://www.r-bloggers.com/2020/06/understanding-lasso-and-ridge-regression-2/
 * blog post by Dr. Atakan Ekiz.
 * Here we wish to predict the number of shark attacks (that is, this number is our response variable),
 * using the following predictor variables:
 * - percentage of swimmers who watched the movie Jaws
 * - the number of swimmers in the water
 * - the average temperature of the day
 * - the price of your favorite tech stock of the day (totally uncorrelated variable) */

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

  PetscScalar y_array[20] = {98,53,39,127,73,42,71,61,83,74,85,82,62,60,43,69,67,69,85,3}; /* Number of shark attacks */

  PetscScalar X_array[80] = {37.92934, 513, 92.89899, 137.2139, /* % watched Jaws, #swimmers, temperature, stock price */
                             52.77429, 451, 87.86271, 145.7987,
                             60.84441, 456, 88.28927, 149.7299,
                             26.54302, 546, 89.43875, 147.1180,
                             54.29125, 431, 88.01132, 124.3068,
                             55.06056, 355, 88.06297, 114.1730,
                             44.25260, 557, 87.78536, 112.5773,
                             44.53368, 398, 87.49603, 125.1628,
                             44.35548, 498, 88.95234, 124.8483,
                             41.09962, 406, 89.00630, 115.9223,
                             45.22807, 610, 86.38794, 148.1111,
                             40.01614, 452, 88.83585, 131.7050,
                             42.23746, 429, 87.78222, 106.3717,
                             50.64459, 450, 87.97008, 121.1523,
                             59.59494, 337, 89.67538, 145.7158,
                             48.89715, 383, 91.12611, 123.3896,
                             44.88990, 282, 93.29563, 145.4085,
                             40.88805, 366, 88.45329, 129.8872,
                             41.62828, 471, 93.21182, 131.5871,
                             74.15835, 453, 87.68438, 143.4579};

  PetscInt rows_ix[20] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  PetscInt cols_ix[4] = {0, 1, 2, 3};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,20);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&y_predicted);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,PETSC_DECIDE,PETSC_DECIDE,20,4);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);

  if (!rank) {
    ierr = VecSetValues(y,20,rows_ix,y_array,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(X,20,rows_ix,4,cols_ix,X_array,ADD_VALUES);CHKERRQ(ierr);
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
