#ifndef PETSCSINDY_H
#define PETSCSINDY_H

#include <petsctao.h>

PETSC_EXTERN PetscErrorCode SINDyCreateBasis(Vec x, PetscInt poly_order, PetscInt sine_order, Mat* Theta, PetscInt *num_bases);
PETSC_EXTERN PetscErrorCode SINDySparseLeastSquares(Mat A, Vec b, Mat D, Vec x);

#endif
