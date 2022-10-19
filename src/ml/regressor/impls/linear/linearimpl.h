#if !defined(__PETSCREGRESSORLINEAR)
#define __PETSCREGRESSORLINEAR

#include <petsc/private/regressorimpl.h>
#include <petscksp.h>

typedef struct {
  /* Note: It might make sense for all of this to eventually be defined as the macro PETSCREGRESSORLINEARHEADER,
   * since that it may form a "base" that other linear models might use. */

  /* Parameters of the fitted regression model */
  Vec coefficients;
  PetscScalar intercept;

  KSP ksp;
  Mat X;   /* Operator passed to the KSP; often the training data matrix, but might be a MATCOMPOSITE */
  Mat XtX; /* Normal matrix formed from X */
  Mat C;   /* Centering matrix */
  Vec rhs; /* Right-hand side used with the KSP; often the target vector, but may be the mean-centered version */

  /* Various options */
  PetscBool fit_intercept;  /* Calculate intercept ("bias" or "offset") if true. Assume centered data if false. */
} PETSCREGRESSOR_LINEAR;

#endif
