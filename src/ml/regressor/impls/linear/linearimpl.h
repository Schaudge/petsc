#if !defined(__PETSCREGRESSORLINEAR)
#define __PETSCREGRESSORLINEAR

#include <petsc/private/regressorimpl.h>
#include <petscksp.h>

/* We define this header, since it serves as a "base" for all linear models. */
#define REGRESSOR_LINEAR_HEADER \
  /* Parameters of the fitted regression model */           \
  Vec coefficients;                                         \
  PetscScalar intercept;                                    \
                                                            \
  Mat X;   /* Operator of the linear model; often the training data matrix, but might be a MATCOMPOSITE */ \
  Mat C;   /* Centering matrix */ \
  Vec rhs; /* Right-hand side of the linear model; often the target vector, but may be the mean-centered version */ \
  /* Various options */ \
  PetscBool fit_intercept  /* Calculate intercept ("bias" or "offset") if true. Assume centered data if false. */

typedef struct {
  REGRESSOR_LINEAR_HEADER;

  KSP ksp;
  Mat XtX; /* Normal matrix formed from X */

} PETSCREGRESSOR_LINEAR;

#endif
