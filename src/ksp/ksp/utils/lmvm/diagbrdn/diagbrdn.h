#if !defined(DIAGBRDN_H)
  #define DIAGBRDN_H
  #include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  "Full" memory implementation of only the diagonal terms in a symmetric Broyden approximation.
*/

typedef struct _SymBroydenScaler *SymBroydenScaler;

struct _SymBroydenScaler {
  PetscInt                   k;
  Vec                        invDnew, BFGS, DFP, U, V, W; /* work vectors for diagonal scaling */
  PetscReal                 *yty, *sts, *yts;             /* scalar arrays for recycling dot products */
  PetscReal                  theta, rho, alpha, beta;     /* convex combination factors for the scalar or diagonal scaling */
  PetscReal                  delta, delta_min, delta_max, sigma, tol;
  PetscInt                   sigma_hist; /* length of update history to be used for scaling */
  PetscBool                  allocated;
  PetscBool                  forward;
  MatLMVMSymBroydenScaleType scale_type;
};

PETSC_INTERN PetscErrorCode SymBroydenScalerSetDiagonalMode(SymBroydenScaler, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenScalerGetType(SymBroydenScaler, MatLMVMSymBroydenScaleType *);
PETSC_INTERN PetscErrorCode SymBroydenScalerSetType(Mat, SymBroydenScaler, MatLMVMSymBroydenScaleType);
PETSC_INTERN PetscErrorCode SymBroydenScalerSetDelta(SymBroydenScaler, PetscReal);
PETSC_INTERN PetscErrorCode SymBroydenScalerInitializeJ0(Mat, SymBroydenScaler);
PETSC_INTERN PetscErrorCode SymBroydenScalerUpdateJ0(Mat, SymBroydenScaler);
PETSC_INTERN PetscErrorCode SymBroydenScalerUpdate(Mat, SymBroydenScaler);
PETSC_INTERN PetscErrorCode SymBroydenScalerCopy(SymBroydenScaler, SymBroydenScaler, PetscInt);
PETSC_INTERN PetscErrorCode SymBroydenScalerView(SymBroydenScaler, PetscViewer);
PETSC_INTERN PetscErrorCode SymBroydenScalerSetFromOptions(Mat, SymBroydenScaler, PetscOptionItems *PetscOptionsObject);
PETSC_INTERN PetscErrorCode SymBroydenScalerReset(Mat, SymBroydenScaler, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenScalerDestroy(SymBroydenScaler *);
PETSC_INTERN PetscErrorCode SymBroydenScalerAllocate(Mat, SymBroydenScaler);
PETSC_INTERN PetscErrorCode SymBroydenScalerCreate(SymBroydenScaler *);

#endif
