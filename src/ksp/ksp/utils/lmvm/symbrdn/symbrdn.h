#include <../src/ksp/ksp/utils/lmvm/lmvm.h>
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*
  Limited-memory Symmetric Broyden method for approximating both
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec                       *P, *Q; /* storage vectors for (B_i)*S[i] and (B_i)^{-1}*Y[i] */
  Vec                        work;
  PetscBool                  allocated, needP, needQ, useP, useQ, use_stp, use_ytq;
  PetscReal                 *stp, *ytq;  /* scalar arrays for recycling dot products */
  PetscScalar               *workscalar; /* work scalar array */
  PetscReal                 *phi, *psi;  /* convex combination factors between DFP and BFGS */
  PetscReal                  phi_scalar, psi_scalar;
  MatLMVMSymBroydenScaleType scale_type;
  PetscInt                   watchdog, max_seq_rejects; /* tracker to reset after a certain # of consecutive rejects */
  SymBroydenScaler           rescale;                   /* context for diagonal or scalar rescaling */
} Mat_SymBrdn;

PETSC_INTERN PetscErrorCode MatSymBrdnApplyJ0Fwd(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSymBrdnApplyJ0Inv(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSymBrdnUpdateJ0Scalar(Mat);

PETSC_INTERN PetscErrorCode MatView_LMVMSymBrdn(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMSymBrdn(Mat, PetscOptionItems *);
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMSymBrdn_Private(Mat, PetscOptionItems *);
