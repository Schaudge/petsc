#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*
  Limited-memory Broyden-Fletcher-Goldfarb-Shano method for approximating both
  the forward product and inverse application of a Jacobian.
*/

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is adapted
   from Algorithm 7.4 on page 178 of Nocedal and Wright "Numerical Optimization"
   2nd edition (https://doi.org/10.1007/978-0-387-40065-5). The initial inverse
   Jacobian application falls back onto the gamma scaling recommended in equation
   (7.20) if the user has not provided any estimation of the initial Jacobian or
   its inverse.

   work <- F

   for i = k,k-1,k-2,...,0
     rho[i] = 1 / (Y[i]^T S[i])
     alpha[i] = rho[i] * (S[i]^T work)
     Fwork <- work - (alpha[i] * Y[i])
   end

   dX <- J0^{-1} * work

   for i = 0,1,2,...,k
     beta = rho[i] * (Y[i]^T dX)
     dX <- dX + ((alpha[i] - beta) * S[i])
   end
*/
PetscErrorCode MatSolve_LMVMBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
  PetscScalar *alpha, beta;
  PetscScalar  stf, ytx;

  PetscFunctionBegin;
  /* Copy the function into the work vector for the first loop */
  PetscCall(VecCopy(F, lbfgs->work));

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  /* Start the first loop */
  PetscCall(PetscMalloc1(lmvm->k + 1, &alpha));
  for (PetscInt i = next - oldest - 1; i >= 0; --i) {
    Vec s_i, y_i;

    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(lbfgs->work, s_i, &stf));
    alpha[i] = stf / lbfgs->rescale->yts[i];
    PetscCall(VecAXPY(lbfgs->work, -alpha[i], y_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }

  /* Invert the initial Jacobian onto the work vector (or apply scaling) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, lbfgs->work, dX));

  /* Start the second loop */
  for (PetscInt i = 0; i < next - oldest; ++i) {
    Vec s_i, y_i;

    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(dX, y_i, &ytx));
    beta = ytx / lbfgs->rescale->yts[i];
    PetscCall(VecAXPY(dX, alpha[i] - beta, s_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }
  PetscCall(PetscFree(alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free
  implementation of Equation (6.19) in Nocedal and Wright "Numerical
  Optimization" 2nd Edition, pg 140.

  This forward product has the same structure as the inverse Jacobian
  application in the DFP formulation, except with S and Y exchanging
  roles.

  Note: P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i = 0,1,2,...,k
    P[i] <- J0 * S[i]
    for j = 0,1,2,...,(i-1)
      gamma = (Y[j]^T S[i]) / (Y[j]^T S[j])
      zeta = (P[j]^T S[i]) / (S[j]^T P[j])
      P[i] <- P[i] - (zeta * P[j]) + (gamma * Y[j])
    end
    gamma = (Y[i]^T X) / (Y[i]^T S[i])
    zeta = (P[i]^T X) / (S[i]^T P[i])
    Z <- Z - (zeta * P[i]) + (gamma * Y[i])
  end
*/
PetscErrorCode MatMult_LMVMBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lbfgs->needP) {
    /* Pre-compute (P[i] = B_i * S[i]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      PetscReal stp;
      Vec       s_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
      PetscCall(MatSymBrdnApplyJ0Fwd(B, s_i, lbfgs->P[i]));
      /* Compute the necessary dot products */
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar pjtsi, yjtsi;
        Vec         y_j;

        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
        PetscCall(VecDot(s_i, lbfgs->P[j], &pjtsi));
        PetscCall(VecDot(s_i, y_j, &yjtsi));
        PetscCall(VecAXPBYPCZ(lbfgs->P[i], -pjtsi / lbfgs->stp[j], yjtsi / lbfgs->rescale->yts[j], 1.0, lbfgs->P[j], y_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_Y, &y_j));
      }
      PetscCall(VecDotRealPart(s_i, lbfgs->P[i], &stp));
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
      lbfgs->stp[i] = stp;
    }
    lbfgs->needP = PETSC_FALSE;
  }

  /* Start the outer loop (i) for the recursive formula */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));
  /* Get all the dot products we need */
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscScalar pitx, yitx;
    Vec         y_i;

    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(X, lbfgs->P[i], &pitx));
    PetscCall(VecDot(X, y_i, &yitx));
    PetscCall(VecAXPBYPCZ(Z, -pitx / lbfgs->stp[i], yitx / lbfgs->rescale->yts[i], 1.0, lbfgs->P[i], y_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMBFGS(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "L-BFGS method for approximating SPD Jacobian actions (MATLMVMBFGS)");
  PetscCall(SymBroydenScalerSetFromOptions(B, lbfgs->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS));
  B->ops->setfromoptions = MatSetFromOptions_LMVMBFGS;
  B->ops->solve          = MatSolve_LMVMBFGS;

  lmvm            = (Mat_LMVM *)B->data;
  lmvm->ops->mult = MatMult_LMVMBFGS;

  lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  lbfgs->needQ   = PETSC_FALSE;
  lbfgs->useQ    = PETSC_FALSE;
  lbfgs->use_ytq = PETSC_FALSE;
  lbfgs->phi     = 0.0;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
  matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by
  construction, and is commonly used to approximate Hessians in optimization
  problems.

  To use the L-BFGS matrix with other vector types, the matrix must be
  created using `MatCreate()` and `MatSetType()`, followed by `MatLMVMAllocate()`.
  This ensures that the internal storage and work vectors are duplicated from the
  correct type of vector.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBFGS`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
