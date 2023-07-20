#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating both
  the forward product and inverse application of a Jacobian.
 */

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is
  matrix-vector product version of the recursive formula given in
  Equation (6.15) of Nocedal and Wright "Numerical Optimization" 2nd
  edition, pg 139.

  Note: Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i = 0,1,2,...,k
    # Q[i] = (B_i)^{-1} * Y[i]
    gamma = (S[i]^T F) / (S[i]^T Y[i])
    zeta = (Q[i]^T F) / (Y[i]^T Q[i])
    dX <- dX + (gamma * S[i]) - (zeta * Q[i])
  end
*/
PetscErrorCode MatSolve_LMVMDFP(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (ldfp->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (PetscInt i = 0; i < next - oldest; ++i) {
      PetscReal ytq;
      Vec       y_i;

      PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      PetscCall(MatSymBrdnApplyJ0Inv(B, y_i, ldfp->Q[i]));
      /* Compute the necessary dot products */
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar qjtyi, sjtyi, yjtsj;
        Vec         s_j;

        PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + j, &yjtsj));
        PetscCall(MatLMVMGetVecsRead(B, oldest + j, LMBASIS_S, &s_j));
        PetscCall(VecDot(y_i, ldfp->Q[j], &qjtyi));
        PetscCall(VecDot(y_i, s_j, &sjtyi));
        PetscCall(VecAXPBYPCZ(ldfp->Q[i], -qjtyi / ldfp->ytq[j], sjtyi / yjtsj, 1.0, ldfp->Q[j], s_j));
        PetscCall(MatLMVMRestoreVecsRead(B, oldest + j, LMBASIS_S, &s_j));
      }
      PetscCall(VecDotRealPart(y_i, ldfp->Q[i], &ytq));
      PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_Y, &y_i));
      ldfp->ytq[i] = ytq;
    }
    ldfp->needQ = PETSC_FALSE;
  }

  /* Start the outer loop (i) for the recursive formula */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));
  /* Get all the dot products we need */
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscScalar qitf, sitf, yitsi;
    Vec         s_i;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i));
    PetscCall(VecDot(F, ldfp->Q[i], &qitf));
    PetscCall(VecDot(F, s_i, &sitf));
    PetscCall(VecAXPBYPCZ(dX, -qitf / ldfp->ytq[i], sitf / yitsi, 1.0, ldfp->Q[i], s_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free
  implementation of the recursive formula given in Equation 6.13 of
  Nocedal and Wright "Numerical Optimization" 2nd edition, pg 139.

  This forward product has a two-loop form similar to the BFGS two-loop
  formulation for the inverse Jacobian application. However, the S and
  Y vectors have interchanged roles.

  work <- X

  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha[i] = rho[i] * (Y[i]^T work)
    work <- work - (alpha[i] * S[i])
  end

  Z <- J0 * work

  for i = 0,1,2,...,k
    beta = rho[i] * (S[i]^T Y)
    Z <- Z + ((alpha[i] - beta) * Y[i])
  end
*/
PetscErrorCode MatMult_LMVMDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  PetscScalar *alpha, beta;
  PetscScalar  ytx, stz;

  PetscFunctionBegin;
  /* Copy the function into the work vector for the first loop */
  PetscCall(VecCopy(X, ldfp->work));

  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  /* Start the first loop */
  PetscCall(PetscMalloc1(next - oldest, &alpha));
  for (PetscInt i = next - oldest - 1; i >= 0; --i) {
    PetscScalar yitsi;
    Vec         s_i, y_i;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(ldfp->work, y_i, &ytx));
    alpha[i] = ytx / yitsi;
    PetscCall(VecAXPY(ldfp->work, -alpha[i], s_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }

  /* Apply the forward product with initial Jacobian */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, ldfp->work, Z));

  /* Start the second loop */
  for (PetscInt i = 0; i < next - oldest; ++i) {
    PetscScalar yitsi;
    Vec         s_i, y_i;

    PetscCall(MatLMVMGramianGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, oldest + i, &yitsi));
    PetscCall(MatLMVMGetVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
    PetscCall(VecDot(Z, s_i, &stz));
    beta = stz / yitsi;
    PetscCall(VecAXPY(Z, alpha[i] - beta, y_i));
    PetscCall(MatLMVMRestoreVecsRead(B, oldest + i, LMBASIS_S, &s_i, LMBASIS_Y, &y_i));
  }
  PetscCall(PetscFree(alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMDFP(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "DFP method for approximating SPD Jacobian actions (MATLMVMDFP)");
  PetscCall(SymBroydenScalerSetFromOptions(B, ldfp->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDFP(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDFP));
  B->ops->setfromoptions = MatSetFromOptions_LMVMDFP;
  B->ops->solve          = MatSolve_LMVMDFP;

  lmvm            = (Mat_LMVM *)B->data;
  lmvm->ops->mult = MatMult_LMVMDFP;

  ldfp             = (Mat_SymBrdn *)lmvm->ctx;
  ldfp->needP      = PETSC_FALSE;
  ldfp->useP       = PETSC_FALSE;
  ldfp->use_stp    = PETSC_FALSE;
  ldfp->phi_scalar = 1.0;
  ldfp->psi_scalar = 0.0;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix
  used for approximating Jacobians. L-DFP is symmetric positive-definite by
  construction, and is the dual of L-BFGS where Y and S vectors swap roles.

  To use the L-DFP matrix with other vector types, the matrix must be
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMDFP`, `MatCreateLMVMBFGS()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
