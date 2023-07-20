#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/

static PetscErrorCode MatSolve_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The bad Broyden kernel can be written as

   $$
     B_{k+1} x = B_k x + (y_k - B_k s_k)^T * (y_k^T B_k s_k)^{-1} s_k^T B_k x
               = (I + (y_k - B_k s_k)^T (y_k^T B_k s_k)^{-1} s_k^T) (B_k x)
   $$

   We recursively compute and store the basis (y_k - B_k s_k) and the diagonal dot products (y_k^T B_k s_k)
   in order to apply each rank one update sequentially
 */

static PetscErrorCode BadBroydenKernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec B0X)
{
  Mat_LMVM        *lmvm        = (Mat_LMVM *)B->data;
  Mat_Brdn        *lbrdn       = (Mat_Brdn *)lmvm->ctx;
  MatLMVMBasisType Y_t         = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          Y_minus_BkS = lbrdn->basis[LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode)];
  LMGramian        YtBkS       = lbrdn->gramian[LMVMModeMap(BROYDEN_GRAMIAN_YTBKS, mode)];
  PetscScalar     *StBiX;

  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGetWorkRow(B, Y_t, &StBiX));
  // These cannot be combined, notice the data dependence
  for (PetscInt i = oldest; i < next; i++) {
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_t, i, i + 1, B0X, NULL, StBiX));
    PetscCall(LMGramianSolve(YtBkS, i, i + 1, LMSOLVE_DIAGONAL, StBiX, PETSC_FALSE));
    PetscCall(LMBasisGEMV(1.0, Y_minus_BkS, i, i + 1, StBiX, 1.0, B0X));
  }
  PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_t, &StBiX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Compute the basis vectors (y_k - B_k s_k) and dot products (y_k^T B_k s_k) recursively
 */

static PetscErrorCode BadBroydenRecursiveBasisUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM *)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn *)lmvm->ctx;
  MatLMVMBasisType   Y_t   = LMVMModeMap(LMBASIS_Y, mode);
  MatLMVMBasisType   B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
  LMBasis            Y_minus_BkS;
  LMGramian          YtBkS;
  BroydenBasisType   Y_minus_BkS_t = LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode);
  BroydenGramianType YtBkS_t       = LMVMModeMap(BROYDEN_GRAMIAN_YTBKS, mode);
  PetscInt           oldest, next;

  PetscFunctionBegin;
  if (!lbrdn->basis[Y_minus_BkS_t]) PetscCall(LMBasisCreate(Y_t == LMBASIS_Y ? lmvm->Fprev : lmvm->Xprev, lmvm->m, &lbrdn->basis[Y_minus_BkS_t]));
  Y_minus_BkS = lbrdn->basis[Y_minus_BkS_t];
  if (!lbrdn->gramian[YtBkS_t]) PetscCall(LMGramianCreate(lmvm->m, &lbrdn->gramian[YtBkS_t]));
  YtBkS = lbrdn->gramian[YtBkS_t];
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (Y_minus_BkS->k < next) {
    LMBasis Y, B0S;

    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y));
    PetscCall(MatLMVMGetUpdatedBasis(B, B0S_t, &B0S));

    PetscCall(LMGramianReset(YtBkS));
    PetscCall(LMGramianUpdateNextIndex(YtBkS, next));
    Y_minus_BkS->k = next; /* k has to be the same as the other window vecs for the
                              ordering of the computations to be correct */
    // recompute each column in Y_minus_BkS in order
    for (PetscInt j = oldest; j < next; j++) {
      Vec p_j, y_j, B0s_j;

      PetscCall(LMBasisGetVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));

      // p_j starts as B_0 * s_j
      PetscCall(LMBasisGetVec(B0S, j, PETSC_MEMORY_ACCESS_READ, &B0s_j));
      PetscCall(VecCopy(B0s_j, p_j));
      PetscCall(LMBasisRestoreVec(B0S, j, PETSC_MEMORY_ACCESS_READ, &B0s_j));

      // Use the matsolve kernel to compute q_j = H_j * y_j
      PetscCall(BadBroydenKernel_Recursive_Inner(B, mode, oldest, j, p_j));
      PetscCall(LMBasisRestoreVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));

      // computes y_j^T B_k s_j and stores it on the diagonal of Y_minus_BkS
      PetscCall(LMGramianForceUpdateBlock(YtBkS, Y, Y_minus_BkS, LMBLOCK_DIAGONAL, j, j + 1));

      PetscCall(LMBasisGetVec(Y, j, PETSC_MEMORY_ACCESS_READ, &y_j));
      PetscCall(LMBasisGetVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));
      PetscCall(VecAYPX(p_j, -1.0, y_j));
      PetscCall(LMBasisRestoreVec(Y, j, PETSC_MEMORY_ACCESS_READ, &y_j));
      PetscCall(LMBasisRestoreVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec Y)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, X, Y));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(BadBroydenRecursiveBasisUpdate(B, mode));
    PetscCall(BadBroydenKernel_Recursive_Inner(B, mode, oldest, next, Y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X)
{
  MatLMVMBasisType   Y_t           = LMVMModeMap(LMBASIS_Y, mode);
  BroydenBasisType   Y_minus_BkS_t = LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode);
  BroydenGramianType YtBkS_t       = LMVMModeMap(BROYDEN_GRAMIAN_YTBKS, mode);
  Mat_LMVM          *lmvm          = (Mat_LMVM *)B->data;
  Mat_Brdn          *lbrdn         = (Mat_Brdn *)lmvm->ctx;
  LMBasis            Y_minus_BkS   = lbrdn->basis[Y_minus_BkS_t];
  LMGramian          YtBkS         = lbrdn->gramian[YtBkS_t];
  PetscScalar       *YmBkStX;

  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGetWorkRow(B, Y_t, &YmBkStX));
  // These cannot be combined, notice the data dependence
  for (PetscInt i = next - 1; i >= oldest; i--) {
    PetscCall(LMBasisGEMVH(1.0, Y_minus_BkS, i, i + 1, X, 0.0, YmBkStX));
    PetscCall(LMGramianSolve(YtBkS, i, i + 1, LMSOLVE_DIAGONAL, YmBkStX, PETSC_TRUE));
    PetscCall(MatLMVMBasisMultAdd(B, Y_t, i, i + 1, YmBkStX, X));
  }
  PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_t, &YmBkStX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;
  Vec              G = X;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(MatLMVMBasisGetWorkVec(B, Y_t, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(BadBroydenRecursiveBasisUpdate(B, mode));
    PetscCall(BadBroydenKernelHermitianTranspose_Recursive_Inner(B, mode, oldest, next, G));
  }
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0HermitianTranspose : MatLMVMApplyJ0InvHermitianTranspose)(B, G, BX));
  if (next > oldest) PetscCall(MatLMVMBasisRestoreWorkVec(B, Y_t, &G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The bad Broyden kernel can be written as

   $$
     B_k = B_0 + (Y_k - B_0 S_k) (Y^T B_0 S - trill(Y^T T))^{-1} Y_k^T B_0
   $$

   where trill is the strictly lower triangular component.  We compute and factorize
   the small matrix in order to apply a single rank m update
 */

static PetscErrorCode BadBroydenCompactGramianUpdate(Mat B, MatLMVMMode mode)
{
  MatLMVMBasisType   Y_t               = LMVMModeMap(LMBASIS_Y, mode);
  MatLMVMBasisType   B0S_t             = LMVMModeMap(LMBASIS_B0S, mode);
  BroydenGramianType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_GRAMIAN_YTB0S_MINUS_YTY, mode);
  Mat_LMVM          *lmvm              = (Mat_LMVM *)B->data;
  Mat_Brdn          *lbrdn             = (Mat_Brdn *)lmvm->ctx;
  LMGramian          YtB0S, YtY, YtB0S_minus_YtY;
  PetscInt           oldest, next;

  PetscFunctionBegin;
  if (!lbrdn->gramian[YtB0S_minus_YtY_t]) PetscCall(LMGramianCreate(lmvm->m, &lbrdn->gramian[YtB0S_minus_YtY_t]));
  YtB0S_minus_YtY = lbrdn->gramian[YtB0S_minus_YtY_t];
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (YtB0S_minus_YtY->k < next) {
    PetscCall(MatLMVMGetUpdatedGramian(B, Y_t, B0S_t, LMBLOCK_ALL, &YtB0S));
    PetscCall(MatLMVMGetUpdatedGramian(B, Y_t, Y_t, LMBLOCK_LOWER_TRIANGLE, &YtY));
    PetscCall(LMGramianCopy(YtB0S, YtB0S_minus_YtY));
    PetscCall(LMGramianAXPY(YtB0S_minus_YtY, LMBLOCK_ALL, -1.0, YtY, LMBLOCK_STRICT_LOWER_TRIANGLE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM          *lmvm              = (Mat_LMVM *)B->data;
    Mat_Brdn          *lbrdn             = (Mat_Brdn *)lmvm->ctx;
    MatLMVMBasisType   Y_t               = LMVMModeMap(LMBASIS_Y, mode);
    MatLMVMBasisType   Y_minus_B0S_t     = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    BroydenGramianType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_GRAMIAN_YTB0S_MINUS_YTY, mode);
    LMGramian          YtB0S_minus_YtY;
    PetscScalar       *YtB0X;

    PetscCall(BadBroydenCompactGramianUpdate(B, mode));
    YtB0S_minus_YtY = lbrdn->gramian[YtB0S_minus_YtY_t];
    PetscCall(MatLMVMBasisGetWorkRow(B, Y_t, &YtB0X));
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_t, oldest, next, BX, NULL, YtB0X));
    PetscCall(LMGramianSolve(YtB0S_minus_YtY, oldest, next, LMSOLVE_LU, YtB0X, PETSC_FALSE));
    PetscCall(MatLMVMBasisMultAdd(B, Y_minus_B0S_t, oldest, next, YtB0X, BX));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_t, &YtB0X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;
  Vec              G = X;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM          *lmvm              = (Mat_LMVM *)B->data;
    Mat_Brdn          *lbrdn             = (Mat_Brdn *)lmvm->ctx;
    MatLMVMBasisType   Y_minus_B0S_t     = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    BroydenGramianType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_GRAMIAN_YTB0S_MINUS_YTY, mode);
    LMGramian          YtB0S_minus_YtY;
    PetscScalar       *YmB0StG;

    PetscCall(MatLMVMBasisGetWorkVec(B, Y_t, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(BadBroydenCompactGramianUpdate(B, mode));
    YtB0S_minus_YtY = lbrdn->gramian[YtB0S_minus_YtY_t];
    PetscCall(MatLMVMBasisGetWorkRow(B, Y_t, &YmB0StG));
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_minus_B0S_t, oldest, next, G, NULL, YmB0StG));
    PetscCall(LMGramianSolve(YtB0S_minus_YtY, oldest, next, LMSOLVE_LU, YmB0StG, PETSC_TRUE));
    PetscCall(MatLMVMBasisMultAdd(B, Y_t, oldest, next, YmB0StG, G));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_t, &YmB0StG));
  }
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0HermitianTranspose : MatLMVMApplyJ0InvHermitianTranspose)(B, G, BHX));
  if (next > oldest) PetscCall(MatLMVMBasisRestoreWorkVec(B, Y_t, &G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_Recursive(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMBadBrdn(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "\"Bad\" Broyden method for approximating Jacobian action (MATLMVMBADBROYDEN)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_matvec_type", "Algorithm used to matrix vector products", "", MatLMVMMatvecTypes, (PetscEnum)lmvm->matvec_type, (PetscEnum *)&lmvm->matvec_type, NULL));
  PetscOptionsHeadEnd();

  switch (lmvm->matvec_type) {
  case MATLMVM_MATVEC_RECURSIVE:
    lmvm->ops->mult    = MatMult_LMVMBadBrdn_Recursive;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_Recursive;
    lmvm->ops->solve   = MatSolve_LMVMBadBrdn_Recursive;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_Recursive;
    break;
  case MATLMVM_MATVEC_COMPACT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBadBrdn_CompactDense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_CompactDense;
    lmvm->ops->solve   = MatSolve_LMVMBadBrdn_CompactDense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_Recursive;
    break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBadBrdn(Mat B)
{
  Mat_LMVM *lmvm;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBADBROYDEN));
  B->ops->setfromoptions = MatSetFromOptions_LMVMBadBrdn;
  lmvm                   = (Mat_LMVM *)B->data;

  lmvm->ops->mult    = MatMult_LMVMBadBrdn_CompactDense;
  lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_CompactDense;
  lmvm->ops->solve   = MatSolve_LMVMBadBrdn_CompactDense;
  lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_CompactDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMBadBroyden - Creates a limited-memory modified (aka "bad") Broyden-type
  approximation matrix used for a Jacobian. L-BadBrdn is not guaranteed to be
  symmetric or positive-definite.

  To use the L-BadBrdn matrix with other vector types, the matrix must be
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBADBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBADBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
