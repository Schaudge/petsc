#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

/*------------------------------------------------------------*/

// MatSolve Broyden is like MatMult bad Broyden

// --- MatSolve recursive ---

static PetscErrorCode MatSolve_LMVMBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// --- MatSolve compact dense ---

static PetscErrorCode MatSolve_LMVMBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode BroydenKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  Vec              G   = X;
  Vec              W   = BX;
  MatLMVMBasisType S_t = MatLMVMBasisMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t = MatLMVMBasisMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscScalar *StG;

    PetscCall(MatLMVMGramianUpdate(B, S_t, S_t, LMBLOCK_DIAGONAL));
    PetscCall(MatLMVMBasisGetWorkVec(B, S_t, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(MatLMVMBasisGetWorkRow(B, S_t, &StG));
    PetscCall(VecZeroEntries(BX));
    for (PetscInt i = next - 1; i >= oldest; i--) {
      PetscCall(MatLMVMBasisMultHermitianTranspose(B, S_t, i, i + 1, G, NULL, StG));
      PetscCall(MatLMVMGramianSolve(B, i, i + 1, S_t, S_t, LMSOLVE_DIAGONAL, StG, PETSC_FALSE));
      PetscCall(MatLMVMBasisMultAdd(B, Y_t, i, i + 1, StG, BX));
      PetscCall(MatLMVMBasisGEMV(B, S_t, i, i + 1, -1.0, StG, 1.0, G));
    }
    PetscCall(MatLMVMBasisRestoreWorkRow(B, S_t, &StG));
    PetscCall(MatLMVMBasisGetWorkVec(B, Y_t, &W));
  }
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, G, W));
  if (next > oldest) {
    PetscCall(VecAXPY(BX, 1.0, W));
    PetscCall(MatLMVMBasisRestoreWorkVec(B, Y_t, &W));
    PetscCall(MatLMVMBasisRestoreWorkVec(B, S_t, &G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  MatLMVMBasisType S_t = MatLMVMBasisMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t = MatLMVMBasisMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0HermitianTranspose : MatLMVMApplyJ0InvHermitianTranspose)(B, X, BHX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscScalar *StBHX;
    PetscScalar *YtX;

    PetscCall(MatLMVMGramianUpdate(B, S_t, S_t, LMBLOCK_DIAGONAL));
    PetscCall(MatLMVMBasisGetWorkRow(B, S_t, &StBHX));
    PetscCall(MatLMVMBasisGetWorkRow(B, Y_t, &YtX));
    for (PetscInt i = oldest; i < next; i++) {
      PetscCall(MatLMVMBasisMultHermitianTranspose(B, S_t, i, i + 1, BHX, NULL, StBHX));
      PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_t, i, i + 1, X, NULL, YtX));
      PetscCall(MatLMVMGramianSolve(B, i, i + 1, S_t, S_t, LMSOLVE_DIAGONAL, StBHX, PETSC_FALSE));
      PetscCall(MatLMVMGramianSolve(B, i, i + 1, S_t, S_t, LMSOLVE_DIAGONAL, YtX, PETSC_FALSE));
      PetscCall(MatLMVMBasisGEMV(B, S_t, i, i + 1, -1.0, StBHX, 1.0, BHX));
      PetscCall(MatLMVMBasisMultAdd(B, S_t, i, i + 1, YtX, BHX));
    }
    PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_t, &YtX));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, S_t, &StBHX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The Broyden kernel can be written as

   $$
     B_k = B_0 + (Y_k - B_0 S_k) (triu(S^T S))^{-1} S_k^T
   $$

   where triu is the upper triangular component.  We solve by back substitution each time
   we apply
 */

PETSC_INTERN PetscErrorCode BroydenKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t           = MatLMVMBasisMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_minus_B0S_t = MatLMVMBasisMap(LMBASIS_Y_MINUS_B0S, mode);
    LMGramian        StS;
    PetscScalar     *StX;

    PetscCall(MatLMVMGetUpdatedGramian(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(MatLMVMBasisGetWorkRow(B, S_t, &StX));
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, S_t, oldest, next, X, NULL, StX));
    PetscCall(LMGramianSolve(StS, oldest, next, LMSOLVE_UPPER_TRIANGLE, StX, PETSC_FALSE));
    PetscCall(MatLMVMBasisMultAdd(B, Y_minus_B0S_t, oldest, next, StX, BX));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, S_t, &StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0HermitianTranspose : MatLMVMApplyJ0InvHermitianTranspose)(B, X, BHX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t           = MatLMVMBasisMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_minus_B0S_t = MatLMVMBasisMap(LMBASIS_Y_MINUS_B0S, mode);
    LMGramian        StS;
    PetscScalar     *YmB0StX;

    PetscCall(MatLMVMGetUpdatedGramian(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(MatLMVMBasisGetWorkRow(B, S_t, &YmB0StX));
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_minus_B0S_t, oldest, next, X, NULL, YmB0StX));
    PetscCall(LMGramianSolve(StS, oldest, next, LMSOLVE_UPPER_TRIANGLE, YmB0StX, PETSC_TRUE));
    PetscCall(MatLMVMBasisMultAdd(B, S_t, oldest, next, YmB0StX, BHX));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, S_t, &YmB0StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBrdn_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBrdn_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBrdn_CompactDense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBrdn_CompactDense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt oldest, next;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Accept the update */
    PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_Brdn *bctx  = (Mat_Brdn *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_Brdn *mctx  = (Mat_Brdn *)mdata->ctx;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < BROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisCopy(bctx->basis[i], mctx->basis[i]));
  for (PetscInt i = 0; i < BROYDEN_GRAMIAN_COUNT; i++) PetscCall(LMGramianCopy(bctx->gramian[i], mctx->gramian[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBrdn_Internal(Mat B)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_Brdn *lbrdn = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < BROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisDestroy(&lbrdn->basis[i]));
  for (PetscInt i = 0; i < BROYDEN_GRAMIAN_COUNT; i++) PetscCall(LMGramianDestroy(&lbrdn->gramian[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMBrdn(Mat B, PetscBool destructive)
{
  PetscFunctionBegin;
  if (destructive) PetscCall(MatDestroy_LMVMBrdn_Internal(B));
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy_LMVMBrdn_Internal(B));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMBrdn(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Broyden method for approximating Jacobian action (MATLMVMBROYDEN)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_matvec_type", "Algorithm used to matrix vector products", "", MatLMVMMatvecTypes, (PetscEnum)lmvm->matvec_type, (PetscEnum *)&lmvm->matvec_type, NULL));
  PetscOptionsHeadEnd();

  switch (lmvm->matvec_type) {
  case MATLMVM_MATVEC_RECURSIVE:
    lmvm->ops->mult    = MatMult_LMVMBrdn_Recursive;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_Recursive;
    lmvm->ops->solve   = MatSolve_LMVMBrdn_Recursive;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_Recursive;
    break;
  case MATLMVM_MATVEC_COMPACT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBrdn_CompactDense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_CompactDense;
    lmvm->ops->solve   = MatSolve_LMVMBrdn_CompactDense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_Recursive;
    break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBrdn(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_Brdn *lbrdn;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBROYDEN));
  B->ops->destroy        = MatDestroy_LMVMBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMBrdn;

  lmvm              = (Mat_LMVM *)B->data;
  lmvm->ops->reset  = MatReset_LMVMBrdn;
  lmvm->ops->update = MatUpdate_LMVMBrdn;
  lmvm->ops->copy   = MatCopy_LMVMBrdn;

  lmvm->matvec_type  = MATLMVM_MATVEC_COMPACT_DENSE;
  lmvm->ops->mult    = MatMult_LMVMBrdn_CompactDense;
  lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_CompactDense;
  lmvm->ops->solve   = MatSolve_LMVMBrdn_CompactDense;
  lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_CompactDense;

  PetscCall(PetscNew(&lbrdn));
  lmvm->ctx = (void *)lbrdn;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMBroyden - Creates a limited-memory "good" Broyden-type approximation
  matrix used for a Jacobian. L-Brdn is not guaranteed to be symmetric or
  positive-definite.

  To use the L-Brdn matrix with other vector types, the matrix must be
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

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
