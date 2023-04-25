#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric-Rank-1 method for approximating both
  the forward product and inverse application of a Jacobian.
*/

typedef enum {
  SR1_BASIS_Y_MINUS_BKS = 0,
  SR1_BASIS_S_MINUS_HKS = 1,
  SR1_BASIS_COUNT
} SR1BasisType;

typedef enum {
  SR1_GRAMIAN_YTS_MINUS_STB0S = 0,
  SR1_GRAMIAN_STY_MINUS_YTH0Y = 1,
  SR1_GRAMIAN_YTS_MINUS_STBKS = 2,
  SR1_GRAMIAN_STY_MINUS_YTHKY = 3,
  SR1_GRAMIAN_COUNT
} SR1GramianType;

static inline SR1BasisType SR1BasisMap(SR1BasisType type, MatLMVMMode mode)
{
  return type ^ mode;
}

static inline SR1GramianType SR1GramianMap(SR1GramianType type, MatLMVMMode mode)
{
  return type ^ mode;
}

typedef struct {
  LMBasis   basis[SR1_BASIS_COUNT];
  LMGramian gramian[SR1_GRAMIAN_COUNT];
} Mat_LSR1;

static PetscErrorCode SR1CompactGramianUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM      *lmvm              = (Mat_LMVM *)B->data;
  Mat_LSR1      *lsr1              = (Mat_LSR1 *)lmvm->ctx;
  SR1GramianType YtS_minus_StB0S_t = SR1GramianMap(SR1_GRAMIAN_YTS_MINUS_STB0S, mode);
  LMGramian      YtS_minus_StB0S;
  PetscInt       oldest, next;

  PetscFunctionBegin;
  if (!lsr1->gramian[YtS_minus_StB0S_t]) PetscCall(LMGramianCreate(lmvm->m, &lsr1->gramian[YtS_minus_StB0S_t]));
  YtS_minus_StB0S = lsr1->gramian[YtS_minus_StB0S_t];

  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (YtS_minus_StB0S->k < next) {
    MatLMVMBasisType S_t   = MatLMVMBasisMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_t   = MatLMVMBasisMap(LMBASIS_Y, mode);
    MatLMVMBasisType B0S_t = MatLMVMBasisMap(LMBASIS_B0S, mode);
    LMGramian        StB0S, YtS;

    PetscCall(MatLMVMGetUpdatedGramian(B, Y_t, S_t, LMBLOCK_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedGramian(B, S_t, B0S_t, LMBLOCK_UPPER_TRIANGLE, &StB0S));
    PetscCall(LMGramianCopy(YtS, YtS_minus_StB0S));
    PetscCall(LMGramianAXPY(YtS_minus_StB0S, LMBLOCK_UPPER_TRIANGLE, -1.0, StB0S, LMBLOCK_UPPER_TRIANGLE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Kernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM        *lmvm              = (Mat_LMVM *)B->data;
    Mat_LSR1        *lsr1              = (Mat_LSR1 *)lmvm->ctx;
    MatLMVMBasisType Y_minus_B0S_t     = MatLMVMBasisMap(LMBASIS_Y_MINUS_B0S, mode);
    SR1GramianType   YtS_minus_StB0S_t = SR1GramianMap(SR1_GRAMIAN_YTS_MINUS_STB0S, mode);
    LMGramian        YtS_minus_StB0S;
    PetscScalar     *YmB0StX;

    PetscCall(SR1CompactGramianUpdate(B, mode));
    YtS_minus_StB0S = lsr1->gramian[YtS_minus_StB0S_t];
    PetscCall(MatLMVMBasisGetWorkRow(B, Y_minus_B0S_t, &YmB0StX));
    PetscCall(MatLMVMBasisMultHermitianTranspose(B, Y_minus_B0S_t, oldest, next, X, BX, YmB0StX));
    PetscCall(LMGramianSolve(YtS_minus_StB0S, oldest, next, LMSOLVE_HERMITIAN_INDEFINITE_UPPER, YmB0StX, PETSC_FALSE));
    PetscCall(MatLMVMBasisMultAdd(B, Y_minus_B0S_t, oldest, next, YmB0StX, BX));
    PetscCall(MatLMVMBasisRestoreWorkRow(B, Y_minus_B0S_t, &YmB0StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Kernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X, Vec BX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;
  ;
  SR1BasisType   Y_minus_BkS_t     = SR1BasisMap(SR1_BASIS_Y_MINUS_BKS, mode);
  SR1GramianType YtS_minus_StBkS_t = SR1GramianMap(SR1_GRAMIAN_STY_MINUS_YTHKY, mode);
  LMBasis        Y_minus_BkS       = lsr1->basis[Y_minus_BkS_t];
  LMGramian      YtS_minus_StBkS   = lsr1->gramian[YtS_minus_StBkS_t];
  PetscScalar   *YmBkStX;

  PetscFunctionBegin;
  PetscCall(LMBasisGetWorkRow(Y_minus_BkS, &YmBkStX));
  PetscCall(LMBasisGEMVH(1.0, Y_minus_BkS, oldest, next, X, 0.0, YmBkStX));
  PetscCall(LMGramianSolve(YtS_minus_StBkS, oldest, next, LMSOLVE_DIAGONAL, YmBkStX, PETSC_FALSE));
  PetscCall(LMBasisGEMV(1.0, Y_minus_BkS, oldest, next, YmBkStX, 1.0, BX));
  PetscCall(LMBasisRestoreWorkRow(Y_minus_BkS, &YmBkStX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1RecursiveBasisUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;
  ;
  MatLMVMBasisType B0S_t             = MatLMVMBasisMap(LMBASIS_B0S, mode);
  MatLMVMBasisType S_t               = MatLMVMBasisMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t               = MatLMVMBasisMap(LMBASIS_Y, mode);
  SR1BasisType     Y_minus_BkS_t     = SR1BasisMap(SR1_BASIS_Y_MINUS_BKS, mode);
  SR1GramianType   YtS_minus_StBkS_t = SR1GramianMap(SR1_GRAMIAN_STY_MINUS_YTHKY, mode);
  LMBasis          Y_minus_BkS;
  LMGramian        YtS_minus_StBkS;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (!lsr1->basis[Y_minus_BkS_t]) PetscCall(LMBasisCreate(mode == MATLMVM_MODE_PRIMAL ? lmvm->Fprev : lmvm->Xprev, lmvm->m, &lsr1->basis[Y_minus_BkS_t]));
  Y_minus_BkS = lsr1->basis[Y_minus_BkS_t];
  if (!lsr1->gramian[YtS_minus_StBkS_t]) PetscCall(LMGramianCreate(lmvm->m, &lsr1->gramian[YtS_minus_StBkS_t]));
  YtS_minus_StBkS = lsr1->gramian[YtS_minus_StBkS_t];
  if (Y_minus_BkS->k < next) {
    LMBasis S;

    Y_minus_BkS->k = next;
    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S));
    PetscCall(LMGramianReset(YtS_minus_StBkS));
    PetscCall(LMGramianUpdateNextIndex(YtS_minus_StBkS, next));
    for (PetscInt j = oldest; j < next; j++) {
      Vec s_j, B0s_j, p_j, y_j;

      PetscCall(LMBasisGetVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));
      PetscCall(MatLMVMGetVecsRead(B, j, B0S_t, &B0s_j, S_t, &s_j, Y_t, &y_j));
      PetscCall(VecCopy(B0s_j, p_j));
      PetscCall(SR1Kernel_Recursive_Inner(B, mode, oldest, j, s_j, p_j));
      PetscCall(VecAYPX(p_j, -1.0, y_j));
      PetscCall(MatLMVMRestoreVecsRead(B, j, B0S_t, &B0s_j, S_t, &s_j, Y_t, &y_j));
      PetscCall(LMBasisRestoreVec(Y_minus_BkS, j, PETSC_MEMORY_ACCESS_WRITE, &p_j));

      PetscCall(LMGramianForceUpdateBlock(YtS_minus_StBkS, Y_minus_BkS, S, LMBLOCK_DIAGONAL, j, j + 1));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Kernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall((mode == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(SR1RecursiveBasisUpdate(B, mode));
    PetscCall(SR1Kernel_Recursive_Inner(B, mode, oldest, next, X, BX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMSR1_CompactDense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSR1_CompactDense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_CompactDense(B, MATLMVM_MODE_DUAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSR1_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSR1_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    PetscReal   snorm, pnorm;
    PetscScalar sktw;
    Vec         work;

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    PetscCall(MatLMVMBasisGetWorkVec(B, LMBASIS_Y, &work));
    /* See if the updates can be accepted
       NOTE: This tests abs(S[k]^T (Y[k] - B_k*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_k*S[k]) */
    PetscCall(MatMult(B, lmvm->Xprev, work));
    PetscCall(VecAYPX(work, -1.0, lmvm->Fprev));
    PetscCall(VecDot(lmvm->Xprev, work, &sktw));
    PetscCall(VecNorm(lmvm->Xprev, NORM_2, &snorm));
    PetscCall(VecNorm(work, NORM_2, &pnorm));
    PetscCall(MatLMVMBasisRestoreWorkVec(B, LMBASIS_Y, &work));
    if (PetscAbsReal(PetscRealPart(sktw)) >= lmvm->eps * snorm * pnorm) {
      /* Update is good, accept it */
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMSR1(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_LSR1 *bctx  = (Mat_LSR1 *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_LSR1 *mctx  = (Mat_LSR1 *)mdata->ctx;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < SR1_BASIS_COUNT; i++) PetscCall(LMBasisCopy(bctx->basis[i], mctx->basis[i]));
  for (PetscInt i = 0; i < SR1_GRAMIAN_COUNT; i++) PetscCall(LMGramianCopy(bctx->gramian[i], mctx->gramian[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSR1_Internal(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < SR1_BASIS_COUNT; i++) PetscCall(LMBasisDestroy(&lsr1->basis[i]));
  for (PetscInt i = 0; i < SR1_GRAMIAN_COUNT; i++) PetscCall(LMGramianDestroy(&lsr1->gramian[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMSR1(Mat B, PetscBool destructive)
{
  PetscFunctionBegin;
  if (destructive) PetscCall(MatReset_LMVMSR1_Internal(B));
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatReset_LMVMSR1_Internal(B));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMSR1(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Symmetric Rank 1 method for approximating Hessian action (MATLMVMSR1)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_matvec_type", "Algorithm used to matrix vector products", "", MatLMVMMatvecTypes, (PetscEnum)lmvm->matvec_type, (PetscEnum *)&lmvm->matvec_type, NULL));
  PetscOptionsHeadEnd();

  switch (lmvm->matvec_type) {
  case MATLMVM_MATVEC_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMSR1_Recursive;
    lmvm->ops->solve = MatSolve_LMVMSR1_Recursive;
    break;
  case MATLMVM_MATVEC_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMSR1_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMSR1_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_LSR1 *lsr1;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSR1));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  B->ops->destroy        = MatDestroy_LMVMSR1;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSR1;

  lmvm               = (Mat_LMVM *)B->data;
  lmvm->square       = PETSC_TRUE;
  lmvm->ops->reset   = MatReset_LMVMSR1;
  lmvm->ops->update  = MatUpdate_LMVMSR1;
  lmvm->ops->mult    = MatMult_LMVMSR1_CompactDense;
  lmvm->ops->multht  = MatMult_LMVMSR1_CompactDense;
  lmvm->ops->solve   = MatSolve_LMVMSR1_CompactDense;
  lmvm->ops->solveht = MatSolve_LMVMSR1_CompactDense;
  lmvm->ops->copy    = MatCopy_LMVMSR1;

  PetscCall(PetscNew(&lsr1));
  lmvm->ctx = (void *)lsr1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMSR1 - Creates a limited-memory Symmetric-Rank-1 approximation
  matrix used for a Jacobian. L-SR1 is symmetric by construction, but is not
  guaranteed to be positive-definite.

  To use the L-SR1 matrix with other vector types, the matrix must be
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMSR1`, `MatCreateLMVMBFGS()`, `MatCreateLMVMDFP()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMSR1(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSR1));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
