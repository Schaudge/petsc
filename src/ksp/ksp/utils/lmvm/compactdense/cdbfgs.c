#include <../src/ksp/ksp/utils/lmvm/compactdense/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>
#if defined(PETSC_HAVE_CUDA)
#include <petscdevice_cuda.h>
#include <petsc/private/vecimpl.h>
#include <cuda_profiler_api.h>
#endif

PetscLogEvent CDBFGS_MatMult;
PetscLogEvent CDBFGS_MatSolve;
PetscLogEvent CDBFGS_J0Inv;
PetscLogEvent CDBFGS_J0Fwd;

static inline PetscInt recycle_index(PetscInt m, PetscInt idx)
{
  return idx % m;
}

static inline PetscInt history_index(PetscInt m, PetscInt num_updates, PetscInt idx)
{
  return (idx - num_updates) + PetscMin(m, num_updates);
}

static inline PetscInt oldest_update(PetscInt m, PetscInt idx)
{
  return PetscMax(0, idx - m);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_J0Fwd, B, X, Z, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lbfgs->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM *dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
    Mat_DiagBrdn *diagctx = (Mat_DiagBrdn *) dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecAXPBYAsync_Private(Z, 1.0 / diagctx->sigma, 0.0, X, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecPointwiseDivideAsync_Private(Z, diagctx->invD, X, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopyAsync_Private(X, Z, dctx));
      break;
    }
  }
  PetscCall(PetscLogEventEnd(CDBFGS_J0Fwd, B, X, Z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_J0Inv, B, F, dX, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lbfgs->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM *dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
    Mat_DiagBrdn *diagctx = (Mat_DiagBrdn *) dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecAXPBYAsync_Private(dX, diagctx->sigma, 0.0, F, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecPointwiseMultAsync_Private(dX, diagctx->invD, F, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopyAsync_Private(F, dX, dctx));
      break;
    }
  }
  PetscCall(PetscLogEventEnd(CDBFGS_J0Inv, B, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// This is not Bunch-Kaufman LDLT: here L is strictly lower triangular part of STY
static PetscErrorCode MatGetLDLT(Mat B, Mat result)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt    m_local;

  PetscFunctionBegin;
  if (!lbfgs->temp_mat) PetscCall(MatDuplicate(lbfgs->YtS_triu_strict, MAT_SHARE_NONZERO_PATTERN, &lbfgs->temp_mat));
  PetscCall(MatCopy(lbfgs->YtS_triu_strict, lbfgs->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(lbfgs->temp_mat, lbfgs->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  if (m_local) {
    Mat temp_local, YtS_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
    PetscCall(MatDenseGetLocalMatrix(lbfgs->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(YtS_local, temp_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &result_local));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCDBFGSUpdateMultData(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt    m = lmvm->m, m_local;
  PetscInt    k = lbfgs->num_updates;
  PetscInt    h = k - oldest_update(m, k);
  PetscInt    j_0;
  PetscInt    prev_oldest;
  Mat         J_local;

  PetscFunctionBegin;
  if (!lbfgs->YtS_triu_strict) {
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->YtS_triu_strict));
    PetscCall(MatDestroy(&lbfgs->StBS));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->StBS));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(MatDuplicate(lbfgs->Yfull, MAT_SHARE_NONZERO_PATTERN, &lbfgs->BS));
    PetscCall(MatZeroEntries(lbfgs->YtS_triu_strict));
    PetscCall(MatZeroEntries(lbfgs->BS));
    PetscCall(MatZeroEntries(lbfgs->StBS));
    PetscCall(MatZeroEntries(lbfgs->J));
    PetscCall(MatShift(lbfgs->StBS, 1.0));
    lbfgs->num_mult_updates = oldest_update(m, k);
  }
  if (lbfgs->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  // B_0 may have been updated, we must recompute B_0 S and S^T B_0 S
  // TODO: automatically track when B_0 s_j is stale
  // TODO: matrix-matrix product for S^T B_0 S
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      s_j;
    Vec      Bs_j;
    Vec      StBs_j;
    PetscInt S_idx = recycle_index(m, j);
    PetscInt StBS_idx = lbfgs->strategy == MAT_LMVM_CD_INPLACE ? S_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatCDBFGSApplyJ0Fwd(B, s_j, Bs_j));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, Bs_j, StBs_j, 0, h));
    lbfgs->St_count++;
    if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, StBs_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
  }
  prev_oldest = oldest_update(m, lbfgs->num_mult_updates);
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER && prev_oldest < oldest_update(m, k)) {
    // move the YtS entries that have been computed and need to be kept back up
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, lbfgs->YtS_triu_strict, m_keep, lbfgs->temp_mat));
  }
  PetscCall(MatGetLocalSize(lbfgs->YtS_triu_strict, &m_local, NULL));
  j_0 = PetscMax(lbfgs->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt S_idx   = recycle_index(m, j);
    PetscInt YtS_idx = lbfgs->strategy == MAT_LMVM_CD_INPLACE ? S_idx : history_index(m, k, j);
    Vec      s_j, Yts_j;

    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, s_j, Yts_j, 0, h));
    lbfgs->Yt_count++;
    if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Yts_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    // zero the corresponding row
    if (m_local > 0) {
      Mat YtS_local, YtS_row;

      PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
      PetscCall(MatDenseGetSubMatrix(YtS_local, YtS_idx, YtS_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &YtS_row));
      PetscCall(MatZeroEntries(YtS_row));
      PetscCall(MatDenseRestoreSubMatrix(YtS_local, &YtS_row));
    }
  }
  {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

    if (!lbfgs->inv_diag_vec) PetscCall(VecDuplicate(lbfgs->diag_vec, &lbfgs->inv_diag_vec));
    PetscCall(VecCopyAsync_Private(lbfgs->diag_vec, lbfgs->inv_diag_vec, dctx));
    PetscCall(VecReciprocalAsync_Private(lbfgs->inv_diag_vec, dctx));
  }
  PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetLDLT(B, lbfgs->J));
  PetscCall(MatAXPY(lbfgs->J, 1.0, lbfgs->StBS, SAME_NONZERO_PATTERN));
  if (m_local && lbfgs->mult_type) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  lbfgs->num_mult_updates = lbfgs->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
 * [ I | -S R^{-T} ] [  I  | 0 ] [ H_0 | 0 ] [ I | Y ] [      I      ]
 *                   [-----+---] [-----+---] [---+---] [-------------]
 *                   [ Y^T | I ] [  0  | D ] [ 0 | I ] [ -R^{-1} S^T ]  */

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscDeviceContext dctx;
  Vec rwork1 = lbfgs->rwork1;
  PetscInt m = lmvm->m;
  PetscInt k = lbfgs->num_updates;
  PetscInt h = k - oldest_update(m, k);
  PetscObjectState Fstate;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatSolve, H, F, dX,0));
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (!lbfgs->num_updates) {
    PetscCall(MatCDBFGSApplyJ0Inv(H, F, dX));
    PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(PetscObjectStateGet((PetscObject)F, &Fstate));
  if (F == lbfgs->Fprev_ref && Fstate == lbfgs->Fprev_state) {
    PetscCall(VecCopyAsync_Private(lbfgs->StFprev, rwork1, dctx));
  } else {
    PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, F, rwork1, 0, h));
    lbfgs->St_count++;
  }

  /* Reordering rwork1, as STY is in history order, while S is in recycled order */
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_FALSE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(VecCopyAsync_Private(F, lbfgs->column_work, dctx));
  PetscCall(MatMultAddColumnRange(lbfgs->Yfull, rwork1, lbfgs->column_work, lbfgs->column_work, 0, h));
  lbfgs->Y_count++;

  PetscCall(VecPointwiseMultAsync_Private(rwork1, lbfgs->diag_vec_recycle_order, rwork1, dctx));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->column_work, dX));

  PetscCall(MatMultTransposeAddColumnRange(lbfgs->Yfull, dX, rwork1, rwork1, 0, h));
  lbfgs->Yt_count++;

  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_TRUE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(MatMultAddColumnRange(lbfgs->Sfull, rwork1, dX, dX, 0, h));
  lbfgs->S_count++;
  PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   B_0 - [ Y | B_0 S] [ -D  |    L^T    ]^-1 [   Y^T   ]
                      [-----+-----------]    [---------]
                      [  L  | S^T B_0 S ]    [ S^T B_0 ]

   Above is equivalent to

   B_0 - [ Y | B_0 S] [[     I     | 0 ][ -D  | 0 ][ I | -D^{-1} L^T ]]^-1 [   Y^T   ]
                      [[-----------+---][-----+---][---+-------------]]    [---------]
                      [[ -L D^{-1} | I ][  0  | J ][ 0 |       I     ]]    [ S^T B_0 ]

   where J = S^T B_0 S + L D^{-1} L^T

   becomes

   B_0 - [ Y | B_0 S] [ I | D^{-1} L^T ][ -D^{-1}  |   0    ][    I     | 0 ] [   Y^T   ]
                      [---+------------][----------+--------][----------+---] [---------]
                      [ 0 |     I      ][     0    | J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      =

   B_0 + [ Y | B_0 S] [ D^{-1} | 0 ][ I | L^T ][ I |    0    ][     I    | 0 ] [   Y^T   ]
                      [--------+---][---+-----][---+---------][----------+---] [---------]
                      [ 0      | I ][ 0 |  I  ][ 0 | -J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      (Note that YtS_triu_strict is L^T)
   Byrd, Nocedal, Schnabel 1994

   Alternative approach: considering the fact that DFP is dual to BFGS, use MatMult of DPF:
   (See cddfp.c's MatMult_LMVMCDDFP)

*/

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat                J_local;
  PetscInt           m_local;
  PetscInt           m = lmvm->m;
  PetscInt           k = lbfgs->num_updates;
  PetscInt           h = k - oldest_update(m, k);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatMult, B, X, Z,0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (!lbfgs->num_updates) {
    PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(MatLMVMCDBFGSUpdateMultData(B));
  PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, X, lbfgs->rwork1, 0, h));
  lbfgs->Yt_count++;
  PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, Z, lbfgs->rwork2, 0, h));
  lbfgs->St_count++;
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork2, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  }

  PetscCall(VecPointwiseMultAsync_Private(lbfgs->rwork3, lbfgs->rwork1, lbfgs->inv_diag_vec, dctx));

  PetscCall(MatMultTransposeAdd(lbfgs->YtS_triu_strict, lbfgs->rwork3, lbfgs->rwork2, lbfgs->rwork2));

  if (!lbfgs->rwork2_local) PetscCall(VecCreateLocalVector(lbfgs->rwork2, &lbfgs->rwork2_local));
  if (!lbfgs->rwork3_local) PetscCall(VecCreateLocalVector(lbfgs->rwork3, &lbfgs->rwork3_local));
  PetscCall(VecGetLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
  PetscCall(VecGetLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
  PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
  PetscCall(VecGetSize(lbfgs->rwork2_local, &m_local));
  if (m_local) {
    Mat J_local;

    PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
    PetscCall(MatSolve(J_local, lbfgs->rwork2_local, lbfgs->rwork3_local));
  }
  PetscCall(VecRestoreLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
  PetscCall(VecRestoreLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
  PetscCall(VecScale(lbfgs->rwork3, -1.0));

  PetscCall(MatMultAdd(lbfgs->YtS_triu_strict, lbfgs->rwork3, lbfgs->rwork1, lbfgs->rwork1));

  PetscCall(VecPointwiseMultAsync_Private(lbfgs->rwork1, lbfgs->rwork1, lbfgs->inv_diag_vec, dctx));

  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork3, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  }

  PetscCall(MatMultAddColumnRange(lbfgs->Yfull, lbfgs->rwork1, Z, Z, 0, h));
  lbfgs->Y_count++;
  PetscCall(MatMultAddColumnRange(lbfgs->BS, lbfgs->rwork3, Z, Z, 0, h));
  lbfgs->S_count++;
  PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM          *dbase = (Mat_LMVM *)lbfgs->diag_bfgs->data;
  Mat_DiagBrdn      *diagctx = (Mat_DiagBrdn *)dbase->ctx;

  PetscScalar        curvature, yTy;
  PetscReal          curvtol;
  Vec                workvec1;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (lmvm->prev_set) {
    Vec FX[2];
    Vec XF[2];
    PetscScalar dotFX[2];
    PetscScalar dotXF[2];
    PetscScalar stFprev;
    PetscScalar ytXprev;

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPXAsync_Private(lmvm->Xprev, -1.0, X, dctx));
    /* Test if the updates can be accepted */
    FX[0] = lmvm->Fprev; // dotFX[0] = s^T Fprev
    FX[1] = F;           // dotFX[1] = s^T F
    XF[0] = lmvm->Xprev; // dotXF[0] = y^T Xprev
    XF[1] = X;           // dotXF[1] = y^T X
    PetscCall(VecMDot(lmvm->Xprev, 2, FX, dotFX));
    PetscCall(VecMDot(lmvm->Fprev, 2, XF, dotXF));
    PetscCall(VecAYPXAsync_Private(lmvm->Fprev, -1.0, F, dctx));
    PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
    stFprev = dotFX[0];
    ytXprev = dotXF[0];
    curvature = (dotFX[1] - dotFX[0]); // s^T y
    if (PetscRealPart(yTy) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(yTy);
    }
    if (PetscRealPart(curvature) > curvtol) {
      PetscInt  m = lmvm->m;
      PetscInt  k = lbfgs->num_updates;
      PetscInt  h_new = k + 1 - oldest_update(m, k + 1);
      PetscInt  idx = recycle_index(m, k);
      PetscInt  StYidx;

      /* Update is good, accept it */
      lmvm->nupdates++;
      lbfgs->num_updates++;
      lbfgs->watchdog = 0;

      if (lmvm->k != m-1) {
        lmvm->k++;
      } else if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
        PetscCall(MatMove_LR3(B, lbfgs->StY_triu, m - 1, lbfgs->temp_mat));
      }

      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Sfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Xprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Sfull, idx, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Yfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Fprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Yfull, idx, &workvec1));

      StYidx = (lbfgs->strategy == MAT_LMVM_CD_REORDER) ? history_index(m, lbfgs->num_updates, k) : idx;

      { // implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the
        // H_k is immediately applied to F after begin updated.   The S^T y computation can be split up as S^T (F - F_prev)
        Vec this_sy_col;
        PetscInt local_n;
        PetscScalar *StFprev;
        PetscMemType memtype;

        if (!lbfgs->StFprev) {
          PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->StFprev));
          PetscCall(VecZeroEntries(lbfgs->StFprev));
        }
        PetscCall(VecGetLocalSize(lbfgs->StFprev, &local_n));
        PetscCall(VecGetArrayAndMemType(lbfgs->StFprev, &StFprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            StFprev[idx] = stFprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&stFprev, PETSC_MEMTYPE_HOST, 1 * sizeof(stFprev)));
            PetscCall(PetscDeviceRegisterMemory(StFprev, memtype, local_n * sizeof(*StFprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &StFprev[idx], &stFprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(lbfgs->StFprev, &StFprev));

        // Now StFprev is updated for the new S vector.  Write -StFprev into the appropriate row
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->StY_triu, StYidx, &this_sy_col));
        PetscCall(VecAXPBYAsync_Private(this_sy_col, -1.0, 0.0, lbfgs->StFprev, dctx));

        // Now compute the new StFprev
        PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, F, lbfgs->StFprev, 0, h_new));
        lbfgs->St_count++;

        // Now add StFprev: this_sy_col == S^T (F - Fprev) == S^T y
        PetscCall(VecAXPYAsync_Private(this_sy_col, 1.0, lbfgs->StFprev, dctx));

        if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col, lbfgs->num_updates, lbfgs->cyclic_work_vec));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StY_triu, StYidx, &this_sy_col));
      }

      { // implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the
        // B_k is immediately applied to X after begin updated.   The Y^T x computation can be split up as Y^T (X - X_prev)
        PetscInt local_n;
        PetscScalar *YtXprev;
        PetscMemType memtype;

        if (!lbfgs->YtXprev) {
          PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->YtXprev));
          PetscCall(VecZeroEntries(lbfgs->YtXprev));
        }
        PetscCall(VecGetLocalSize(lbfgs->YtXprev, &local_n));
        PetscCall(VecGetArrayAndMemType(lbfgs->YtXprev, &YtXprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            YtXprev[idx] = ytXprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&ytXprev, PETSC_MEMTYPE_HOST, 1 * sizeof(ytXprev)));
            PetscCall(PetscDeviceRegisterMemory(YtXprev, memtype, local_n * sizeof(*YtXprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &YtXprev[idx], &ytXprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(lbfgs->YtXprev, &YtXprev));

        // Now compute the new YtXprev
        PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, X, lbfgs->YtXprev, 0, h_new));
        lbfgs->Yt_count++;
      }

      PetscCall(MatGetDiagonal(lbfgs->StY_triu, lbfgs->diag_vec));
      if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
        if (!lbfgs->diag_vec_recycle_order) PetscCall(VecDuplicate(lbfgs->diag_vec, &lbfgs->diag_vec_recycle_order));
        PetscCall(VecCopyAsync_Private(lbfgs->diag_vec, lbfgs->diag_vec_recycle_order, dctx));
        PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->diag_vec_recycle_order, lbfgs->num_updates, lbfgs->cyclic_work_vec));
      } else {
        if (!lbfgs->diag_vec_recycle_order) {
          PetscCall(PetscObjectReference((PetscObject)lbfgs->diag_vec));
          lbfgs->diag_vec_recycle_order = lbfgs->diag_vec;
        }
      }

      if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        PetscScalar sTy = curvature;

        PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
        diagctx->sigma = sTy / yTy;
      } else if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
        // Diagonal Barzilai-Borwein after Park et al. TODO
        PetscBool   bb = PETSC_FALSE;
        PetscBool   forward = PETSC_TRUE;
        PetscScalar sTy = curvature;
        PetscReal   mu = 0.25;

        if (!diagctx->invD) {
          PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->invD));
          PetscCall(VecSetAsync_Private(diagctx->invD, sTy / yTy, dctx));
        }
        if (!diagctx->U) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->U));
        if (!diagctx->V) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->V));
        if (!diagctx->W) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->W));

        if (forward) {
          if (bb) {
            PetscScalar sTs;
            PetscCall(VecDot(lmvm->Xprev, lmvm->Xprev, &sTs));

            PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Xprev, lmvm->Xprev, dctx));
            PetscCall(VecShiftAsync_Private(diagctx->U, mu, dctx));
            PetscCall(VecPointwiseMultAsync_Private(diagctx->V, lmvm->Fprev, lmvm->Xprev, dctx));
            PetscCall(VecPointwiseMultAsync_Private(diagctx->V, diagctx->V, diagctx->invD, dctx));
            PetscCall(VecShiftAsync_Private(diagctx->V, mu, dctx));
            PetscCall(VecPointwiseDivideAsync_Private(diagctx->U, diagctx->U, diagctx->V, dctx)); // (s o s + mu) / (s o H o y + mu)

            PetscCall(VecSetAsync_Private(diagctx->V, sTy / yTy, dctx));
            PetscCall(VecPointwiseMinAsync_Private(diagctx->V, diagctx->V, diagctx->invD, dctx)); // lower bound = min(invD, sTy / yTy)
            PetscCall(VecSetAsync_Private(diagctx->W, sTs / sTy, dctx));
            PetscCall(VecPointwiseMaxAsync_Private(diagctx->W, diagctx->W, diagctx->invD, dctx)); // lower bound = max(invD, sTs / sTy)

            PetscCall(VecPointwiseMultAsync_Private(diagctx->invD, diagctx->invD, diagctx->U, dctx));
            PetscCall(VecPointwiseMaxAsync_Private(diagctx->invD, diagctx->invD, diagctx->V, dctx));    // enforce lower bound
            PetscCall(VecPointwiseMinAsync_Private(diagctx->invD, diagctx->invD, diagctx->W, dctx));    // enforce upper bound
          } else {
            PetscScalar sTDs, yTDy;

            // diagonal Broyden
            PetscCall(VecReciprocalAsync_Private(diagctx->invD, dctx));
            PetscCall(VecPointwiseMultAsync_Private(diagctx->V, diagctx->invD, lmvm->Xprev, dctx));
            PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Fprev, dctx));
            PetscCall(VecAXPYAsync_Private(diagctx->invD, 1.0 / sTy, diagctx->U, dctx));
            PetscCall(VecDot(diagctx->V, lmvm->Xprev, &sTDs));
            PetscCall(VecPointwiseMultAsync_Private(diagctx->V, diagctx->V, diagctx->V, dctx));
            PetscCall(VecAXPYAsync_Private(diagctx->invD, -1.0 / PetscMax(sTDs, diagctx->tol), diagctx->V, dctx));
            PetscCall(VecReciprocalAsync_Private(diagctx->invD, dctx));
            PetscCall(VecAbsAsync_Private(diagctx->invD, dctx));
            PetscCall(VecDot(diagctx->U, diagctx->invD, &yTDy));
            PetscCall(VecScaleAsync_Private(diagctx->invD, sTy / yTDy, dctx));
          }
        } else {
          PetscScalar sTs;
          PetscCall(VecDot(lmvm->Xprev, lmvm->Xprev, &sTs));

          PetscCall(VecSetAsync_Private(diagctx->V, sTy / yTy, dctx));
          PetscCall(VecPointwiseMinAsync_Private(diagctx->V, diagctx->V, diagctx->invD, dctx)); // lower bound = min(invD, sTy / yTy)
          PetscCall(VecSetAsync_Private(diagctx->W, sTs / sTy, dctx));
          PetscCall(VecPointwiseMaxAsync_Private(diagctx->W, diagctx->W, diagctx->invD, dctx)); // lower bound = max(invD, sTs / sTy)

          PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Xprev, dctx));
          PetscCall(VecAYPXAsync_Private(diagctx->invD, mu, diagctx->U, dctx));
          PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Fprev, dctx));
          PetscCall(VecShiftAsync_Private(diagctx->U, mu, dctx));
          PetscCall(VecPointwiseDivideAsync_Private(diagctx->invD, diagctx->invD, diagctx->U, dctx)); // (s o y + mu invD) / (y o y + mu)
          PetscCall(VecPointwiseMaxAsync_Private(diagctx->invD, diagctx->invD, diagctx->V, dctx));    // enforce lower bound
          PetscCall(VecPointwiseMinAsync_Private(diagctx->invD, diagctx->invD, diagctx->W, dctx));    // enforce upper bound
        }

      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
      lmvm->k = lmvm->k - 1;
      PetscInt  m = lmvm->m;
      PetscInt  k = lbfgs->num_updates;
      PetscInt  h = k - oldest_update(m, k);

      // we still have to maintain StFprev
      if (!lbfgs->StFprev) {
        PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->StFprev));
        PetscCall(VecZeroEntries(lbfgs->StFprev));
      }
      PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, F, lbfgs->StFprev, 0, h));
      lbfgs->St_count++;
      // we still have to maintain YtXprev
      if (!lbfgs->YtXprev) {
        PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->YtXprev));
        PetscCall(VecZeroEntries(lbfgs->YtXprev));
      }
      PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, X, lbfgs->YtXprev, 0, h));
      lbfgs->Yt_count++;
    }
  } else {
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecSetAsync_Private(diagctx->invD, diagctx->delta, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      diagctx->sigma = diagctx->delta;
      break;
    default:
      diagctx->sigma = 1.0;
      break;
    }
  }

  if (lbfgs->watchdog > lbfgs->max_seq_rejects) PetscCall(MatLMVMReset(B, PETSC_FALSE));

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopyAsync_Private(X, lmvm->Xprev, dctx));
  PetscCall(VecCopyAsync_Private(F, lmvm->Fprev, dctx));
  PetscCall(PetscObjectReference((PetscObject)F));
  PetscCall(VecDestroy(&lbfgs->Fprev_ref));
  lbfgs->Fprev_ref = F;
  PetscCall(PetscObjectStateGet((PetscObject)F, &lbfgs->Fprev_state));

  PetscCall(PetscObjectReference((PetscObject)X));
  PetscCall(VecDestroy(&lbfgs->Xprev_ref));
  lbfgs->Xprev_ref = X;
  PetscCall(PetscObjectStateGet((PetscObject)X, &lbfgs->Xprev_state));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMCDBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM   *bdata  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *blbfgs = (Mat_CDBFGS*)bdata->ctx;
  Mat_LMVM   *mdata  = (Mat_LMVM*)M->data;
  Mat_CDBFGS *mlbfgs = (Mat_CDBFGS*)mdata->ctx;

  PetscFunctionBegin;
  mlbfgs->watchdog        = blbfgs->watchdog;
  mlbfgs->max_seq_rejects = blbfgs->max_seq_rejects;
  if (!(bdata->J0 || bdata->user_pc || bdata->user_ksp || bdata->user_scale)) {
    PetscCall(MatCopy(blbfgs->diag_bfgs, mlbfgs->diag_bfgs, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatLMVMCDBFGSResetDestructive(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lbfgs->Sfull));
  PetscCall(MatDestroy(&lbfgs->Yfull));
  PetscCall(MatDestroy(&lbfgs->BS));
  PetscCall(MatDestroy(&lbfgs->StY_triu));
  PetscCall(VecDestroy(&lbfgs->StFprev));
  PetscCall(VecDestroy(&lbfgs->YtXprev));
  PetscCall(VecDestroy(&lbfgs->Fprev_ref));
  PetscCall(VecDestroy(&lbfgs->Xprev_ref));
  lbfgs->Fprev_state = 0;
  lbfgs->Xprev_state = 0;
  PetscCall(MatDestroy(&lbfgs->YtS_triu_strict));
  PetscCall(MatDestroy(&lbfgs->LDLt));
  PetscCall(MatDestroy(&lbfgs->StBS));
  PetscCall(MatDestroy(&lbfgs->J));
  PetscCall(MatDestroy(&lbfgs->temp_mat));
  PetscCall(VecDestroy(&lbfgs->diag_vec));
  PetscCall(VecDestroy(&lbfgs->diag_vec_recycle_order));
  PetscCall(VecDestroy(&lbfgs->inv_diag_vec));
  PetscCall(VecDestroy(&lbfgs->column_work));
  PetscCall(VecDestroy(&lbfgs->rwork1));
  PetscCall(VecDestroy(&lbfgs->rwork2));
  PetscCall(VecDestroy(&lbfgs->rwork3));
  PetscCall(VecDestroy(&lbfgs->rwork2_local));
  PetscCall(VecDestroy(&lbfgs->rwork3_local));
  PetscCall(VecDestroy(&lbfgs->cyclic_work_vec));
  lbfgs->allocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMCDBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMReset(lbfgs->diag_bfgs, destructive));
  if (lbfgs->Sfull) PetscCall(MatZeroEntries(lbfgs->Sfull));
  if (lbfgs->Yfull) PetscCall(MatZeroEntries(lbfgs->Yfull));
  if (lbfgs->BS) PetscCall(MatZeroEntries(lbfgs->BS));
  if (lbfgs->StY_triu) { // Set to identity by default so it is invertible
    PetscCall(MatZeroEntries(lbfgs->StY_triu));
    PetscCall(MatShift(lbfgs->StY_triu, 1.0));
  }
  if (lbfgs->YtS_triu_strict) PetscCall(MatZeroEntries(lbfgs->YtS_triu_strict));
  if (lbfgs->LDLt) PetscCall(MatZeroEntries(lbfgs->LDLt));
  if (lbfgs->StBS) {
    PetscCall(MatZeroEntries(lbfgs->StBS));
    PetscCall(MatShift(lbfgs->StBS, 1.0));
  }
  if (lbfgs->Fprev_ref) PetscCall(VecDestroy(&lbfgs->Fprev_ref));
  if (lbfgs->Xprev_ref) PetscCall(VecDestroy(&lbfgs->Xprev_ref));
  lbfgs->Fprev_state = 0;
  lbfgs->Xprev_state = 0;
  if (lbfgs->StFprev) PetscCall(VecZeroEntries(lbfgs->StFprev));
  if (lbfgs->YtXprev) PetscCall(VecZeroEntries(lbfgs->YtXprev));
  if (destructive) {
    PetscCall(MatLMVMCDBFGSResetDestructive(B));
  }
  lbfgs->num_updates = 0;
  lbfgs->num_mult_updates = 0;
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscBool  same, allocate = PETSC_FALSE;
  VecType    vec_type;
  PetscInt   m, n, M, N;
  MPI_Comm   comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  if (lmvm->allocated) {
    PetscCall(VecGetType(X, &vec_type));
    PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->Xprev, vec_type, &same));
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      PetscCall(MatLMVMReset(B, PETSC_TRUE));
    } else {
      VecCheckMatCompatible(B, X, 2, F, 3);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    PetscCall(VecGetLocalSize(X, &n));
    PetscCall(VecGetSize(X, &N));
    PetscCall(VecGetLocalSize(F, &m));
    PetscCall(VecGetSize(F, &M));
    if (N != M) SETERRQ(comm, PETSC_ERR_ARG_SIZ, "Incorrect problem sizes! dim(X) not equal to dim(F)");
    PetscCall(MatSetSizes(B, m, n, M, N));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    PetscCall(VecDuplicate(X, &lmvm->Xprev));
    PetscCall(VecDuplicate(F, &lmvm->Fprev));
    if (lmvm->m > 0) {
      PetscMPIInt rank;
      PetscInt m, M;

      PetscCall(MPI_Comm_rank(comm, &rank));
      M   = lmvm->m;
      m   = (rank == 0) ? M : 0;

      // Create data needed for MatSolve() eagerly; data needed for MatMult() will be created on demand
      PetscCall(VecGetType(X, &vec_type));
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, n, m, N, M, -1, NULL, &lbfgs->Sfull));
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lbfgs->StY_triu));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_SHARE_NONZERO_PATTERN, &lbfgs->Yfull));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_SHARE_NONZERO_PATTERN, &lbfgs->BS));
      PetscCall(MatZeroEntries(lbfgs->Sfull));
      PetscCall(MatZeroEntries(lbfgs->Yfull));
      // initialize StY_triu to identity so it is invertible
      PetscCall(MatZeroEntries(lbfgs->StY_triu));
      PetscCall(MatShift(lbfgs->StY_triu, 1.0));
      PetscCall(MatCreateVecs(lbfgs->StY_triu, &lbfgs->diag_vec, &lbfgs->rwork1));
      PetscCall(MatCreateVecs(lbfgs->StY_triu, &lbfgs->rwork2, &lbfgs->rwork3));
      PetscCall(VecDuplicate(lbfgs->rwork2, &lbfgs->cyclic_work_vec));
      PetscCall(VecZeroEntries(lbfgs->rwork1));
      PetscCall(VecZeroEntries(lbfgs->rwork2));
      PetscCall(VecZeroEntries(lbfgs->rwork3));
      PetscCall(VecZeroEntries(lbfgs->diag_vec));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->column_work));
    //TODO hacky way to turn diagbrdn lmvm off...
    //PetscCall(MatLMVMSetJ0Scale(B,1.));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMAllocate(lbfgs->diag_bfgs, X, F));
    }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDBFGS(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatLMVMCDBFGSResetDestructive(B));
  PetscCall(MatDestroy(&lbfgs->diag_bfgs));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMCDBFGS(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM*)B->data;

  PetscInt    m, n, M, N;
  PetscMPIInt size;
  MPI_Comm    comm = PetscObjectComm((PetscObject)B);
  Vec         Xtmp, Ftmp;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(MatGetSize(B, &M, &N));
  if (M == 0 && N == 0) SETERRQ(comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    // MatCreateVecs() ???
    PetscCall(MPI_Comm_size(comm, &size));
    if (size == 1) {
      PetscCall(VecCreateSeq(comm, N, &Xtmp));
      PetscCall(VecCreateSeq(comm, M, &Ftmp));
    } else {
      PetscCall(MatGetLocalSize(B, &m, &n));
      PetscCall(VecCreateMPI(comm, n, N, &Xtmp));
      PetscCall(VecCreateMPI(comm, m, M, &Ftmp));
    }
    PetscCall(MatAllocate_LMVMCDBFGS(B, Xtmp, Ftmp));
    PetscCall(VecDestroy(&Xtmp));
    PetscCall(VecDestroy(&Ftmp));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMCDBFGS(Mat B, PetscViewer pv)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  PetscCall(MatView_LMVM(B, pv));
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatView(lbfgs->diag_bfgs, pv));
  }
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "Counts: S x : %" PetscInt_FMT ", S^T x : %" PetscInt_FMT ", Y x : %" PetscInt_FMT ",  Y^T x: %" PetscInt_FMT "\n", lbfgs->S_count, lbfgs->St_count, lbfgs->Y_count, lbfgs->Yt_count));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMCDBFGS(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix,  "Compact dense BFGS method (MATLMVMCDBFGS)", NULL);
  PetscCall(PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLMVMCompactDenseType", MatLMVMCompactDenseTypes, (PetscEnum)lbfgs->strategy, (PetscEnum *)&lbfgs->strategy, NULL));
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBrdnScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)lbfgs->scale_type, (PetscEnum *)&lbfgs->scale_type, NULL));
  PetscCall(PetscOptionsBool("-mat_lbfgs_mult_type", "True for Cholesky type MatMult_CDBFGS, False for DFP type..", "", lbfgs->mult_type, &lbfgs->mult_type, NULL));
  lbfgs->allocated       = PETSC_FALSE;
  PetscOptionsEnd();
  if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    const char *prefix;

    PetscCall(MatGetOptionsPrefix(B, &prefix));
    PetscCall(MatSetOptionsPrefix(lbfgs->diag_bfgs, prefix));
    PetscCall(MatAppendOptionsPrefix(lbfgs->diag_bfgs, "J0_"));
    PetscCall(MatSetFromOptions(lbfgs->diag_bfgs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDBFGS(Mat B)
{
  Mat_LMVM   *lmvm;
  Mat_CDBFGS *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDBFGS));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view = MatView_LMVMCDBFGS;
  B->ops->setup = MatSetUp_LMVMCDBFGS;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDBFGS;
  B->ops->destroy = MatDestroy_LMVMCDBFGS;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDBFGS;
  lmvm->ops->reset = MatReset_LMVMCDBFGS;
  lmvm->ops->update = MatUpdate_LMVMCDBFGS;
  lmvm->ops->mult = MatMult_LMVMCDBFGS;
  lmvm->ops->solve = MatSolve_LMVMCDBFGS;
  lmvm->ops->copy = MatCopy_LMVMCDBFGS;

  PetscCall(PetscNew(&lbfgs));
  lmvm->ctx = (void*)lbfgs;
  lbfgs->allocated       = PETSC_FALSE;
  lbfgs->mult_type       = PETSC_TRUE;
  lbfgs->watchdog        = 0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  lbfgs->strategy        = MAT_LMVM_CD_INPLACE;
  lbfgs->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &lbfgs->diag_bfgs));
  PetscCall(MatSetType(lbfgs->diag_bfgs, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(lbfgs->diag_bfgs, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMCDBFGS - Creates a compact dense representation of the limited-memory
   Broyden-Fletcher-Goldfarb-Shanno (BFGS) approximation to a Hessian. This compact
   dense representation reduces the L-BFGS update to a series of matrix-vector products
   with compact dense matrices in lieu of the conventional matrix-free two-loop
   algorithm. For most problems on CPUs, this compact dense representation is not as
   fast as the matrix-free two-loop implementation provided via MATLMVMBFGS. However,
   it may be faster on GPUs for large enough problems (note: requires CUDA/HIP/KOKKOS).

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Level: advanced

.seealso: MatCreate(), MATLMVM, MATLMVMCDBFGS, MatCreateLMVMBFGS()
@*/
PetscErrorCode MatCreateLMVMCDBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
