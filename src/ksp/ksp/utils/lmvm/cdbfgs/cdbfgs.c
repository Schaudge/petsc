#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
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

const char *const MatLBFGSTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLBFGSType", "MAT_LBFGS_", NULL};

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
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscCall(MatMult(lbfgs->diag_bfgs, X, Z));
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
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscCall(MatSolve(lbfgs->diag_bfgs, F, dX));
  }
  PetscCall(PetscLogEventEnd(CDBFGS_J0Inv, B, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// shift a vector so that whatever is at index d becomes index 0
static PetscErrorCode VecCyclicShift(Mat B, Vec X, PetscInt d)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     n;
  const PetscScalar *src;
  PetscScalar *dest;
  PetscMemType src_memtype;
  PetscMemType dest_memtype;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  if (!lbfgs->cyclic_work_vec) PetscCall(VecDuplicate(X, &lbfgs->cyclic_work_vec));
  PetscCall(VecCopy(X, lbfgs->cyclic_work_vec));
  PetscCall(VecGetArrayReadAndMemType(lbfgs->cyclic_work_vec, &src, &src_memtype));
  PetscCall(VecGetArrayWriteAndMemType(X, &dest, &dest_memtype));
  if (n == 0) { // no work on this process
    PetscCall(VecRestoreArrayWriteAndMemType(X, &dest));
    PetscCall(VecRestoreArrayReadAndMemType(lbfgs->cyclic_work_vec, &src));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssert(src_memtype == dest_memtype, PETSC_COMM_SELF, PETSC_ERR_PLIB, "memtype of duplicate does not match");
  if (PetscMemTypeHost(src_memtype)) {
    PetscCall(PetscArraycpy(dest, &src[d], m - d));
    PetscCall(PetscArraycpy(&dest[m-d], src, d));
  } else {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceRegisterMemory(dest, dest_memtype, m * sizeof(*dest)));
    PetscCall(PetscDeviceRegisterMemory(src, src_memtype, m * sizeof(*src)));
    PetscCall(PetscDeviceArrayCopy(dctx, dest, &src[d], m - d));
    PetscCall(PetscDeviceArrayCopy(dctx, &dest[m - d], src, d));
  }
  PetscCall(VecRestoreArrayWriteAndMemType(X, &dest));
  PetscCall(VecRestoreArrayReadAndMemType(lbfgs->cyclic_work_vec, &src));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRecycleOrderToHistoryOrder(Mat B, Vec X)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = lbfgs->num_updates;
  PetscInt     oldest_index;

  PetscFunctionBegin;
  oldest_index = recycle_index(m, oldest_update(m, k));
  if (oldest_index == 0) PetscFunctionReturn(PETSC_SUCCESS); // vector is already in history order
  PetscCall(VecCyclicShift(B, X, oldest_index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecHistoryOrderToRecycleOrder(Mat B, Vec X)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = lbfgs->num_updates;
  PetscInt     oldest_index;

  PetscFunctionBegin;
  oldest_index = recycle_index(m, oldest_update(m, k));
  if (oldest_index == 0) PetscFunctionReturn(PETSC_SUCCESS); // vector is already in recycle order
  PetscCall(VecCyclicShift(B, X, m - oldest_index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpperTriangularSolveInPlace_Internal(MatLBFGSType lbfgs_type, PetscMemType memtype, PetscBool hermitian_transpose, PetscInt N, PetscInt oldest_index, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
{
  PetscFunctionBegin;
  // if oldest_index == 0, the two strategies are equivalent, redirect to the simpler one
  if (oldest_index == 0) lbfgs_type = MAT_LBFGS_CD_REORDER;
  switch (lbfgs_type) {
  case MAT_LBFGS_CD_REORDER:
    if (PetscMemTypeHost(memtype)) {
      PetscBLASInt n, lda_blas, one = 1;
      PetscCall(PetscBLASIntCast(N, &n));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCallBLAS("BLAStrsv", BLAStrsv_("U", hermitian_transpose ? "C" : "N", "NotUnitTriangular", &n, A, &lda_blas, x, &one));
      PetscCall(PetscLogFlops(1.0 * n * n));
    } else if (PetscMemTypeDevice(memtype)) {
      PetscCall(MatUpperTriangularSolveInPlace_CUPM(hermitian_transpose, N, A, lda, x, 1));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported memtype");
    break;
  case MAT_LBFGS_CD_INPLACE:
    if (PetscMemTypeHost(memtype)) {
      PetscBLASInt n_old, n_new, lda_blas, one = 1;
      PetscScalar  minus_one = -1.0;
      PetscScalar  sone = 1.0;
      PetscCall(PetscBLASIntCast(N - oldest_index, &n_old));
      PetscCall(PetscBLASIntCast(oldest_index, &n_new));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      if (!hermitian_transpose) {
        PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "NotUnitTriangular", &n_new, A, &lda_blas, x, &one));
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &n_old, &n_new, &minus_one, &A[oldest_index], &lda_blas, x, &one, &sone, &x[oldest_index], &one));
        PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "NotUnitTriangular", &n_old, &A[oldest_index * (lda + 1)], &lda_blas, &x[oldest_index], &one));
      } else {
        PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "NotUnitTriangular", &n_old, &A[oldest_index * (lda + 1)], &lda_blas, &x[oldest_index], &one));
        PetscCallBLAS("BLASgemv", BLASgemv_("C", &n_old, &n_new, &minus_one, &A[oldest_index], &lda_blas, &x[oldest_index], &one, &sone, x, &one));
        PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "NotUnitTriangular", &n_new, A, &lda_blas, x, &one));
      }
      PetscCall(PetscLogFlops(1.0 * N * N));
    } else if (PetscMemTypeDevice(memtype)) {
      PetscCall(MatUpperTriangularSolveInPlaceCyclic_CUPM(hermitian_transpose, N, oldest_index, A, lda, x, stride));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported memtype");
    break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpperTriangularSolveInPlace(Mat B, Mat Amat, Vec X, PetscBool hermitian_transpose)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = lbfgs->num_updates;
  PetscInt     h, local_n;
  PetscInt     oldest_index;
  PetscInt     lda;
  PetscScalar *x;
  PetscMemType memtype_r, memtype_x;
  const PetscScalar *A;

  PetscFunctionBegin;
  h = k - oldest_update(m, k);
  if (!h) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetLocalSize(X, &local_n));
  PetscCall(VecGetArrayAndMemType(X, &x, &memtype_x));
  PetscCall(MatDenseGetArrayReadAndMemType(Amat, &A, &memtype_r));
  if (!local_n) {
    PetscCall(MatDenseRestoreArrayReadAndMemType(Amat, &A));
    PetscCall(VecRestoreArrayAndMemType(X, &x));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssert(memtype_x == memtype_r, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incompatible device pointers");
  PetscCall(MatDenseGetLDA(lbfgs->StY_triu, &lda));
  oldest_index = recycle_index(m, oldest_update(m, k));
  PetscCall(MatUpperTriangularSolveInPlace_Internal(lbfgs->strategy, memtype_x, hermitian_transpose, h, oldest_index, A, lda, x, 1));
  PetscCall(VecRestoreArrayWriteAndMemType(X, &x));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Amat, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts R[end-m_keep:end,end-m_keep:end] to R[0:m_keep, 0:m_keep] */

static PetscErrorCode MatMove_LR3(Mat B, Mat R, PetscInt m_keep)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscInt     M;
  Mat          mat_local, local_sub, local_temp, temp_sub;

  PetscFunctionBegin;
  if (!lbfgs->temp_mat) PetscCall(MatDuplicate(R, MAT_SHARE_NONZERO_PATTERN, &lbfgs->temp_mat));
  PetscCall(MatGetLocalSize(R, &M, NULL));
  if (M == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatDenseGetLocalMatrix(R, &mat_local));
  PetscCall(MatDenseGetLocalMatrix(lbfgs->temp_mat, &local_temp));
  PetscCall(MatDenseGetSubMatrix(mat_local, lmvm->m - m_keep, lmvm->m, lmvm->m - m_keep, lmvm->m, &local_sub));
  PetscCall(MatDenseGetSubMatrix(local_temp, lmvm->m - m_keep, lmvm->m, lmvm->m - m_keep, lmvm->m, &temp_sub));
  PetscCall(MatCopy(local_sub, temp_sub, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));
  PetscCall(MatDenseGetSubMatrix(mat_local, 0, m_keep, 0, m_keep, &local_sub));
  PetscCall(MatCopy(temp_sub, local_sub, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));
  PetscCall(MatDenseRestoreSubMatrix(local_temp, &temp_sub));
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
    PetscInt StBS_idx = lbfgs->strategy == MAT_LBFGS_CD_INPLACE ? S_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatCDBFGSApplyJ0Fwd(B, s_j, Bs_j));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatMultTranspose(lbfgs->Sfull, Bs_j, StBs_j));
    lbfgs->St_count++;
    if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, StBs_j));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
  }
  prev_oldest = oldest_update(m, lbfgs->num_mult_updates);
  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER && prev_oldest < oldest_update(m, k)) {
    // move the YtS entries that have been computed and need to be kept back up
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, lbfgs->YtS_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(lbfgs->YtS_triu_strict, &m_local, NULL));
  j_0 = PetscMax(lbfgs->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt S_idx   = recycle_index(m, j);
    PetscInt YtS_idx = lbfgs->strategy == MAT_LBFGS_CD_INPLACE ? S_idx : history_index(m, k, j);
    Vec      s_j, Yts_j;

    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatMultTranspose(lbfgs->Yfull, s_j, Yts_j));
    lbfgs->Yt_count++;
    if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Yts_j));
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
  if (m_local) {
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
    PetscCall(MatMultTranspose(lbfgs->Sfull, F, rwork1));
    lbfgs->St_count++;
  }

  /* Reordering rwork1, as STY is in history order, while S is in recycled order */
  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_FALSE));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1));

  PetscCall(VecCopyAsync_Private(F, lbfgs->column_work, dctx));
  PetscCall(MatMultAdd(lbfgs->Yfull, rwork1, lbfgs->column_work, lbfgs->column_work));
  lbfgs->Y_count++;

  PetscCall(VecPointwiseMultAsync_Private(rwork1, lbfgs->diag_vec_recycle_order, rwork1, dctx));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->column_work, dX));

  PetscCall(MatMultTransposeAdd(lbfgs->Yfull, dX, rwork1, rwork1));
  lbfgs->Yt_count++;

  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_TRUE));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1));

  PetscCall(MatMultAdd(lbfgs->Sfull, rwork1, dX, dX));
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
   Byrd, Nocedal, Schnabel 1994                                            */

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat                J_local;
  PetscInt           m_local;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatMult, B, X, Z,0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (!lbfgs->num_updates) {
    PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(MatLMVMCDBFGSUpdateMultData(B));
  PetscCall(MatMultTranspose(lbfgs->Yfull, X, lbfgs->rwork1));
  lbfgs->Yt_count++;
  PetscCall(MatMultTranspose(lbfgs->Sfull, Z, lbfgs->rwork2));
  lbfgs->St_count++;
  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) {
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork1));
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork2));
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

  if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) {
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork1));
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork3));
  }

  PetscCall(MatMultAdd(lbfgs->Yfull, lbfgs->rwork1, Z, Z));
  lbfgs->Y_count++;
  PetscCall(MatMultAdd(lbfgs->BS, lbfgs->rwork3, Z, Z));
  lbfgs->S_count++;
  PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *diag_ctx;

  PetscScalar        curvature, ststmp;
  PetscReal          curvtol;
  Vec                workvec1;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (lmvm->prev_set) {
    Vec FX[3];
    PetscScalar dotFX[3];
    PetscScalar stF;

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPXAsync_Private(lmvm->Xprev, -1.0, X, dctx));
    /* Test if the updates can be accepted */
    FX[0] = lmvm->Xprev;
    FX[1] = F;
    FX[2] = lmvm->Fprev;
    PetscCall(VecMDot(lmvm->Xprev, 3, FX, dotFX));
    PetscCall(VecAYPXAsync_Private(lmvm->Fprev, -1.0, F, dctx));
    ststmp = dotFX[0];
    curvature = (dotFX[1] - dotFX[2]);
    stF = dotFX[2];
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      PetscInt  m = lmvm->m;
      PetscInt  k = lbfgs->num_updates;
      PetscInt  idx = recycle_index(m, k);
      PetscInt  StYidx;

      /* Update is good, accept it */
      lmvm->nupdates++;
      lbfgs->num_updates++;
      lbfgs->watchdog = 0;

      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        PetscCall(MatLMVMUpdate(lbfgs->diag_bfgs, X, F));
      }

      if (lmvm->k != m-1) {
        lmvm->k++;
      } else if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) {
        PetscCall(MatMove_LR3(B, lbfgs->StY_triu, m - 1));
      }


      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Sfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Xprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Sfull, idx, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Yfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Fprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Yfull, idx, &workvec1));

      StYidx = (lbfgs->strategy == MAT_LBFGS_CD_REORDER) ? history_index(m, lbfgs->num_updates, k) : idx;

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
            StFprev[idx] = stF;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&stF, PETSC_MEMTYPE_HOST, 1 * sizeof(stF)));
            PetscCall(PetscDeviceRegisterMemory(StFprev, memtype, local_n * sizeof(*StFprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &StFprev[idx], &stF, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(lbfgs->StFprev, &StFprev));

        // Now StFprev is updated for the new S vector.  Write -StFprev into the appropriate row
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->StY_triu, StYidx, &this_sy_col));
        PetscCall(VecAXPBYAsync_Private(this_sy_col, -1.0, 0.0, lbfgs->StFprev, dctx));

        // Now compute the new StFprev
        PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->StFprev));
        lbfgs->St_count++;

        // Now add StFprev: this_sy_col == S^T (F - Fprev) == S^T y
        PetscCall(VecAXPYAsync_Private(this_sy_col, 1.0, lbfgs->StFprev, dctx));

        if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StY_triu, StYidx, &this_sy_col));
      }

      PetscCall(MatGetDiagonal(lbfgs->StY_triu, lbfgs->diag_vec));
      if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) {
        if (!lbfgs->diag_vec_recycle_order) PetscCall(VecDuplicate(lbfgs->diag_vec, &lbfgs->diag_vec_recycle_order));
        PetscCall(VecCopy(lbfgs->diag_vec, lbfgs->diag_vec_recycle_order));
        PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->diag_vec_recycle_order));
      } else {
        if (!lbfgs->diag_vec_recycle_order) {
          PetscCall(PetscObjectReference((PetscObject)lbfgs->diag_vec));
          lbfgs->diag_vec_recycle_order = lbfgs->diag_vec;
        }
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
      lmvm->k = lmvm->k - 1;

      // we still have to maintain StFprev
      if (!lbfgs->StFprev) {
        PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->StFprev));
        PetscCall(VecZeroEntries(lbfgs->StFprev));
      }
      PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->StFprev));
      lbfgs->St_count++;
    }
  } else {
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      /* No previous updates have been set, so we just update the diagonal with an initial scalar */

      //TODO THIS PART IS ERROR AFTER RESET
      //1. USER_SCALE DISAPPEARS. NEED TO ADDRESS THIS
      //2. invD disappears (or was it ever there?)
      //
      //This is more fundamental to overall MATLMVM - set J0Scale is "Wrong", for an example
      dbase    = (Mat_LMVM *)lbfgs->diag_bfgs->data;
      diag_ctx = (Mat_DiagBrdn *)dbase->ctx;
      PetscCall(VecSet(diag_ctx->invD, lbfgs->delta));
    }
  }

  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMReset(lbfgs->diag_bfgs, PETSC_FALSE));
    }
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopyAsync_Private(X, lmvm->Xprev, dctx));
  PetscCall(VecCopyAsync_Private(F, lmvm->Fprev, dctx));
  PetscCall(PetscObjectReference((PetscObject)F));
  PetscCall(VecDestroy(&lbfgs->Fprev_ref));
  lbfgs->Fprev_ref = F;
  PetscCall(PetscObjectStateGet((PetscObject)F, &lbfgs->Fprev_state));
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

static PetscErrorCode MatLMVMCDBFGSResetDetructive(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lbfgs->Sfull));
  PetscCall(MatDestroy(&lbfgs->Yfull));
  PetscCall(MatDestroy(&lbfgs->BS));
  PetscCall(MatDestroy(&lbfgs->StY_triu));
  PetscCall(VecDestroy(&lbfgs->StFprev));
  PetscCall(VecDestroy(&lbfgs->Fprev_ref));
  lbfgs->Fprev_state = 0;
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
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatLMVMReset(lbfgs->diag_bfgs, destructive));
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
    lbfgs->Fprev_state = 0;
    if (lbfgs->StFprev) PetscCall(VecZeroEntries(lbfgs->StFprev));
  }
  if (destructive) {
    PetscCall(MatLMVMCDBFGSResetDetructive(B));
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
      PetscCall(VecZeroEntries(lbfgs->diag_vec));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->column_work));
    //TODO hacky way to turn diagbrdn lmvm off...
    PetscCall(MatLMVMSetJ0Scale(B,1.));
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
  PetscCall(MatLMVMCDBFGSResetDetructive(B));
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
  PetscCall(PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLBFGSType", MatLBFGSTypes, (PetscEnum)lbfgs->strategy, (PetscEnum *)&lbfgs->strategy, NULL));
  PetscOptionsEnd();
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
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  lbfgs->strategy        = MAT_LBFGS_CD_INPLACE;

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
