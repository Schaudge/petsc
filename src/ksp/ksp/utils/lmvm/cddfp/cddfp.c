#include <../src/ksp/ksp/utils/lmvm/cddfp/cddfp.h> /*I "petscksp.h" I*/
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

static PETSC_UNUSED PetscErrorCode MatMultColumnRange(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_Mult, (PetscObject)A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultColumnRange_C", (Mat, Vec, Vec, PetscInt, PetscInt), (A, xx, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_Mult, (PetscObject)A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAddColumnRange(Mat A, Vec xx, Vec zz, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultAdd, (PetscObject)A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultAddColumnRange_C", (Mat, Vec, Vec, Vec, PetscInt, PetscInt), (A, xx, zz, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultAdd, (PetscObject)A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeColumnRange(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultTranspose, (PetscObject)A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultTransposeColumnRange_C", (Mat, Vec, Vec, PetscInt, PetscInt), (A, xx, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultTranspose, (PetscObject)A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAddColumnRange(Mat A, Vec xx, Vec zz, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultTransposeAdd, (PetscObject)A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultTransposeAddColumnRange_C", (Mat, Vec, Vec, Vec, PetscInt, PetscInt), (A, xx, zz, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultTransposeAdd, (PetscObject)A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

const char *const MatLDFPTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLDFPType", "MAT_LDFP_", NULL};

PetscLogEvent CDDFP_MatMult;
PetscLogEvent CDDFP_MatSolve;
PetscLogEvent CDDFP_J0Inv;
PetscLogEvent CDDFP_J0Fwd;

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

PetscErrorCode MatCDDFPApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_J0Fwd, B, X, Z, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    ldfp->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM *dbase = (Mat_LMVM*)ldfp->diag_dfp->data;
    Mat_DiagBrdn *diagctx = (Mat_DiagBrdn *) dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (ldfp->scale_type) {
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
  PetscCall(PetscLogEventEnd(CDDFP_J0Fwd, B, X, Z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDDFPApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_J0Inv, B, F, dX, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    ldfp->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM *dbase = (Mat_LMVM*)ldfp->diag_dfp->data;
    Mat_DiagBrdn *diagctx = (Mat_DiagBrdn *) dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (ldfp->scale_type) {
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
  PetscCall(PetscLogEventEnd(CDDFP_J0Inv, B, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// shift a vector so that whatever is at index d becomes index 0
static PetscErrorCode VecCyclicShift(Mat B, Vec X, PetscInt d)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP  *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     n;
  const PetscScalar *src;
  PetscScalar *dest;
  PetscMemType src_memtype;
  PetscMemType dest_memtype;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(VecGetLocalSize(X, &n));
  if (!ldfp->cyclic_work_vec) PetscCall(VecDuplicate(X, &ldfp->cyclic_work_vec));
  PetscCall(VecCopyAsync_Private(X, ldfp->cyclic_work_vec, dctx));
  PetscCall(VecGetArrayReadAndMemType(ldfp->cyclic_work_vec, &src, &src_memtype));
  PetscCall(VecGetArrayWriteAndMemType(X, &dest, &dest_memtype));
  if (n == 0) { // no work on this process
    PetscCall(VecRestoreArrayWriteAndMemType(X, &dest));
    PetscCall(VecRestoreArrayReadAndMemType(ldfp->cyclic_work_vec, &src));
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
  PetscCall(VecRestoreArrayReadAndMemType(ldfp->cyclic_work_vec, &src));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRecycleOrderToHistoryOrder(Mat B, Vec X)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP  *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = ldfp->num_updates;
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
  Mat_CDDFP  *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = ldfp->num_updates;
  PetscInt     oldest_index;

  PetscFunctionBegin;
  oldest_index = recycle_index(m, oldest_update(m, k));
  if (oldest_index == 0) PetscFunctionReturn(PETSC_SUCCESS); // vector is already in recycle order
  PetscCall(VecCyclicShift(B, X, m - oldest_index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpperTriangularSolveInPlace_Internal(MatLDFPType ldfp_type, PetscMemType memtype, PetscBool hermitian_transpose, PetscInt N, PetscInt oldest_index, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
{
  PetscFunctionBegin;
  // if oldest_index == 0, the two strategies are equivalent, redirect to the simpler one
  if (oldest_index == 0) ldfp_type = MAT_LDFP_CD_REORDER;
  switch (ldfp_type) {
  case MAT_LDFP_CD_REORDER:
    if (PetscMemTypeHost(memtype)) {
      PetscBLASInt n, lda_blas, one = 1;
      PetscCall(PetscBLASIntCast(N, &n));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCallBLAS("BLAStrsv", BLAStrsv_("U", hermitian_transpose ? "C" : "N", "NotUnitTriangular", &n, A, &lda_blas, x, &one));
      PetscCall(PetscLogFlops(1.0 * n * n));
    } else if (PetscMemTypeDevice(memtype)) {
      PetscCall(MatUpperTriangularSolveInPlace_CUPM_DFP(hermitian_transpose, N, A, lda, x, 1));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported memtype");
    break;
  case MAT_LDFP_CD_INPLACE:
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
      PetscCall(MatUpperTriangularSolveInPlaceCyclic_CUPM_DFP(hermitian_transpose, N, oldest_index, A, lda, x, stride));
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
  Mat_CDDFP  *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt     m = lmvm->m;
  PetscInt     k = ldfp->num_updates;
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
  PetscCall(MatDenseGetLDA(ldfp->StY_triu, &lda));
  oldest_index = recycle_index(m, oldest_update(m, k));
  PetscCall(MatUpperTriangularSolveInPlace_Internal(ldfp->strategy, memtype_x, hermitian_transpose, h, oldest_index, A, lda, x, 1));
  PetscCall(VecRestoreArrayWriteAndMemType(X, &x));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Amat, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts R[end-m_keep:end,end-m_keep:end] to R[0:m_keep, 0:m_keep] */

static PetscErrorCode MatMove_LR3(Mat B, Mat R, PetscInt m_keep)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP   *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt     M;
  Mat          mat_local, local_sub, local_temp, temp_sub;

  PetscFunctionBegin;
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(R, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatGetLocalSize(R, &M, NULL));
  if (M == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatDenseGetLocalMatrix(R, &mat_local));
  PetscCall(MatDenseGetLocalMatrix(ldfp->temp_mat, &local_temp));
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
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt    m_local;

  PetscFunctionBegin;
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(ldfp->YtS_triu_strict, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatCopy(ldfp->YtS_triu_strict, ldfp->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(ldfp->temp_mat, ldfp->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  if (m_local) {
    Mat temp_local, YtS_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(ldfp->YtS_triu_strict, &YtS_local));
    PetscCall(MatDenseGetLocalMatrix(ldfp->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(YtS_local, temp_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &result_local));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCDDFPUpdateMultData(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscInt    m = lmvm->m, m_local;
  PetscInt    k = ldfp->num_updates;
  PetscInt    h = k - oldest_update(m, k);
  PetscInt    j_0;
  PetscInt    prev_oldest;
  Mat         J_local;

  PetscFunctionBegin;
  if (!ldfp->YtS_triu_strict) {
    PetscCall(MatDuplicate(ldfp->StY_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->YtS_triu_strict));
    PetscCall(MatDestroy(&ldfp->YtHY));
    PetscCall(MatDuplicate(ldfp->StY_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->YtHY));
    PetscCall(MatDestroy(&ldfp->J));
    PetscCall(MatDuplicate(ldfp->StY_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->J));
    PetscCall(MatDestroy(&ldfp->HY));
    PetscCall(MatDuplicate(ldfp->Yfull, MAT_SHARE_NONZERO_PATTERN, &ldfp->HY));
    PetscCall(MatZeroEntries(ldfp->YtS_triu_strict));
    PetscCall(MatZeroEntries(ldfp->HY));
    PetscCall(MatZeroEntries(ldfp->YtHY));
    PetscCall(MatZeroEntries(ldfp->J));
    PetscCall(MatShift(ldfp->YtHY, 1.0));
    ldfp->num_mult_updates = oldest_update(m, k);
  }
  if (ldfp->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  // H_0 may have been updated, we must recompute H_0 Y and Y^T H_0 Y
  // TODO: automatically track when H_0 y_j is stale
  // TODO: matrix-matrix product for Y^T H_0 Y
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      y_j;
    Vec      Hy_j;
    Vec      YtHy_j;
    PetscInt Y_idx = recycle_index(m, j);
    PetscInt YtHY_idx = ldfp->strategy == MAT_LDFP_CD_INPLACE ? Y_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
    PetscCall(MatDenseGetColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    PetscCall(MatCDDFPApplyJ0Inv(B, y_j, Hy_j));
    PetscCall(MatDenseRestoreColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, Hy_j, YtHy_j, 0, h));
    ldfp->Yt_count++;
    if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, YtHy_j));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
  }
  prev_oldest = oldest_update(m, ldfp->num_mult_updates);
  if (ldfp->strategy == MAT_LDFP_CD_REORDER && prev_oldest < oldest_update(m, k)) {
    // move the YtS entries that have been computed and need to be kept back up
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, ldfp->YtS_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(ldfp->YtS_triu_strict, &m_local, NULL));
  j_0 = PetscMax(ldfp->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt S_idx   = recycle_index(m, j);
    PetscInt YtS_idx = ldfp->strategy == MAT_LDFP_CD_INPLACE ? S_idx : history_index(m, k, j);
    Vec      s_j, Yts_j;

    PetscCall(MatDenseGetColumnVecRead(ldfp->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, s_j, Yts_j, 0, h));
    ldfp->Yt_count++;
    if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Yts_j));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatDenseRestoreColumnVecRead(ldfp->Sfull, S_idx, &s_j));
    // zero the corresponding row
    if (m_local > 0) {
      Mat YtS_local, YtS_row;

      PetscCall(MatDenseGetLocalMatrix(ldfp->YtS_triu_strict, &YtS_local));
      PetscCall(MatDenseGetSubMatrix(YtS_local, YtS_idx, YtS_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &YtS_row));
      PetscCall(MatZeroEntries(YtS_row));
      PetscCall(MatDenseRestoreSubMatrix(YtS_local, &YtS_row));
    }
  }
  {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

    if (!ldfp->inv_diag_vec) PetscCall(VecDuplicate(ldfp->diag_vec, &ldfp->inv_diag_vec));
    PetscCall(VecCopyAsync_Private(ldfp->diag_vec, ldfp->inv_diag_vec, dctx));
    PetscCall(VecReciprocalAsync_Private(ldfp->inv_diag_vec, dctx));
  }
  PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetLDLT(B, ldfp->J));
  PetscCall(MatAXPY(ldfp->J, 1.0, ldfp->YtHY, SAME_NONZERO_PATTERN));
  if (m_local) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  ldfp->num_mult_updates = ldfp->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for

   H_0 - [ S | H_0 Y] [ -D  |    R      ]^-1 [   S^T   ]
                      [-----+-----------]    [---------]
                      [ R^T | Y^T H_0 Y ]    [ Y^T H_0 ]

   Above is equivalent to

   H_0 - [ S | H_0 Y] [[      I      | 0 ][ -D | 0 ][ I | -D^{-1} R ]]^-1 [   S^T   ]
                      [[-------------+---][----+---][---+-----------]]    [---------]
                      [[ -R^T D^{-1} | I ][  0 | J ][ 0 |     I     ]]    [ Y^T H_0 ]

   where J = Y^T H_0 Y + R^T D^{-1} R

   becomes

   H_0 - [ S | H_0 Y] [ I | D^{-1} R ][ -D^{-1}  |   0    ][      I     | 0 ] [   S^T   ]
                      [---+----------][----------+--------][------------+---] [---------]
                      [ 0 |     I    ][     0    | J^{-1} ][ R^T D^{-1} | I ] [ Y^T H_0 ]

                      =

   H_0 + [ S | H_0 Y] [ D^{-1} | 0 ][ I | R  ][ I |    0    ][      I     | 0 ] [   S^T   ]
                      [--------+---][---+----][---+---------][------------+---] [---------]
                      [ 0      | I ][ 0 | I  ][ 0 | -J^{-1} ][ R^T D^{-1} | I ] [ Y^T H_0 ]

                      (Note that YtS_triu_strict is L^T)
   Byrd, Nocedal, Schnabel 1994

*/
static PetscErrorCode MatSolve_LMVMCDDFP(Mat H, Vec F, Vec dX)
{
  Mat_LMVM   *lmvm = (Mat_LMVM*)H->data;
  Mat_CDDFP  *ldfp = (Mat_CDDFP*)lmvm->ctx;
  PetscDeviceContext dctx;
  PetscInt m = lmvm->m;
  PetscInt k = ldfp->num_updates;
  PetscInt h = k - oldest_update(m, k);
  PetscInt m_local;
  Mat J_local;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_MatSolve, H, F, dX,0));
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatCDDFPApplyJ0Inv(H, F, dX));
  if (!ldfp->num_updates) {
    PetscCall(PetscLogEventEnd(CDDFP_MatSolve, H, F, dX, 0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(MatLMVMCDDFPUpdateMultData(H));

  PetscCall(MatMultTransposeColumnRange(ldfp->Sfull, F, ldfp->rwork1, 0, h));
  ldfp->St_count++;
  PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, dX, ldfp->rwork2, 0, h));
  ldfp->Yt_count++;
  if (ldfp->strategy == MAT_LDFP_CD_REORDER) {
    PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork1));
    PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork2));
  }

  PetscCall(VecPointwiseMultAsync_Private(ldfp->rwork3, ldfp->rwork1, ldfp->inv_diag_vec, dctx));
  //TODO below is L, we need R
  //Dumb way to get strictly upper triangular. fix later
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(ldfp->StY_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatCopy(ldfp->StY_triu, ldfp->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(VecZeroEntries(ldfp->rwork4));
  PetscCall(MatDiagonalSet(ldfp->temp_mat, ldfp->rwork4, INSERT_VALUES));
  //PetscCall(MatMultTransposeAdd(ldfp->temp_mat, ldfp->rwork3, ldfp->rwork2, ldfp->rwork2));
  PetscCall(MatMultAdd(ldfp->temp_mat, ldfp->rwork3, ldfp->rwork2, ldfp->rwork2));

  if (!ldfp->rwork2_local) PetscCall(VecCreateLocalVector(ldfp->rwork2, &ldfp->rwork2_local));
  if (!ldfp->rwork3_local) PetscCall(VecCreateLocalVector(ldfp->rwork3, &ldfp->rwork3_local));
  PetscCall(VecGetLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
  PetscCall(VecGetLocalVector(ldfp->rwork3, ldfp->rwork3_local));
  PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
  PetscCall(VecGetSize(ldfp->rwork2_local, &m_local));
  if (m_local) {
    Mat J_local;

    PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
    PetscCall(MatSolve(J_local, ldfp->rwork2_local, ldfp->rwork3_local));
  }
  PetscCall(VecRestoreLocalVector(ldfp->rwork3, ldfp->rwork3_local));
  PetscCall(VecRestoreLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
  PetscCall(VecScale(ldfp->rwork3, -1.0));
  //TODO below is L, we need R
  //PetscCall(MatMultAdd(ldfp->temp_mat, ldfp->rwork3, ldfp->rwork1, ldfp->rwork1));
  PetscCall(MatMultTransposeAdd(ldfp->temp_mat, ldfp->rwork3, ldfp->rwork1, ldfp->rwork1));

  PetscCall(VecPointwiseMultAsync_Private(ldfp->rwork1, ldfp->rwork1, ldfp->inv_diag_vec, dctx));

  if (ldfp->strategy == MAT_LDFP_CD_REORDER) {
    PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork1));
    PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork3));
  }

  PetscCall(MatMultAddColumnRange(ldfp->Sfull, ldfp->rwork1, dX, dX, 0, h));
  ldfp->S_count++;
  PetscCall(MatMultAddColumnRange(ldfp->HY, ldfp->rwork3, dX, dX, 0, h));
  ldfp->Y_count++;
  PetscCall(PetscLogEventEnd(CDDFP_MatMult, H, F, dX,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   (Theorem 1, Erway, Jain, and Marcia, 2013)

   B_0 - [ Y | B_0 S] [ -R^{-T} (D + S^T B_0 S) R^{-1} | R^{-T} ] [   Y^T   ]
                      ---------------------------------+--------] [---------]
                      [             R^{-1}             |   0    ] [ S^T B_0 ]

   (Note: R above is right triangular part of YTS)
   which becomes,

   [ I | -Y L^{-1} ] [  I  | 0 ] [ B_0 | 0 ] [ I | S ] [      I      ]
                     [-----+---] [-----+---] [---+---] [-------------]
                     [ S^T | I ] [  0  | D ] [ 0 | I ] [ -L^{-T} Y^T ]

   (Note: L above is right triangular part of STY)

*/
static PetscErrorCode MatMult_LMVMCDDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP        *ldfp = (Mat_CDDFP*)lmvm->ctx;
  Vec rwork1 = ldfp->rwork1;
  PetscInt           m = lmvm->m;
  PetscInt           k = ldfp->num_updates;
  PetscInt           h = k - oldest_update(m, k);
  PetscDeviceContext dctx;
  PetscObjectState   Xstate;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_MatMult, B, X, Z,0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* DFP Version. Erway, Jain, Marcia, 2013, Theorem 1 */
  /* Block Version */
  if (!ldfp->num_updates) {
    PetscCall(MatCDDFPApplyJ0Fwd(B, X, Z));
    PetscCall(PetscLogEventEnd(CDDFP_MatMult, B, X, Z,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(MatLMVMCDDFPUpdateMultData(B));

  PetscCall(PetscObjectStateGet((PetscObject)X, &Xstate));
  if (X == ldfp->Xprev_ref && Xstate == ldfp->Xprev_state) {
    PetscCall(VecCopyAsync_Private(ldfp->YtXprev, rwork1, dctx));
  } else {
    PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, X, rwork1, 0, h));
    ldfp->Yt_count++;
  }

  /* Reordering rwork1, as STY is in history order, while Y is in recycled order */
  if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1));
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(ldfp->YtS_triu_strict, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatCopy(ldfp->YtS_triu_strict, ldfp->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalSet(ldfp->temp_mat, ldfp->diag_vec, INSERT_VALUES));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->temp_mat, rwork1, PETSC_FALSE));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1));

  PetscCall(VecCopyAsync_Private(X, ldfp->column_work, dctx));
  PetscCall(MatMultAddColumnRange(ldfp->Sfull, rwork1, ldfp->column_work, ldfp->column_work, 0, h));
  ldfp->S_count++;

  PetscCall(VecPointwiseMultAsync_Private(rwork1, ldfp->diag_vec_recycle_order, rwork1, dctx));
  PetscCall(MatCDDFPApplyJ0Fwd(B, ldfp->column_work, Z));

  PetscCall(MatMultTransposeAddColumnRange(ldfp->Sfull, Z, rwork1, rwork1, 0, h));
  ldfp->St_count++;

  if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->temp_mat, rwork1, PETSC_TRUE));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1));

  PetscCall(MatMultAddColumnRange(ldfp->Yfull, rwork1, Z, Z, 0, h));
  ldfp->Y_count++;
  PetscCall(PetscLogEventEnd(CDDFP_MatMult, B, X, Z,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP        *ldfp = (Mat_CDDFP*)lmvm->ctx;
  Mat_LMVM          *dbase = (Mat_LMVM *)ldfp->diag_dfp->data;
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
      PetscInt  k = ldfp->num_updates;
      PetscInt  h_new = k + 1 - oldest_update(m, k + 1);
      PetscInt  idx = recycle_index(m, k);
      PetscInt  StYidx;

      /* Update is good, accept it */
      lmvm->nupdates++;
      ldfp->num_updates++;
      ldfp->watchdog = 0;

      if (lmvm->k != m-1) {
        lmvm->k++;
      } else if (ldfp->strategy == MAT_LDFP_CD_REORDER) {
        PetscCall(MatMove_LR3(B, ldfp->StY_triu, m - 1));
      }

      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(ldfp->Sfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Xprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(ldfp->Sfull, idx, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(ldfp->Yfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Fprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(ldfp->Yfull, idx, &workvec1));

      StYidx = (ldfp->strategy == MAT_LDFP_CD_REORDER) ? history_index(m, ldfp->num_updates, k) : idx;

      { // implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the
        // H_k is immediately applied to F after begin updated.   The S^T y computation can be split up as S^T (F - F_prev)
        Vec this_sy_col;
        PetscInt local_n;
        PetscScalar *StFprev;
        PetscMemType memtype;

        if (!ldfp->StFprev) {
          PetscCall(VecDuplicate(ldfp->rwork1, &ldfp->StFprev));
          PetscCall(VecZeroEntries(ldfp->StFprev));
        }
        PetscCall(VecGetLocalSize(ldfp->StFprev, &local_n));
        PetscCall(VecGetArrayAndMemType(ldfp->StFprev, &StFprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            StFprev[idx] = stFprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&stFprev, PETSC_MEMTYPE_HOST, 1 * sizeof(stFprev)));
            PetscCall(PetscDeviceRegisterMemory(StFprev, memtype, local_n * sizeof(*StFprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &StFprev[idx], &stFprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(ldfp->StFprev, &StFprev));

        // Now StFprev is updated for the new S vector.  Write -StFprev into the appropriate row
        PetscCall(MatDenseGetColumnVecWrite(ldfp->StY_triu, StYidx, &this_sy_col));
        PetscCall(VecAXPBYAsync_Private(this_sy_col, -1.0, 0.0, ldfp->StFprev, dctx));

        // Now compute the new StFprev
        PetscCall(MatMultTransposeColumnRange(ldfp->Sfull, F, ldfp->StFprev, 0, h_new));
        ldfp->St_count++;

        // Now add StFprev: this_sy_col == S^T (F - Fprev) == S^T y
        PetscCall(VecAXPYAsync_Private(this_sy_col, 1.0, ldfp->StFprev, dctx));

        if (ldfp->strategy == MAT_LDFP_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col));
        PetscCall(MatDenseRestoreColumnVecWrite(ldfp->StY_triu, StYidx, &this_sy_col));
      }

      { // implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the
        // B_k is immediately applied to X after begin updated.   The Y^T x computation can be split up as Y^T (X - X_prev)
        PetscInt local_n;
        PetscScalar *YtXprev;
        PetscMemType memtype;

        if (!ldfp->YtXprev) {
          PetscCall(VecDuplicate(ldfp->rwork1, &ldfp->YtXprev));
          PetscCall(VecZeroEntries(ldfp->YtXprev));
        }
        PetscCall(VecGetLocalSize(ldfp->YtXprev, &local_n));
        PetscCall(VecGetArrayAndMemType(ldfp->YtXprev, &YtXprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            YtXprev[idx] = ytXprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&ytXprev, PETSC_MEMTYPE_HOST, 1 * sizeof(ytXprev)));
            PetscCall(PetscDeviceRegisterMemory(YtXprev, memtype, local_n * sizeof(*YtXprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &YtXprev[idx], &ytXprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(ldfp->YtXprev, &YtXprev));

        // Now compute the new YtXprev
        PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, X, ldfp->YtXprev, 0, h_new));
        ldfp->Yt_count++;
      }

      PetscCall(MatGetDiagonal(ldfp->StY_triu, ldfp->diag_vec));
      if (ldfp->strategy == MAT_LDFP_CD_REORDER) {
        if (!ldfp->diag_vec_recycle_order) PetscCall(VecDuplicate(ldfp->diag_vec, &ldfp->diag_vec_recycle_order));
        PetscCall(VecCopyAsync_Private(ldfp->diag_vec, ldfp->diag_vec_recycle_order, dctx));
        PetscCall(VecHistoryOrderToRecycleOrder(B, ldfp->diag_vec_recycle_order));
      } else {
        if (!ldfp->diag_vec_recycle_order) {
          PetscCall(PetscObjectReference((PetscObject)ldfp->diag_vec));
          ldfp->diag_vec_recycle_order = ldfp->diag_vec;
        }
      }

      if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        PetscScalar sTy = curvature;

        PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
        diagctx->sigma = sTy / yTy;
      } else if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
        // Diagonal Barzilai-Borwein after Park et al.
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
      ++ldfp->watchdog;
      lmvm->k = lmvm->k - 1;
      PetscInt  m = lmvm->m;
      PetscInt  k = ldfp->num_updates;
      PetscInt  h = k - oldest_update(m, k);

      // we still have to maintain StFprev
      if (!ldfp->StFprev) {
        PetscCall(VecDuplicate(ldfp->rwork1, &ldfp->StFprev));
        PetscCall(VecZeroEntries(ldfp->StFprev));
      }
      PetscCall(MatMultTransposeColumnRange(ldfp->Sfull, F, ldfp->StFprev, 0, h));
      ldfp->St_count++;
      // we still have to maintain YtXprev
      if (!ldfp->YtXprev) {
        PetscCall(VecDuplicate(ldfp->rwork1, &ldfp->YtXprev));
        PetscCall(VecZeroEntries(ldfp->YtXprev));
      }
      PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, X, ldfp->YtXprev, 0, h));
      ldfp->Yt_count++;
    }
  } else {
    switch (ldfp->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecSetAsync_Private(diagctx->invD, diagctx->delta, dctx));
      //printf("h:\n");
      //PetscCall(VecView(diagctx->invD, PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)B))));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      diagctx->sigma = diagctx->delta;
      break;
    default:
      diagctx->sigma = 1.0;
      break;
    }
  }

  if (ldfp->watchdog > ldfp->max_seq_rejects) PetscCall(MatLMVMReset(B, PETSC_FALSE));

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopyAsync_Private(X, lmvm->Xprev, dctx));
  PetscCall(VecCopyAsync_Private(F, lmvm->Fprev, dctx));
  PetscCall(PetscObjectReference((PetscObject)F));
  PetscCall(VecDestroy(&ldfp->Fprev_ref));
  ldfp->Fprev_ref = F;
  PetscCall(PetscObjectStateGet((PetscObject)F, &ldfp->Fprev_state));

  PetscCall(PetscObjectReference((PetscObject)X));
  PetscCall(VecDestroy(&ldfp->Xprev_ref));
  ldfp->Xprev_ref = X;
  PetscCall(PetscObjectStateGet((PetscObject)X, &ldfp->Xprev_state));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMCDDFP(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM   *bdata  = (Mat_LMVM*)B->data;
  Mat_CDDFP *bldfp = (Mat_CDDFP*)bdata->ctx;
  Mat_LMVM   *mdata  = (Mat_LMVM*)M->data;
  Mat_CDDFP *mldfp = (Mat_CDDFP*)mdata->ctx;

  PetscFunctionBegin;
  mldfp->watchdog        = bldfp->watchdog;
  mldfp->max_seq_rejects = bldfp->max_seq_rejects;
  if (!(bdata->J0 || bdata->user_pc || bdata->user_ksp || bdata->user_scale)) {
    PetscCall(MatCopy(bldfp->diag_dfp, mldfp->diag_dfp, SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatLMVMCDDFPResetDetructive(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ldfp->Sfull));
  PetscCall(MatDestroy(&ldfp->Yfull));
  PetscCall(MatDestroy(&ldfp->HY));
  PetscCall(MatDestroy(&ldfp->StY_triu));
  PetscCall(VecDestroy(&ldfp->StFprev));
  PetscCall(VecDestroy(&ldfp->Fprev_ref));
  PetscCall(VecDestroy(&ldfp->Xprev_ref));
  ldfp->Fprev_state = 0;
  ldfp->Xprev_state = 0;
  PetscCall(MatDestroy(&ldfp->YtS_triu_strict));
  PetscCall(MatDestroy(&ldfp->LDLt));
  PetscCall(MatDestroy(&ldfp->YtHY));
  PetscCall(MatDestroy(&ldfp->J));
  PetscCall(MatDestroy(&ldfp->temp_mat));
  PetscCall(VecDestroy(&ldfp->diag_vec));
  PetscCall(VecDestroy(&ldfp->diag_vec_recycle_order));
  PetscCall(VecDestroy(&ldfp->inv_diag_vec));
  PetscCall(VecDestroy(&ldfp->column_work));
  PetscCall(VecDestroy(&ldfp->rwork1));
  PetscCall(VecDestroy(&ldfp->rwork2));
  PetscCall(VecDestroy(&ldfp->rwork3));
  PetscCall(VecDestroy(&ldfp->rwork4));
  PetscCall(VecDestroy(&ldfp->rwork2_local));
  PetscCall(VecDestroy(&ldfp->rwork3_local));
  if (!ldfp->cyclic_work_vec) PetscCall(VecDestroy(&ldfp->cyclic_work_vec));
  ldfp->allocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMCDDFP(Mat B, PetscBool destructive)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  ldfp->watchdog = 0;
  if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMReset(ldfp->diag_dfp, destructive));
  if (ldfp->Sfull) PetscCall(MatZeroEntries(ldfp->Sfull));
  if (ldfp->Yfull) PetscCall(MatZeroEntries(ldfp->Yfull));
  if (ldfp->HY) PetscCall(MatZeroEntries(ldfp->HY));
  if (ldfp->StY_triu) { // Set to identity by default so it is invertible
    PetscCall(MatZeroEntries(ldfp->StY_triu));
    PetscCall(MatShift(ldfp->StY_triu, 1.0));
  }
  if (ldfp->YtS_triu_strict) PetscCall(MatZeroEntries(ldfp->YtS_triu_strict));
  if (ldfp->LDLt) PetscCall(MatZeroEntries(ldfp->LDLt));
  if (ldfp->YtHY) {
    PetscCall(MatZeroEntries(ldfp->YtHY));
    PetscCall(MatShift(ldfp->YtHY, 1.0));
  }
  if (ldfp->Fprev_ref) PetscCall(VecDestroy(&ldfp->Fprev_ref));
  if (ldfp->Xprev_ref) PetscCall(VecDestroy(&ldfp->Xprev_ref));
  ldfp->Fprev_state = 0;
  ldfp->Xprev_state = 0;
  if (ldfp->StFprev) PetscCall(VecZeroEntries(ldfp->StFprev));
  if (ldfp->YtXprev) PetscCall(VecZeroEntries(ldfp->YtXprev));
  if (destructive) {
    PetscCall(MatLMVMCDDFPResetDetructive(B));
  }
  ldfp->num_updates = 0;
  ldfp->num_mult_updates = 0;
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;
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
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, n, m, N, M, -1, NULL, &ldfp->Sfull));
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &ldfp->StY_triu));
      PetscCall(MatDuplicate(ldfp->Sfull, MAT_SHARE_NONZERO_PATTERN, &ldfp->Yfull));
      PetscCall(MatDuplicate(ldfp->Yfull, MAT_SHARE_NONZERO_PATTERN, &ldfp->HY));
      PetscCall(MatZeroEntries(ldfp->Sfull));
      PetscCall(MatZeroEntries(ldfp->Yfull));
      // initialize StY_triu to identity so it is invertible
      PetscCall(MatZeroEntries(ldfp->StY_triu));
      PetscCall(MatShift(ldfp->StY_triu, 1.0));
      PetscCall(MatCreateVecs(ldfp->StY_triu, &ldfp->diag_vec, &ldfp->rwork1));
      PetscCall(MatCreateVecs(ldfp->StY_triu, &ldfp->rwork2, &ldfp->rwork3));
      PetscCall(MatCreateVecs(ldfp->StY_triu, NULL, &ldfp->rwork4));
      PetscCall(VecZeroEntries(ldfp->rwork1));
      PetscCall(VecZeroEntries(ldfp->rwork2));
      PetscCall(VecZeroEntries(ldfp->rwork3));
      PetscCall(VecZeroEntries(ldfp->rwork4));
      PetscCall(VecZeroEntries(ldfp->diag_vec));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &ldfp->column_work));
    //TODO hacky way to turn diagbrdn lmvm off...
    //PetscCall(MatLMVMSetJ0Scale(B,1.));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMAllocate(ldfp->diag_dfp, X, F));
    }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDDFP(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatLMVMCDDFPResetDetructive(B));
  PetscCall(MatDestroy(&ldfp->diag_dfp));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMCDDFP(Mat B)
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
    PetscCall(MatAllocate_LMVMCDDFP(B, Xtmp, Ftmp));
    PetscCall(VecDestroy(&Xtmp));
    PetscCall(VecDestroy(&Ftmp));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMCDDFP(Mat B, PetscViewer pv)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  PetscCall(MatView_LMVM(B, pv));
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatView(ldfp->diag_dfp, pv));
  }
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "Counts: S x : %" PetscInt_FMT ", S^T x : %" PetscInt_FMT ", Y x : %" PetscInt_FMT ",  Y^T x: %" PetscInt_FMT "\n", ldfp->S_count, ldfp->St_count, ldfp->Y_count, ldfp->Yt_count));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMCDDFP(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDDFP *ldfp = (Mat_CDDFP*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix,  "Compact dense DFP method (MATLMVMCDDFP)", NULL);
  PetscCall(PetscOptionsEnum("-mat_ldfp_type", "Implementation options for L-DFP", "MatLDFPType", MatLDFPTypes, (PetscEnum)ldfp->strategy, (PetscEnum *)&ldfp->strategy, NULL));
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBrdnScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)ldfp->scale_type, (PetscEnum *)&ldfp->scale_type, NULL));
  PetscCall(PetscOptionsBool("-mat_ldfp_mult_type", "True for Cholesky type MatMult_CDDFP, False for DFP type..", "", ldfp->mult_type, &ldfp->mult_type, NULL));
  ldfp->allocated       = PETSC_FALSE;
  PetscOptionsEnd();
  if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    const char *prefix;

    PetscCall(MatGetOptionsPrefix(B, &prefix));
    PetscCall(MatSetOptionsPrefix(ldfp->diag_dfp, prefix));
    PetscCall(MatAppendOptionsPrefix(ldfp->diag_dfp, "J0_"));
    PetscCall(MatSetFromOptions(ldfp->diag_dfp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDDFP(Mat B)
{
  Mat_LMVM   *lmvm;
  Mat_CDDFP *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDDFP));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view = MatView_LMVMCDDFP;
  B->ops->setup = MatSetUp_LMVMCDDFP;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDDFP;
  B->ops->destroy = MatDestroy_LMVMCDDFP;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDDFP;
  lmvm->ops->reset = MatReset_LMVMCDDFP;
  lmvm->ops->update = MatUpdate_LMVMCDDFP;
  lmvm->ops->mult = MatMult_LMVMCDDFP;
  lmvm->ops->solve = MatSolve_LMVMCDDFP;
  lmvm->ops->copy = MatCopy_LMVMCDDFP;

  PetscCall(PetscNew(&ldfp));
  lmvm->ctx = (void*)ldfp;
  ldfp->allocated       = PETSC_FALSE;
  ldfp->mult_type       = PETSC_FALSE;
  ldfp->watchdog        = 0;
  ldfp->max_seq_rejects = lmvm->m/2;
  ldfp->strategy        = MAT_LDFP_CD_INPLACE;
  ldfp->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &ldfp->diag_dfp));
  PetscCall(MatSetType(ldfp->diag_dfp, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(ldfp->diag_dfp, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMCDDFP - Creates a compact dense representation of the limited-memory
   Broyden-Fletcher-Goldfarb-Shanno (DFP) approximation to a Hessian. This compact
   dense representation reduces the L-DFP update to a series of matrix-vector products
   with compact dense matrices in lieu of the conventional matrix-free two-loop
   algorithm. For most problems on CPUs, this compact dense representation is not as
   fast as the matrix-free two-loop implementation provided via MATLMVMDFP. However,
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

.seealso: MatCreate(), MATLMVM, MATLMVMCDDFP, MatCreateLMVMDFP()
@*/
PetscErrorCode MatCreateLMVMCDDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
