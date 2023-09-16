#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#if defined(PETSC_HAVE_CUDA)
#include <petscdevice_cuda.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/vecimpl.h>
#include <cuda_profiler_api.h>
#endif
typedef enum{
  MAT_CDBFGS_LOWER_TRIANGULAR,
  MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE,
  MAT_CDBFGS_UPPER_TRIANGULAR,
  MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE,
} TriangularTypes;

const char *const MatLBFGSTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLBFGSType", "MAT_LBFGS_", NULL};

PetscLogEvent CDBFGS_MatMult;
PetscLogEvent CDBFGS_MatSolve;
PetscLogEvent CDBFGS_J0Inv;
PetscLogEvent CDBFGS_J0Fwd;

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

/* MatTransposeMatMult, but for CDINPLACE. Instead of computing
 * the whole matrix again, this routine just updates row and col at certain index. */
static PETSC_UNUSED PetscErrorCode MtMT_Internal(Mat H, Mat S, Mat Y, Mat *STY)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscMemType       stype, ytype, stytype, vec_type;
  const PetscScalar  *s_array, *y_array;
  PetscScalar        *sty_array, *vec_array, Alpha = 1.0, zero = 0.;

  PetscInt idx = lbfgs->idx_rplc, lda, lda_sty, rown, coln, i;
  MPI_Comm comm = PetscObjectComm((PetscObject)H);
  PetscFunctionBegin;

  PetscCall(VecGetArrayAndMemType(lbfgs->rwork1, &vec_array, &vec_type));
  PetscCall(MatDenseGetArrayReadAndMemType(S, &s_array, &stype));
  PetscCall(MatDenseGetArrayReadAndMemType(Y, &y_array, &ytype));
  PetscCall(MatDenseGetArrayAndMemType(*STY, &sty_array, &stytype));
  PetscAssert((vec_type == stype) == (ytype == stytype), comm, PETSC_ERR_PLIB, "Incompatible device pointers");
  PetscCall(MatDenseGetLDA(S, &lda));
  PetscCall(MatDenseGetLDA(*STY, &lda_sty));
  PetscCall(MatGetSize(S, &rown, &coln));
  switch (stype) {
  case PETSC_MEMTYPE_HOST:
    {
      /* First fill the column: S^T @ idx_col(Y) */
      PetscBLASInt row_blas, col_blas, lda_blas, lda_sty_blas, one = 1;
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCall(PetscBLASIntCast(lda_sty, &lda_sty_blas));
      PetscCall(PetscBLASIntCast(rown, &row_blas));
      PetscCall(PetscBLASIntCast(coln, &col_blas));
      PetscCallBLAS("BLASgemv", BLASgemv_("T", &row_blas, &col_blas, &Alpha, s_array, &lda_blas, &y_array[idx*lda], &one, &zero, &sty_array[idx*lda_sty_blas], &one));

      /* Second, compute row, and store it on vector */
      PetscCallBLAS("BLASgemv", BLASgemv_("T", &row_blas, &col_blas, &Alpha, y_array, &lda_blas, &s_array[idx*lda], &one, &zero, vec_array, &one));

      /* Write it back */
      for (i=0; i< lmvm->m; i++) {
        sty_array[i*lda_sty+idx] = vec_array[i];
      }
    }
    break;
  case PETSC_MEMTYPE_CUDA:
  case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
    { 
      PetscDeviceContext dctx;
      cublasHandle_t     handle;
      PetscCuBLASInt     row_blas, col_blas, lda_blas, lda_sty_blas, one = 1;

      PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
      PetscCall(PetscCUBLASGetHandle(&handle));
      PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
      PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
      PetscCall(PetscCuBLASIntCast(lda_sty, &lda_sty_blas));
      PetscCall(PetscCuBLASIntCast(rown, &row_blas));
      PetscCall(PetscCuBLASIntCast(coln, &col_blas));
      PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_T, row_blas, col_blas, &Alpha, s_array, lda_blas, &y_array[idx*lda], one, &zero, &sty_array[idx*lda_sty_blas], one));
      PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_T, row_blas, col_blas, &Alpha, y_array, lda_blas, &s_array[idx*lda], one, &zero, vec_array, one));

      for (i=0; i< lmvm->m ; i++) {
        PetscCall(PetscDeviceRegisterMemory(&sty_array[i*lda_sty+idx], stytype, 1*sizeof(*sty_array)));
        PetscCall(PetscDeviceRegisterMemory(&vec_array[i], vec_type, 1*sizeof(*vec_array)));
	PetscCall(PetscDeviceArrayCopy(dctx, &sty_array[i*lda_sty+idx], &vec_array[i], 1));
      }
    }
#endif
    break;
  case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
    {
      hipblasHandle_t handle;
      PetscHIPBLASInt row_blas, col_blas, lda_blas, lda_sty_blas, one = 1;

      PetscCall(PetscHIPBLASGetHandle(&handle));
      PetscCallHIPBLAS(cublasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
      PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
      PetscCall(PetscHIPBLASIntCast(lda_sty, &lda_sty_blas));
      PetscCall(PetscHIPBLASIntCast(rown, &row_blas));
      PetscCall(PetscHIPBLASIntCast(coln, &col_blas));
      PetscCallHIPBLAS(cublasDgemv(handle, HIPBLAS_OP_T, row_blas, col_blas, &Alpha, s_array, lda_blas, &y_array[idx*lda], one, &Alpha, &sty_array[idx*lda_sty_blas], one));
    }
#endif
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
  }
  
  PetscCall(VecRestoreArrayAndMemType(lbfgs->rwork1, &vec_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(S, &s_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Y, &y_array));
  PetscCall(MatDenseRestoreArrayAndMemType(*STY, &sty_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts vector rightward by given step. If given input is negative, it shifts leftward. 
 * This routine is only for Replace version. */

static PetscErrorCode VecRightward_Shift(Mat B, Vec X, PetscInt step)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscScalar *buffer1, *x_array;
  PetscInt     size, N;
  MPI_Comm     comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &N));
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  switch (lbfgs->strategy) {
    case MAT_LBFGS_CD_REORDER:
      if ((lbfgs->idx_rplc == lmvm->m) || lbfgs->idx_rplc <= 0) {
        break;
      } else {
        PetscMemType memtype_x;
        size = PetscAbs(step);
        PetscCall(VecGetArrayAndMemType(X, &x_array, &memtype_x));
        switch (memtype_x) {
          case PETSC_MEMTYPE_HOST:
            {
              PetscCall(PetscMalloc1(size, &buffer1));
              if (step < 0) {
                PetscCall(PetscArraycpy(buffer1, x_array, size));
                PetscCall(PetscArraymove(x_array, &x_array[size], N - size));
                PetscCall(PetscArraycpy(&x_array[N - size], buffer1, size));
              } else {
                PetscCall(PetscArraycpy(buffer1, &x_array[N-size], size));
                PetscCall(PetscArraymove(&x_array[size], x_array, N - size));
                PetscCall(PetscArraycpy(x_array, buffer1, size));
              }
            }
            PetscCall(PetscFree(buffer1));
            break;
          case PETSC_MEMTYPE_CUDA:
          case PETSC_MEMTYPE_NVSHMEM:
          case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
            {
              PetscScalar *buffer2;
              PetscDeviceContext dctx;
              PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
              PetscCall(PetscDeviceRegisterMemory(x_array, memtype_x, N*sizeof(*x_array)));
              PetscCall(PetscDeviceMalloc(dctx, memtype_x, size, &buffer1));
              PetscCall(PetscDeviceMalloc(dctx, memtype_x, N-size, &buffer2));
              if (step < 0) {
                PetscCall(PetscDeviceArrayCopy(dctx, buffer1, x_array, size));
                PetscCall(PetscDeviceArrayCopy(dctx, buffer2, &x_array[size], N-size));
                PetscCall(PetscDeviceArrayCopy(dctx, x_array, buffer2, N-size));
                PetscCall(PetscDeviceArrayCopy(dctx, &x_array[N - size], buffer1, size));
              } else {
                PetscCall(PetscDeviceArrayCopy(dctx, buffer1, &x_array[N-size], size));
                PetscCall(PetscDeviceArrayCopy(dctx, buffer2, x_array, N-size));
                PetscCall(PetscDeviceArrayCopy(dctx, &x_array[size], buffer2, N-size));
                PetscCall(PetscDeviceArrayCopy(dctx, x_array, buffer1, size));
              }
              PetscCall(PetscDeviceFree(dctx, buffer1));
              PetscCall(PetscDeviceFree(dctx, buffer2));
            }
#endif
            break;
          default:
            SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
        }

        PetscCall(VecRestoreArrayAndMemType(X, &x_array));
      }
      break;
    case MAT_LBFGS_CD_INPLACE:
    case MAT_LBFGS_BASIC:
    default:
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatMult for strictly lower triangular part of StYfull matrix.  */

PetscErrorCode MatLowerTriangularMult(Mat B, Vec X, TriangularTypes tri_type)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscInt     lda, index;
  PetscScalar *x_array, Alpha = 1.0;
  PetscMemType memtype_r, memtype_x;
  MPI_Comm     comm = PetscObjectComm((PetscObject)B);

  const PetscScalar *r_array;
  
  PetscFunctionBegin;

  if (lmvm->k == 0) {
    PetscCall(VecZeroEntries(X)); 
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (lbfgs->idx_begin == -1) {
    index = 0;
  } else {
    index = lbfgs->idx_begin;
  }

  PetscCall(VecGetArrayWriteAndMemType(X, &x_array, &memtype_x));
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->StYfull, &r_array, &memtype_r));
  PetscAssert(memtype_x == memtype_r, comm, PETSC_ERR_PLIB, "Incompatible device pointers");
  /* We need four int for dimensions. 
   * C : [index-1 x index-1], strictly LT
   * D : [index x m-index] 
   * A : [m-index-1 x m-index-1], strictly LT */
  switch (tri_type) {
  case MAT_CDBFGS_LOWER_TRIANGULAR:
    switch (lbfgs->strategy) {
    case MAT_LBFGS_CD_REORDER:
      {        
        switch (memtype_x) {
        case PETSC_MEMTYPE_HOST:
          {
            PetscBLASInt m_blas, lda_blas, one = 1;
            PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscBLASIntCast(lda, &lda_blas));
            PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Normal", "NotUnitTriangular", &m_blas, &r_array[1], &lda_blas, x_array, &one));
            /* Shift */
            PetscCall(PetscArraymove(&x_array[1],x_array,lmvm->k));
            x_array[0] = 0;
          }
          break;
        case PETSC_MEMTYPE_CUDA:
        case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
          { 
            cublasHandle_t handle;
            PetscCuBLASInt m_blas, lda_blas, one = 1;

            PetscCall(PetscCUBLASGetHandle(&handle));
            PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
            PetscCallCUBLAS(cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_blas, &r_array[1], lda_blas, x_array, one));
            /* Shift */
            PetscCall(PetscArraymove(&x_array[1],x_array,lmvm->k));
            x_array[0] = 0;
          }
#endif
          break;
        case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
          {
            hipblasHandle_t handle;
            PetscHIPBLASInt m_blas, lda_blas, one = 1;

            PetscCall(PetscHIPBLASGetHandle(&handle));
            PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            PetscCall(PetscHIPBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
            PetscCallHIPBLAS(hipblasXtrmv(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, m_blas, &r_array[1], lda_blas, x_array, one));
            /* Shift */
            PetscCall(PetscArraymove(&x_array[1],x_array,lmvm->k));
            x_array[0] = 0;
          }
#endif
          break;
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
        }
        break;
      }
    case MAT_LBFGS_CD_INPLACE:
      {
        switch (memtype_x) {
        case PETSC_MEMTYPE_HOST:
          {
            PetscBLASInt m_blas, idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscBLASIntCast(lda, &lda_blas));
            PetscCall(PetscBLASIntCast(index, &idx_blas));
            PetscCall(PetscBLASIntCast(lda, &lda_blas));
            PetscCall(PetscBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscBLASIntCast(index - 1, &idx_n_1));
            }
            /* Lower Triangular Normal Case:
             * Below, C,A are Strictly LT, and D is rectangular.
             * [ C | D ] [y] => [ C y + D x ]
             * [ 0 | A ] [x]    [    A x    ] */
            /* Copy x for work */
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(lmvm->m - index, &buffer));
            PetscCall(PetscArraycpy(buffer, &x_array[index], lmvm->m-index));
    
            /* Applying A: x' = A x */
            if (index != lmvm->k) {
              PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Normal", "NotUnitTriangular", &diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], &lda_blas, &x_array[idx_blas], &one));
              PetscCall(PetscArraymove(&x_array[idx_blas+1], &x_array[idx_blas], lmvm->m-index-1));
            }
            x_array[idx_blas] = 0;
    
            /* Applying C: buffer2 = C y */
            if (index > 1) {
              PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Normal", "NotUnitTriangular", &idx_n_1, &r_array[1], &lda_blas, x_array, &one));
              PetscCall(PetscArraymove(&x_array[1], x_array, index-1));
            }
            x_array[0] = 0;
    
            /* Applying D: buffer2 =  D x */
            if (index != 0) {
              PetscCallBLAS("BLASgemv", BLASgemv_("N", &idx_blas, &diff_blas, &Alpha, &r_array[idx_blas*lda], &lda_blas, buffer, &one, &Alpha, x_array, &one));
            }
            PetscCall(PetscFree(buffer));
          }
          break;
        case PETSC_MEMTYPE_CUDA:
        case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
          {
            cublasHandle_t handle;

            PetscCall(PetscCUBLASGetHandle(&handle));
            PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

            PetscCuBLASInt m_blas, idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
            PetscCall(PetscCuBLASIntCast(index, &idx_blas));
            PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
            PetscCall(PetscCuBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscCuBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscCuBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscCuBLASIntCast(index - 1, &idx_n_1));
            }
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(lmvm->m - index, &buffer));
            PetscCall(PetscArraycpy(buffer, &x_array[index], lmvm->m-index));
            if (index != lmvm->k) {
              PetscCallCUBLAS(cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], lda_blas, &x_array[idx_blas], one));
              PetscCall(PetscArraymove(&x_array[idx_blas+1], &x_array[idx_blas], lmvm->m-index-1));
            }
            x_array[idx_blas] = 0;
            if (index > 1) {
              PetscCallCUBLAS(cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, idx_n_1, &r_array[1], lda_blas, x_array, one));
              PetscCall(PetscArraymove(&x_array[1], x_array, index-1));
            }
            x_array[0] = 0;
            if (index != 0) {
              PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_N, idx_blas, diff_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, buffer, one, &Alpha, x_array, one));
            }
            PetscCall(PetscFree(buffer));
          }
#endif
          break;
        case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
          {
            hipblasHandle_t handle;

            PetscCall(PetscHIPBLASGetHandle(&handle));
            PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

            PetscHIPBLASInt m_blas, idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(PetscHIPBLASIntCast(lmvm->k, &m_blas));
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
            PetscCall(PetscHIPBLASIntCast(index, &idx_blas));
            PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
            PetscCall(PetscHIPBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscHIPBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscHIPBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscHIPBLASIntCast(index - 1, &idx_n_1));
            }
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(lmvm->m - index, &buffer));
            PetscCall(PetscArraycpy(buffer, &x_array[index], lmvm->m-index));
            if (index != lmvm->k) {
              PetscCallHIPBLAS(hipblasXtrmv(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], lda_blas, &x_array[idx_blas], one));
              PetscCall(PetscArraymove(&x_array[idx_blas+1], &x_array[idx_blas], lmvm->m-index-1));
            }
            x_array[idx_blas] = 0;
            if (index > 1) {
              PetscCallHIPBLAS(hipblasXtrmv(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, idx_n_1, &r_array[1], lda_blas, x_array, one));
              PetscCall(PetscArraymove(&x_array[1], x_array, index-1));
            }
            x_array[0] = 0;
            if (index != 0) {
              PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_N, idx_blas, diff_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, buffer, one, &Alpha, x_array, one));
            }
            PetscCall(PetscFree(buffer));
          }
#endif
          break;
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
        }
      }
      break; 
    case MAT_LBFGS_BASIC:
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
    }
    break;
  case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
    switch (lbfgs->strategy) {
    case MAT_LBFGS_CD_REORDER:
      {
        PetscBLASInt m_blas, lda_blas, one = 1;
        PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
        PetscCall(PetscBLASIntCast(lda, &lda_blas));
        PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
        PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Transpose", "NotUnitTriangular", &m_blas, &r_array[1], &lda_blas, x_array, &one));
        x_array[lmvm->k] = 0;
      }
      break;
    case MAT_LBFGS_CD_INPLACE:
      {
        switch (memtype_x) {
        case PETSC_MEMTYPE_HOST:
          {
            PetscBLASInt idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscBLASIntCast(index, &idx_blas));
            PetscCall(PetscBLASIntCast(lda, &lda_blas));
            PetscCall(PetscBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscBLASIntCast(index - 1, &idx_n_1));
            }
            /* Lower Triangular Transpose Case:
             * Below, C,A are Strictly LT, and D is rectangular.
             * [ C^T |  0  ] [y] => [     C^T y     ]
             * [ D^T | A^T ] [x]    [ D^T y + A^T x ] */
            /* Copy  y */
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(index, &buffer));
            PetscCall(PetscArraycpy(buffer, x_array, index));
            /* Applying C: y' = C^T y */
            if (index > 1) {
              PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Transpose", "NotUnitTriangular", &idx_n_1, &r_array[1], &lda_blas, x_array, &one));
              x_array[index-1] = 0;
            }
            /* Applying A^T: x' = A^T x */
            if (index != lmvm->k) {
              PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Transpose", "NotUnitTriangular", &diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], &lda_blas, &x_array[idx_blas], &one));
              x_array[lmvm->k] = 0;
            }
            /* Applying D^T: y' = y' + D^T x */
            if (index != 0) {
              PetscCallBLAS("BLASgemv", BLASgemv_("T", &diff_blas, &idx_blas, &Alpha, &r_array[idx_blas*lda], &lda_blas, &x_array[idx_blas], &one, &Alpha, x_array, &one));
              PetscCallBLAS("BLASgemv", BLASgemv_("N", &idx_blas, &diff_blas, &Alpha, &r_array[idx_blas*lda], &lda_blas, buffer, &one, &Alpha, x_array, &one));
            }
          }
          break;
        case PETSC_MEMTYPE_CUDA:
        case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
          {
            cublasHandle_t handle;

            PetscCall(PetscCUBLASGetHandle(&handle));
            PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

            PetscCuBLASInt idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscCuBLASIntCast(index, &idx_blas));
            PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
            PetscCall(PetscCuBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscCuBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscCuBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscCuBLASIntCast(index - 1, &idx_n_1));
            }
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(index, &buffer));
            PetscCall(PetscArraycpy(buffer, x_array, index));
            if (index > 1) {
              PetscCallCUBLAS(cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, idx_n_1, &r_array[1], lda_blas, x_array, one));
              x_array[index-1] = 0;
            }
            if (index != lmvm->k) {
              PetscCallCUBLAS(cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], lda_blas, &x_array[idx_blas], one));
              x_array[lmvm->k] = 0;
            }
            if (index != 0) {
              PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_T, diff_blas, idx_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
              PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_N, idx_blas, diff_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, buffer, one, &Alpha, x_array, one));
            }
          }
#endif
          break;
        case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
          {
            hipblasHandle_t handle;

            PetscCall(PetscHIPBLASGetHandle(&handle));
            PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

            PetscHIPBLASInt idx_blas, lda_blas, idx_n_1, diff_blas, diff_blas_n_1, one = 1;
            PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
            PetscCall(PetscHIPBLASIntCast(index, &idx_blas));
            PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
            PetscCall(PetscHIPBLASIntCast(lda - index, &diff_blas));
            PetscCall(PetscHIPBLASIntCast(lda - index - 1, &diff_blas_n_1));
            if (index == 0 ) {
              PetscCall(PetscHIPBLASIntCast(0, &idx_n_1));
            } else {
              PetscCall(PetscHIPBLASIntCast(index - 1, &idx_n_1));
            }
            PetscScalar *buffer;
            PetscCall(PetscCalloc1(index, &buffer));
            PetscCall(PetscArraycpy(buffer, x_array, index));
            if (index > 1) {
              PetscCallHIPBLAS(hipblasXtrmv(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, idx_n_1, &r_array[1], lda_blas, x_array, one));
              x_array[index-1] = 0;
            }
            if (index != lmvm->k) {
              PetscCallHIPBLAS(hipblasXtrmv(handle, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, diff_blas_n_1, &r_array[idx_blas*(lda_blas+1)+1], lda_blas, &x_array[idx_blas], one));
              x_array[lmvm->k] = 0;
            }
            if (index != 0) {
              PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_T, diff_blas, idx_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
              PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_N, idx_blas, diff_blas, &Alpha, &r_array[idx_blas*lda], lda_blas, buffer, one, &Alpha, x_array, one));
            }
          }
#endif
          break;
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
        }
      }
      break;
    case MAT_LBFGS_BASIC:
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
    }
    break;
  case MAT_CDBFGS_UPPER_TRIANGULAR:
  case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
  default:          
    SETERRQ(comm, PETSC_ERR_SUP, "This routine is only for lower triangular matrices.");
  }
  PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->StYfull, &r_array));
  PetscCall(VecRestoreArrayWriteAndMemType(lbfgs->rwork2, &x_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves triangular matrix, stored in either recycled order, re rewritten regular order.
 * One can solve for lower triangle, transpose of lower triangle, 
 * upper triangle, and transpoe of upper triangle. 
 * It assumes the input matrix is square matrix, n x n.
 *
 * Recycled order: 
 *
 * StY matrix is S^T Y. In a inner product form, it would look like
 *
 * [ <s_1,y_1> | <s_1,y_2> | <s_1,y_3> ]
 * [ <s_2,y_1> | <s_2,y_2> | <s_2,y_3> ]
 * [ <s_3,y_1> | <s_3,y_2> | <s_3,y_3> ].
 *
 * However, in recycled, in_place method, instead of overwriting
 * S and Y matrices whenever we update s,y vectors, we recycle them: 
 * 
 * old S^T : [s_1, s_2, s_3], and we want to remove s_1, and add s_4. 
 * Here, subscript j is a according s vector from j-th iteration.
 *
 * Rewrite: [s_2, s_3, s_4].  Recycled: [s_4, s_2, s_3]
 *
 * This would turn S^T Y matrix to a following form:
 *
 * [ <s_4,y_4> | <s_4,y_2> | <s_4,y_3> ]
 * [ <s_2,y_4> | <s_2,y_2> | <s_2,y_3> ]
 * [ <s_3,y_4> | <s_3,y_2> | <s_3,y_3> ]. 
 *                                                            */

static PetscErrorCode MatSolveTriangular(Mat B, Mat R, PetscInt lowest_index, Vec x, TriangularTypes tri_type)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  MPI_Comm     comm  = PetscObjectComm((PetscObject)R);
  PetscScalar  Alpha = 1.0, neg_one = -1.;
  PetscMemType memtype_r, memtype_x;
  PetscScalar *x_array;
  PetscInt     lda, M = 0;
  PetscMPIInt rank;
  const PetscScalar *r_array;

  PetscFunctionBegin;
  PetscCall(MPI_Comm_rank(comm, &rank));

  PetscCall(MatGetLocalSize(R, &M, NULL));
  PetscCall(MatDenseGetArrayReadAndMemType(R, &r_array, &memtype_r));
  PetscCall(VecGetArrayAndMemType(x, &x_array, &memtype_x));
  if (M == 0) {
    PetscCall(VecRestoreArrayAndMemType(x, &x_array));
    PetscCall(MatDenseRestoreArrayReadAndMemType(R, &r_array));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  // TODO rank non zero exits, but rank 0 goes on 
  // Inside get arrays, there is ValidLogical thing, which has Allreduce
  // as non rank 0 never gets there, it stalls..
  PetscCall(MatDenseGetLDA(R, &lda));
  //PetscAssert(memtype_x == memtype_r, comm, PETSC_ERR_PLIB, "Incompatible device pointers");


  switch (lbfgs->strategy) {
  case MAT_LBFGS_CD_REORDER:
    {
      switch (memtype_r) {
      case PETSC_MEMTYPE_HOST:
        /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
        {
          //PetscAssert(PetscDefined(BLAS)...));
          PetscBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscBLASIntCast(lmvm->k+1, &m_blas));
          PetscCall(PetscBLASIntCast(lda, &lda_blas));
          PetscBLASInt ldb_blas = lda_blas;
          switch (tri_type) {
          case MAT_CDBFGS_UPPER_TRIANGULAR:
            PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
            PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR:
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          default:
            SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
          }
        }
        break;
      case PETSC_MEMTYPE_CUDA:
      case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
        {
          cublasHandle_t handle;
          PetscDeviceContext dctx;

          PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
          PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, &handle));

          PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

          PetscCuBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscCuBLASIntCast(lmvm->k+1, &m_blas));
          PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
          PetscCuBLASInt ldb_blas = lda_blas;
          switch (tri_type) {
          case MAT_CDBFGS_UPPER_TRIANGULAR:
            PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
            break;
          case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
            PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR:
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          default:
            SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
          }
        }
#endif
        break;
      case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
        {
          hipblasHandle_t handle;

          PetscCall(PetscHIPBLASGetHandle(&handle));
          PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

          PetscHIPBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscHIPBLASIntCast(lmvm->k+1, &m_blas));
          PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
          PetscHIPBLASInt ldb_blas = lda_blas;
          switch (tri_type) {
          case MAT_CDBFGS_UPPER_TRIANGULAR:
            PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, m_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
            break;
          case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
            PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, m_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR:
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          default:
            SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
          }
        }
#endif
        break;
      default:
        SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
      }
    }
    break;
  case MAT_LBFGS_CD_INPLACE:
    switch (memtype_r) {
    case PETSC_MEMTYPE_HOST:
      {
        //PetscAssert(PetscDefined(BLAS)...));
        PetscBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscBLASIntCast(lda, &lda_blas));
        PetscCall(PetscBLASIntCast(lmvm->k + 1 - lowest_index, &diff_blas));
        PetscBLASInt ldb_blas = lda_blas;

        switch (tri_type) {
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          /* Upper Triangular Case: 
           * A : [diff_blas x diff_blas], B : [diff_blas x idx_blas], C : [idx_blas x idx_blas], D : [idx_blas x diff_blas]
           * Reorder    ->  Memory 
           * [ A | B ]  ->  [ C | D ] 
           * [ D | C ]      [ B | A ] 
           * Below, C,A are UT.
           * [ C | 0 ]^-1 [y] => [C^-1 y          ]
           * [ B | A ]    [x]    [A^-1(x- BC^-1 y)] */
          //TODO for number of rows, does it have to be local, or is global size fine?
          /* Applying C: y' = C^-1 y */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
          /* Applying B: x' = x - BC^-1 y' */
          PetscCallBLAS("BLASgemv", BLASgemv_("N",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying A: x' = A^-1 (x - BC^-1 y) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(lda_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          /* Upper Triangular Transpose Case: 
           * Below, C,A are UT.
           * [ C | 0 ]^-T [y] => [C^-T(y - B^T A^-T x)]
           * [ B | A ]    [x]    [A^-T x              ] */
          /* Applying A: x' = A^-T x */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(lda_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          /* Applying B: y' = y - B^T A^-T x */
          PetscCallBLAS("BLASgemv", BLASgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, &x_array[idx_blas], &one, &Alpha, x_array, &one));
          /* Applying C: y' = C^-T (y - B^T A^-T x) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
        }
      }
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
      {
        cublasHandle_t handle;
        PetscDeviceContext dctx;


        PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
        PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, &handle));

        //PetscAssert(PetscDefined(BLAS)...));
        PetscCuBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscCuBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
        PetscCall(PetscCuBLASIntCast(lmvm->k + 1 - lowest_index, &diff_blas));
        PetscCuBLASInt ldb_blas = lda_blas;

        switch (tri_type) {
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, idx_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_N, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, x_array, one, &Alpha, &x_array[idx_blas], one));
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, diff_blas, one, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, diff_blas, one, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_T, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, idx_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
        }
      }
#endif
      break;
    case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
      {
        hipblasHandle_t handle;

        PetscCall(PetscHIPBLASGetHandle(&handle));
        PetscCallHIPBLAS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        //PetscAssert(PetscDefined(BLAS)...));
        PetscHIPBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscHIPBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscHIPBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
        PetscCall(PetscHIPBLASIntCast(lmvm->k + 1 - lowest_index, &diff_blas));
        PetscHIPBLASInt ldb_blas = lda_blas;

        switch (tri_type) {
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, idx_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_N, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, x_array, one, &Alpha, &x_array[idx_blas], one));
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, diff_blas, one, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, diff_blas, one, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_T, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, idx_blas, one, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
        default:
          SETERRQ(comm, PETSC_ERR_SUP, "MatSolveTriangular is only for Upper Triangular Matrices.");
        }
      }
#endif
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented TRSM");
    }
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
  }
  PetscCall(VecRestoreArrayAndMemType(x, &x_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(R, &r_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Truncates vector and zeros the entries.
 * When Reset happens, only indices are reset'd, but data is still preserved.
 * In this case, we need to zero out all the "corrupted data" in resulting vectors. */

static PetscErrorCode Vec_Truncate(Mat H, Vec X)
{
  Mat_LMVM *lmvm  = (Mat_LMVM*)H->data;

  PetscInt i, N;
  MPI_Comm     comm  = PetscObjectComm((PetscObject)H);

  PetscFunctionBegin;

  if (lmvm->k == lmvm->m - 1){
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    PetscMemType memtype_x;
    PetscScalar *x_array;

    PetscCall(VecGetSize(X,&N));
    PetscCall(VecGetArrayWriteAndMemType(X, &x_array, &memtype_x));
    switch (memtype_x) {
    case PETSC_MEMTYPE_HOST:
      {
        for (i=lmvm->k+1; i<lmvm->m; i++){ x_array[i] = 0; }
      }
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
    case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
      {
        PetscDeviceContext dctx;

        PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
        PetscCall(PetscDeviceRegisterMemory(x_array, memtype_x, N*sizeof(*x_array)));
        if (lmvm->k != lmvm->m -1) {
          PetscCall(PetscDeviceArrayZero(dctx, &x_array[lmvm->k+1], lmvm->m - lmvm->k -1));
        }
      }
#endif
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
    }
    PetscCall(VecRestoreArrayAndMemType(X, &x_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Adds LDL^T to J mat */

static PetscErrorCode MatAdd_LDLT(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
//  MPI_Comm    comm  = PetscObjectComm((PetscObject)B);

  const PetscScalar *r_array;
  PetscScalar       *x_array, *buffer;
  PetscMemType       x_type, r_type;
  PetscInt           i, j, k, query_idx_i, query_idx_j, query_idx_k, index;
  MPI_Comm           comm = PetscObjectComm((PetscObject)B);
  Vec                workvec1, workvec2;

  PetscFunctionBegin;

  if (lbfgs->idx_begin == -1) {
    index = 0;
  } else {
    index = lbfgs->idx_begin;
  }
  /* L D^{-1} L^T :  (L_i is ith column of strictly low tri mat. Below, multiply is pointwise mult.
   * [ 0 | L_0*L_0[1]/d_0 | L_0*L_0[2]/d_0 + L_1*L_1[2]/d_1 | ... ].  
   * 
   * Struture is similar for inplace version, but just two clockwise shifts in block-form.            */
  PetscCall(VecGetArrayReadAndMemType(lbfgs->diag_vec, &r_array, &r_type));
  for (i=0; i<lmvm->m-1; i++) {
    query_idx_i = (index + i) % lmvm->m;
    PetscCall(MatDenseGetColumnVecRead(lbfgs->StYfull, query_idx_i, &workvec1));

    /* Copying to emulate strictly lower triangular */
    PetscCall(VecCopy(workvec1, lbfgs->rwork1));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->StYfull, query_idx_i, &workvec1));
    PetscCall(VecGetArrayAndMemType(lbfgs->rwork1, &x_array, &x_type));
    switch (x_type) {
    case PETSC_MEMTYPE_HOST:
    {
      for (j=0; j<i+1; j++) {
        query_idx_j = (index + j) % lmvm->m;
        x_array[query_idx_j] = 0;
      }

      /* Creating array for scale = L_i[i+1]/d_0 */
      PetscCall(PetscCalloc1(lmvm->m-i-1, &buffer));
      for (j=0; j < lmvm->m-i-1; j++) {
        query_idx_j = query_idx_i+j+1  % lmvm->m;
        if (r_array[query_idx_i] != 0) {
          buffer[j] = x_array[query_idx_j]/r_array[query_idx_i];
        } else {
          buffer[j] = 0;
        }
      }
      for (j=0, k=i+1; k<lmvm->m; k++, j++) {
        query_idx_j = (index + j) % lmvm->m;
        query_idx_k = (index + k) % lmvm->m;
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
        PetscCall(VecAXPY(workvec2, buffer[j], lbfgs->rwork1));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
      }
      PetscCall(PetscFree(buffer));
    }
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
    case PETSC_MEMTYPE_HIP:
    {
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
      PetscDeviceContext dctx;
      PetscInt           N;
      PetscCall(VecGetSize(lbfgs->rwork1,&N));
      PetscCall(PetscDeviceContextGetCurrentContext(&dctx));//TODO async thing here?

      PetscCall(PetscDeviceRegisterMemory(x_array, x_type, N*sizeof(*x_array)));
      PetscCall(PetscDeviceRegisterMemory(r_array, r_type, N*sizeof(*x_array)));

      for (j=0; j<i+1; j++) {
        query_idx_j = (index + j) % lmvm->m;
        PetscCall(PetscDeviceArrayZero(dctx, &x_array[query_idx_j],1));
      }

      PetscCall(PetscDeviceCalloc(dctx, x_type, lmvm->m-i-1, &buffer));
      PetscCall(PetscDeviceRegisterMemory(buffer, x_type, (lmvm->m-i-1)*sizeof(*buffer)));
      for (j=0; j < lmvm->m-i-1; j++) {
        query_idx_j = query_idx_i+j+1  % lmvm->m;
        /* TODO 
         * 1. host alloc size 1, and devicearraycpy r[idx_i]
         * 2. do the same for tmp1, tmp2?
         * 3. buffer[j] at the end */

        PetscScalar *r_q_i;
        PetscCall(PetscMalloc1(1, &r_q_i));
        PetscCall(PetscDeviceRegisterMemory(r_q_i, PETSC_MEMTYPE_HOST, sizeof(*r_q_i)));
        PetscCall(PetscDeviceArrayCopy(dctx, r_q_i, &r_array[query_idx_i], 1));
        if (r_q_i != 0) {//TODO can't do this..
          // TODO maybe create vector with x[j]/r[i], copy, and set zero elsewhere?
          PetscScalar *tmp1, *tmp2;
          PetscCall(PetscMalloc2(1, &tmp1, 1, &tmp2));
          PetscCall(PetscDeviceRegisterMemory(tmp1, PETSC_MEMTYPE_HOST, sizeof(*tmp1)));
          PetscCall(PetscDeviceRegisterMemory(tmp2, PETSC_MEMTYPE_HOST, sizeof(*tmp2)));
          PetscCall(PetscDeviceArrayCopy(dctx, tmp1, &x_array[query_idx_i], 1));
          PetscCall(PetscDeviceArrayCopy(dctx, tmp2, &r_array[query_idx_i], 1));
          *tmp1 /= *tmp2;
          PetscCall(PetscDeviceArrayCopy(dctx, &buffer[j], tmp1, 1));
          PetscCall(PetscFree2(tmp1, tmp2));
        } else {
          PetscCall(PetscDeviceArrayZero(dctx, &buffer[j],1));
        }
      }
      for (j=0, k=i+1; k<lmvm->m; k++, j++) {
        PetscScalar *b_j;
        query_idx_j = (index + j) % lmvm->m;
        query_idx_k = (index + k) % lmvm->m;
        PetscCall(PetscMalloc1(1, &b_j));
        PetscCall(PetscDeviceRegisterMemory(b_j, PETSC_MEMTYPE_HOST, sizeof(*b_j)));
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
        PetscCall(PetscDeviceArrayCopy(dctx, &b_j, &buffer[j], 1));
        PetscCall(VecAXPY(workvec2, *b_j, lbfgs->rwork1));
        PetscCall(PetscFree(b_j));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
      }
      PetscCall(PetscDeviceFree(dctx, buffer));
#endif
    }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented MEMTYPE");
    }
    PetscCall(VecRestoreArrayAndMemType(lbfgs->rwork1, &x_array));
  }
  PetscCall(VecRestoreArrayReadAndMemType(lbfgs->diag_vec, &r_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Solves for
 * [ I | S R^{-T} ] [   I  | 0 ] [ H_0 | 0 ] [ I | -Y ] [     I      ]
 *                  [ -Y^T | I ] [  0  | D ] [ 0 |  I ] [ R^{-1} S^T ]  */

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  Vec rwork1 = lbfgs->rwork1;
  Vec rwork2 = lbfgs->rwork2;
  Vec rwork1_host =  lbfgs->bind ? lbfgs->rwork1_host : lbfgs->rwork1;
  Vec rwork2_host =  lbfgs->bind ? lbfgs->rwork2_host : lbfgs->rwork2;

  PetscInt index;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatSolve, H, F, dX,0));
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (lmvm->k == -1) {
    PetscCall(MatCDBFGSApplyJ0Inv(H, F, dX));
    PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscDeviceContext dctx;
  PetscMPIInt rank;

  MPI_Comm     comm = PetscObjectComm((PetscObject)H);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(MPI_Comm_rank(comm, &rank));

  if (lbfgs->idx_begin == -1) {
    index = 0;
  } else {
    index = lbfgs->idx_begin;
  }

  /* Start with reusable part: rwork1 = R^-1 S^T F */
  //STYFull: host
  PetscCall(MatMultTranspose(lbfgs->Sfull, F, rwork1));
  PetscCall(VecCopyAsync_Private(rwork1, rwork1_host, dctx));

  /* Reordering rwork1, as STY is in canonical order, while S is in recycled order */
  PetscCall(VecRightward_Shift(H, rwork1_host, -lbfgs->idx_rplc));
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, index, rwork1_host, MAT_CDBFGS_UPPER_TRIANGULAR));
  PetscCall(VecRightward_Shift(H, rwork1_host, lbfgs->idx_rplc));

  /* lwork1 :H_0 (F - Y R^{-1} S^T X) */
  /* dX : H_0 ( F + (Y (-R^{-1} S^T X)) */
  /* dX : H_0 ( F + rwork ) */
  PetscCall(VecScaleAsync_Private(rwork1_host, -1.0, dctx));

  if (lbfgs->bind) {
    PetscCall(VecCopy(rwork1_host, rwork1));
  }

  PetscCall(VecCopyAsync_Private(F, lbfgs->lwork1, dctx));
  PetscCall(MatMultAdd(lbfgs->Yfull, rwork1, lbfgs->lwork1, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->lwork1, dX));

  /* -S R^{-T} ( Y^T lwork1 - D rwork1 ) */
  if (lbfgs->bind) {
    PetscCall(VecCopy(rwork1, rwork1_host));
  }
  PetscCall(VecPointwiseMultAsync_Private(rwork1_host, lbfgs->diag_vec, rwork1_host, dctx));

  if (lbfgs->bind) {
    PetscCall(MatMultTranspose(lbfgs->Yfull, dX, rwork2));
    PetscCall(VecCopy(rwork2, rwork2_host));
    PetscCall(VecAXPY(rwork1_host, 1.0, rwork2_host));
  } else {
    PetscCall(MatMultTransposeAdd(lbfgs->Yfull, dX, rwork1, rwork1_host));
  }

  /* Reordering rwork2, as STY is in canonical order, while S is in recycled order */
  PetscCall(VecRightward_Shift(H, rwork1_host, -lbfgs->idx_rplc));
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, index, rwork1_host, MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE));
  PetscCall(VecRightward_Shift(H, rwork1_host, lbfgs->idx_rplc));
  PetscCall(VecScaleAsync_Private(rwork1_host, -1.0, dctx));
  if (lbfgs->bind) {
    PetscCall(VecCopy(rwork1_host, rwork1));
  }
  PetscCall(MatMultAdd(lbfgs->Sfull, rwork1, dX, dX));
  PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
 * B_0 - [ Y | B_0 S] [ -D  |    L^T    ]^-1 [   Y^T   ]
 *                    [  L  | S^T B_0 S ]    [ S^T B_0 ]
 * Above is equivalent to
 *
 * B_0 - [ Y | B_0 S] [ -D^{1/2} | D^{-1/2} L^T ]^{-1} [   D^{1/2}   | 0 ]^{-1} [   Y^T   ]
 *                    [    0     |      J^T     ]      [ -L D^{-1/2} | J ]      [ S^T B_0 ]
 *
 * becomes
 *
 * B_0 - [ Y | B_0 S ] [ -D^{-1/2} | D^{-1} L^T J^{-T} ] [     D^{-1/2}    |        ] [   Y^T   ]
 *                     [           |        J^{-T}     ] [ J^{-1} L D^{-1} | J^{-1} ] [ S^T B_0 ]
 *
 * where J J^T = S^T B_0 S + L D^{-1} L^T. J exsits and is non singular.
 *
 * Byrd, Nocedal, Schnabel 1994                                            */

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  IS  temp_is;
  Vec temp_1, temp_2;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatMult, B, X, Z,0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (lmvm->k == -1) {
    PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  if (lbfgs->chol_ldlt_lazy) {
    /* Cholesky, and LDLT is done lazily to avoid unncessary computation, in case MatMult is not so frequently used *	 
     * Now, SBS is in shifted order in all strategies. *
     * Compute S^T B S + L D^{-1} L^T *
     * J = S^T B S + L D^{-1} L^T */
    PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->BS, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->J));
    /* Adds L D L^T to J matrix */
    PetscCall(MatAdd_LDLT(B));

    /* Cholesky factorization */
    if (lmvm->k == lmvm->m - 1) {
      PetscCall(MatDestroy(&lbfgs->J_solve));
      PetscCall(MatConvert(lbfgs->J, MATSAME, MAT_INITIAL_MATRIX, &lbfgs->J_solve));
      PetscCall(MatSetOption(lbfgs->J_solve, MAT_SPD, PETSC_TRUE));
      PetscCall(MatCholeskyFactor(lbfgs->J_solve,NULL,NULL));
    } else {
      PetscCall(MatDenseGetSubMatrix(lbfgs->J, 0, lmvm->k+1, 0, lmvm->k+1, &lbfgs->J_work));
      PetscCall(MatDestroy(&lbfgs->J_solve));
      PetscCall(MatDuplicate(lbfgs->J_work, MAT_COPY_VALUES, &lbfgs->J_solve));
      PetscCall(MatSetOption(lbfgs->J_solve, MAT_SPD, PETSC_TRUE));
      PetscCall(MatCholeskyFactor(lbfgs->J_solve,NULL,NULL));
      PetscCall(MatDenseRestoreSubMatrix(lbfgs->J, &lbfgs->J_work));
    }
    lbfgs->chol_ldlt_lazy = PETSC_FALSE;
  }    
  /* Apply Phi^T = [S^TB; Y^t] to incoming vector X */
  /* The result is stored in two halves, (rwork4 = S^T B X) and (rwork3 = Y^T X) */
  PetscCall(MatMultTranspose(lbfgs->Sfull, Z, lbfgs->rwork4));
  PetscCall(MatMultTranspose(lbfgs->Yfull, X, lbfgs->rwork3));
  PetscCall(Vec_Truncate(B,lbfgs->rwork3));
  PetscCall(Vec_Truncate(B,lbfgs->rwork4));

  /* Common part: rwork1:  J^{-T} J^{-1} (L D^{-1} Y^T X + S^T B_0 X) */
  PetscCall(VecPointwiseDivide(lbfgs->rwork2,lbfgs->rwork3, lbfgs->diag_vec));
  PetscCall(MatLowerTriangularMult(B, lbfgs->rwork2, MAT_CDBFGS_LOWER_TRIANGULAR));
  PetscCall(Vec_Truncate(B,lbfgs->rwork2));
  PetscCall(VecAXPY(lbfgs->rwork2, 1., lbfgs->rwork4));

  PetscCall(ISCreateStride(PETSC_COMM_WORLD, lmvm->k+1, 0, 1, &temp_is));
  PetscCall(VecGetSubVector(lbfgs->rwork1, temp_is, &temp_1));
  PetscCall(VecGetSubVector(lbfgs->rwork2, temp_is, &temp_2));

  PetscCall(MatSolve(lbfgs->J_solve, temp_2, temp_1));
  PetscCall(VecRestoreSubVector(lbfgs->rwork1, temp_is, &temp_1));
  PetscCall(VecRestoreSubVector(lbfgs->rwork2, temp_is, &temp_2));
  PetscCall(Vec_Truncate(B,lbfgs->rwork1));

  /* Bottom part: - B_0 S rwork1 */
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork1, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(Z, -1., lbfgs->lwork2));

  /* Top part: + Y D^{-1} ( Y^T X - D^{-1} L^T rwork1 ) */
  PetscCall(MatLowerTriangularMult(B, lbfgs->rwork1, MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE));
  PetscCall(Vec_Truncate(B,lbfgs->rwork1));
  PetscCall(VecAXPY(lbfgs->rwork3, -1., lbfgs->rwork1));
  PetscCall(VecPointwiseDivide(lbfgs->rwork4, lbfgs->rwork3, lbfgs->diag_vec));
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork4, lbfgs->lwork1));
  PetscCall(VecAXPY(Z, 1., lbfgs->lwork1));
  PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts R[1:end,1:end] to R[0:end-1, 0:end-1] */

static PetscErrorCode MatMove_LR3(Mat H, Mat R)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscInt     M, lda;
  PetscMPIInt  rank;
  MPI_Comm     comm = PetscObjectComm((PetscObject)R);
  Mat          mat_local, local_sub;

  PetscFunctionBegin;

  PetscCall(MPI_Comm_rank(comm, &rank));

  PetscCall(MatGetLocalSize(R, &M, NULL));
  if (M == 0) PetscFunctionReturn(PETSC_SUCCESS);
  
  PetscCall(MatDenseGetLocalMatrix(R, &mat_local));
  PetscCall(MatDenseGetLDA(mat_local, &lda));

  PetscCall(MatDenseGetSubMatrix(mat_local, 1, lmvm->m, 1, lmvm->m, &local_sub));
  if (!lbfgs->temp_mat) PetscCall(MatDuplicate(local_sub, MAT_COPY_VALUES, &lbfgs->temp_mat));
  else PetscCall(MatCopy(local_sub, lbfgs->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));

  PetscCall(MatDenseGetSubMatrix(mat_local, 0, lmvm->m-1, 0, lmvm->m-1, &local_sub));
  PetscCall(MatCopy(lbfgs->temp_mat, local_sub, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fills last row of STY matrix with Y^T @ Xprev  */

static PetscErrorCode FillLastRow(Mat R, Mat STY)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)R->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  const PetscScalar *x_array;
  PetscScalar *r_array;
  PetscInt     M, i, lda, M_vec;
  PetscMemType memtype_r, memtype_x;
  MPI_Comm     comm = PetscObjectComm((PetscObject)R);
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCall(MPI_Comm_rank(comm, &rank));

  PetscCall(MatMultTranspose(lbfgs->Yfull, lmvm->Xprev, lbfgs->rwork1));
  PetscCall(VecRightward_Shift(R, lbfgs->rwork1, -lbfgs->idx_rplc));

  PetscCall(MatGetLocalSize(STY, &M, NULL));
  PetscCall(VecGetLocalSize(lbfgs->rwork1, &M_vec));
  PetscCall(VecGetArrayReadAndMemType(lbfgs->rwork1, &x_array, &memtype_x));
  PetscCall(MatDenseGetArrayWriteAndMemType(STY, &r_array, &memtype_r));
  if (M == 0) {
    PetscCall(MatDenseRestoreArrayAndMemType(STY, &r_array));
    PetscCall(VecRestoreArrayReadAndMemType(lbfgs->rwork1, &x_array));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MatDenseGetLDA(STY, &lda));

  /* i: col, j: row. Not caring about setting zero at bottom of col since 
   * we only care about upper triangular part */
  for (i=0; i< lmvm->m-1; i++) {
    switch (memtype_r) {
    case PETSC_MEMTYPE_HOST:
      {
        r_array[lda*(i+1)-1] = x_array[i];
      }
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
    case PETSC_MEMTYPE_HIP:
      {
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
        PetscDeviceContext dctx;

        PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
        PetscCall(PetscDeviceRegisterMemory(r_array, memtype_r, M*M*sizeof(*r_array)));
        PetscCall(PetscDeviceRegisterMemory(x_array, memtype_x, M_vec*sizeof(*x_array)));
        PetscCall(PetscDeviceArrayCopy(dctx, &r_array[lda*(i+1)-1], &x_array[i], 1));
#endif
      }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented MEMTYPE");
    }
  }

  PetscCall(MatDenseRestoreArrayAndMemType(STY, &r_array));
  PetscCall(VecRestoreArrayReadAndMemType(lbfgs->rwork1, &x_array));
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
  PetscInt           N, n, low, high, i;
  //MPI_Comm           comm = PetscObjectComm((PetscObject)B);
  Vec                workvec1, workvec2;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (lmvm->prev_set) {
    Vec FX[2];
    PetscScalar dotFX[2];
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPXAsync_Private(lmvm->Xprev, -1.0, X, dctx));
    PetscCall(VecAYPXAsync_Private(lmvm->Fprev, -1.0, F, dctx));
    /* Test if the updates can be accepted */
    FX[0] = lmvm->Fprev;
    FX[1] = lmvm->Xprev;
    PetscCall(VecMDot(lmvm->Xprev, 2, FX, dotFX));
    curvature = dotFX[0];
    ststmp = dotFX[1];
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      PetscInt  StYidx;

      /* LMVM is updated. Need to update Chol and LDLT inside MatMult */
      lbfgs->chol_ldlt_lazy = PETSC_TRUE;
      lbfgs->iter_count++;//TODO for debugging purpose. delete later
      lmvm->nupdates++;

      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetSize(lmvm->Xprev, &N));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));

      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        PetscCall(MatLMVMUpdate(lbfgs->diag_bfgs, X, F));
      }

      lbfgs->idx_rplc = (lbgs->idx_rplc + 1) % lmvm->m;
      if (lmvm->k != lmvm->m-1) {
        lmvm->k = lmvm->k + 1;
      } else if (lbfgs->strategy == MAT_LBFGS_CD_REORDER) {
        PetscCall(MatMove_LR3(B, lbfgs->StYfull));
      }


      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Sfull, lbfgs->idx_rplc, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Xprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Sfull, lbfgs->idx_rplc, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Fprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec1));

      StYidx = (lbfgs->strategy == MAT_LBFGS_CD_REORDER) ?  lmvm->k : lbfgs->idx_cols;

      {
        Vec this_sy_col;

        PetscCall(MatDenseGetColumnVecWrite(lbfgs->StYfull, StYidx, &this_sy_col));
        PetscCall(MatMultTranspose(lbfgs->Sfull, lmvm->Fprev, this_sy_col));
        if ((lmvm->k == lmvm->m - 1) && (lbfgs->idx_cols != lmvm->m - 1)) PetscCall(VecRightward_Shift(B, this_sy_col, -lbfgs->idx_cols));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StYfull, StYidx, &this_sy_col));
      }

      ///* TODO perhaps unify idx? too many idx floating around rn */
      //switch (lbfgs->strategy) {
      //case MAT_LBFGS_CD_INPLACE:
      //  if (lbfgs->bind) {
//    //      PetscCall(MtMT_Internal(B, lbfgs->Sfull, lbfgs->Yfull, &lbfgs->StYfull));
      //    PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull_device));
      //    PetscCall(MatCopy(lbfgs->StYfull_device, lbfgs->StYfull, SAME_NONZERO_PATTERN)); 
      //  } else {
      //    Vec this_sy_col;

      //    PetscCall(MatDenseGetColumnVecWrite(lbfgs->StYfull, lbfgs->idx_cols, &this_sy_col));
      //    PetscCall(MatMultTranspose(lbfgs->Sfull, lmvm->Fprev, this_sy_col));
      //    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StYfull, lbfgs->idx_cols, &this_sy_col));
      //    //PetscCall(MatDenseGetSubMatrix(lbfgs->Sfull, PETSC_DECIDE, PETSC_DECIDE, lbfgs->idx_cols, lbfgs->idx_cols+1, &this_s));
      //    //PetscCall(MatDenseGetSubMatrix(lbfgs->StYfull, lbfgs->idx_cols, lbfgs->idx_cols+1, PETSC_DECIDE, PETSC_DECIDE, &this_sy_row));
      //    //PetscCall(MatTransposeMatMult(this_s, lbfgs->Yfull, MAT_REUSE_MATRIX, PETSC_DECIDE, &this_sy_row));
      //    //PetscCall(MatDenseRestoreSubMatrix(lbfgs->StYfull, &this_sy_row));
      //    //PetscCall(MatDenseRestoreSubMatrix(lbfgs->Sfull, &this_s));
      //  }

      //  if (lmvm->k == lmvm->m-1) {
      //    lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
      //    lbfgs->idx_cols = lbfgs->idx_begin;
      //  }
      //  break;
      //case MAT_LBFGS_CD_REORDER:
      //  {
      //    if (lmvm->k == lmvm->m - 1) {
      //      Vec workvec;
      //      lbfgs->idx_b_r = (lbfgs->idx_b_r+ 1) % lmvm->m;
      //      lbfgs->idx_cols = lbfgs->idx_b_r;
      //      if (lbfgs->idx_rplc == -1) {
      //        lbfgs->idx_rplc++;
      //        //When STY becomes full for first time, this routine shouldn't be run.
      //      } else if (lbfgs->idx_rplc == 0) {
      //        lbfgs->idx_rplc++;
      //        PetscCall(MatMove_LR3(B, lbfgs->StYfull));
      //      } else {
      //        lbfgs->idx_rplc = (lbfgs->idx_rplc ) % lmvm->m + 1;
      //      }
      //      /* Filling last column */
      //      PetscCall(MatDenseGetColumnVecWrite(lbfgs->StYfull, lmvm->m-1, &workvec));
      //      PetscCall(MatMultTranspose(lbfgs->Sfull, lmvm->Fprev, lbfgs->rwork1));
      //      PetscCall(VecRightward_Shift(B, lbfgs->rwork1, -lbfgs->idx_cols));
      //      PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
      //      PetscCall(VecCopyAsync_Private(lbfgs->rwork1, workvec, dctx));
      //      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StYfull, lmvm->m-1, &workvec));
      //      /* Filling last row */
      //      //PetscCall(FillLastRow(B, lbfgs->StYfull));
      //    } else {
      //      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      //    }
      //  }
      //case MAT_LBFGS_BASIC:
      //default:
      //  //TODO seterrq here? do we even need basic flag?
      //  break;
      //}

      PetscCall(MatGetDiagonal(lbfgs->StYfull, lbfgs->diag_vec));
      PetscCall(VecRightward_Shift(B, lbfgs->diag_vec, lbfgs->idx_rplc));

      if (lmvm->user_scale) {
        ///* Update B_0 S matrix */
        //PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec1));
        //PetscCall(MatCDBFGSApplyJ0Fwd(B, lmvm->Xprev, workvec1));
        //PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec1));
      } else {
        /* J0 is non-static. Need to manually compute BS matrix. */
        for (i=0; i<lmvm->m; i++) {
          PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, i, &workvec1));
          PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, i, &workvec2));
          PetscCall(MatCDBFGSApplyJ0Fwd(B, workvec1, workvec2));
          PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, i, &workvec1));
          PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, i, &workvec2));
        }
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
      lmvm->k = lmvm->k - 1;
    }
  } else {
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      /* No previous updates have been set, so we just update the diagonal with an initial scalar */
    //TODO THIS PART IS ERROR AFTER RESET 
    //1. USER_SCALE DISAPPEARS. NEED TO ADDRESS THIS
    //2. invD disappears (or was it ever there?)
    //
    //This is more fundamental to overall MATLMVM - set J0Scale is "Wrong", for an example
      dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
      diag_ctx = (Mat_DiagBrdn*)dbase->ctx;
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

static PetscErrorCode MatReset_LMVMCDBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatLMVMReset(lbfgs->diag_bfgs, destructive));
  }
  if (lbfgs->allocated && destructive) {
    PetscCall(MatDestroy(&lbfgs->temp_mat));
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Yfull));
    PetscCall(MatDestroy(&lbfgs->Sfull));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->J_work));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
    PetscCall(VecDestroy(&lbfgs->diag_vec));
    lbfgs->allocated = PETSC_FALSE;
  }
  lbfgs->idx_begin = -1;
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

      /* Create iteration storage matrices */
      PetscCall(VecGetType(X, &vec_type));
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, n, m, N, M, -1, NULL, &lbfgs->Sfull));
      /* If rwork is bound to CPU, then STYfull is on host */
      if (lbfgs->bind) {
        PetscCall(MatCreateDense(comm, m, m, M, M,  NULL, &lbfgs->StYfull));
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lbfgs->StYfull_device));
      } else {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lbfgs->StYfull));
//        PetscObjectReference((PetscObject)lbfgs->StYfull_device);
      }
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Yfull));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->BS));
      PetscCall(MatZeroEntries(lbfgs->Sfull));
      PetscCall(MatZeroEntries(lbfgs->Yfull));
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
//      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      //TODO MatMult is not really parallel here..
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lbfgs->J));
      PetscCall(MatDuplicate(lbfgs->J, MAT_DO_NOT_COPY_VALUES, &lbfgs->J_work));
      PetscCall(MatCreateVecs(lbfgs->J, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->J, &lbfgs->rwork3, &lbfgs->rwork4));

      if (lbfgs->bind) {
        PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork1_host, &lbfgs->rwork2_host));
        PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->diag_vec, NULL));
      } else {
        PetscObjectReference((PetscObject)lbfgs->rwork1);
        PetscObjectReference((PetscObject)lbfgs->rwork2);
        lbfgs->rwork1_host = lbfgs->rwork1;
        lbfgs->rwork2_host = lbfgs->rwork2;
        PetscCall(MatCreateVecs(lbfgs->J, &lbfgs->diag_vec, NULL));
      }
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork1));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork2));
    //TODO hacky way to turn diagbrdn lmvm off...
    PetscCall(MatLMVMSetJ0Scale(B,1.));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMAllocate(lbfgs->diag_bfgs, X, F));
    }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled = PETSC_TRUE;

    switch (lbfgs->strategy) {
    case MAT_LBFGS_CD_INPLACE:
      lbfgs->idx_rplc = 0;
      break;
    case MAT_LBFGS_CD_REORDER:
    case MAT_LBFGS_BASIC:
    default:
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDBFGS(Mat B)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lbfgs->temp_mat));
  PetscCall(MatDestroy(&lbfgs->StYfull));
  PetscCall(MatDestroy(&lbfgs->StYfull_device));
  PetscCall(MatDestroy(&lbfgs->Sfull));
  PetscCall(MatDestroy(&lbfgs->Yfull));
  PetscCall(MatDestroy(&lbfgs->J));
  PetscCall(MatDestroy(&lbfgs->J_work));
  PetscCall(MatDestroy(&lbfgs->BS));
  PetscCall(VecDestroy(&lbfgs->rwork1));
  PetscCall(VecDestroy(&lbfgs->rwork2));
  PetscCall(VecDestroy(&lbfgs->rwork3));
  PetscCall(VecDestroy(&lbfgs->rwork4));
  PetscCall(VecDestroy(&lbfgs->rwork1_host));
  PetscCall(VecDestroy(&lbfgs->rwork2_host));
  PetscCall(VecDestroy(&lbfgs->lwork1));
  PetscCall(VecDestroy(&lbfgs->lwork2));
  PetscCall(VecDestroy(&lbfgs->diag_vec));
  lbfgs->allocated = PETSC_FALSE;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMCDBFGS(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), NULL,  "Compact dense BFGS method (MATLMVMCDBFGS)", NULL);
  PetscCall(PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLBFGSType", MatLBFGSTypes, (PetscEnum)lbfgs->strategy, (PetscEnum *)&lbfgs->strategy, NULL));
  PetscCall(PetscOptionsBool("-mat_lbfgs_bind_to_cpu", "Bind work vectors to CPU for Device Memtype", "", lbfgs->bind, &lbfgs->bind, NULL));
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
  lbfgs->chol_ldlt_lazy  = PETSC_TRUE;
  lbfgs->idx_begin       = -1;
  lbfgs->idx_b_r         = -1;
  lbfgs->idx_rplc        = -1;
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  lbfgs->strategy        = MAT_LBFGS_CD_REORDER;
  lbfgs->bind            = PETSC_FALSE;

  lbfgs->iter_count =0;

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
