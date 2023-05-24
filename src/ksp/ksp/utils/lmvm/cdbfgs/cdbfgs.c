#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#include <petscdevice_cuda.h>
#include <petsc/private/deviceimpl.h>

typedef enum{
  MAT_CDBFGS_LOWER_TRIANGULAR,
  MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE,
  MAT_CDBFGS_UPPER_TRIANGULAR,
  MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE,
} TriangularTypes;

const char *const MatLBFGSTypes[] = {"basic", "cd_reorder", "cd_inplace", "MatLBFGSType", "MAT_LBFGS_", NULL};

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscCall(MatMult(lbfgs->diag_bfgs, X, Z));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCDBFGSApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscCall(MatSolve(lbfgs->diag_bfgs, F, dX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts vector rightward by given step. If given input is negative, it shifts leftward. 
 * This routine is only for Replace version. */

static PetscErrorCode VecRightward_Shift(Mat B, Vec X, PetscInt step)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscScalar *buffer1, *buffer2, *x_array;
  PetscInt     size, N;
  PetscFunctionBegin;
  switch (lbfgs->strategy) {
    case MAT_LBFGS_CD_REORDER:
      if ((lbfgs->idx_rplc == lmvm->m) || lbfgs->idx_rplc <= 0) {
        break;
      } else {
	PetscMemType memtype_x;
        size = PetscAbs(step);

        PetscCall(VecGetSize(X, &N));
      
        PetscCall(VecGetArrayAndMemType(X, &x_array, &memtype_x));
        PetscCall(PetscDeviceRegisterMemory(x_array, memtype_x, N*sizeof(*x_array)));
	//TODO devicecontext?
        PetscCall(PetscDeviceMalloc(NULL, memtype_x, size, &buffer1));
        PetscCall(PetscDeviceMalloc(NULL, memtype_x, N-size, &buffer2));
      
        if (step < 0) {
          PetscCall(PetscDeviceArrayCopy(NULL, buffer1, x_array, size));
          PetscCall(PetscDeviceArrayCopy(NULL, buffer2, &x_array[size], N-size));
          PetscCall(PetscDeviceArrayCopy(NULL, x_array, buffer2, N-size));
          PetscCall(PetscDeviceArrayCopy(NULL, &x_array[N - size], buffer1, size));
        } else {
          PetscCall(PetscDeviceArrayCopy(NULL, buffer1, &x_array[N-size], size));
          PetscCall(PetscDeviceArrayCopy(NULL, buffer2, x_array, N-size));
          PetscCall(PetscDeviceArrayCopy(NULL, &x_array[size], buffer2, N-size));
          PetscCall(PetscDeviceArrayCopy(NULL, x_array, buffer1, size));
        }
        PetscCall(VecRestoreArrayAndMemType(X, &x_array));
        PetscCall(PetscDeviceFree(NULL, buffer1));
        PetscCall(PetscDeviceFree(NULL, buffer2));
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
  //TODO mat and input vec size check assert
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
	    //TODO does this actually work?
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
    
            //TODO technically, when size of BLAS call is only one, I can just manuall compute it to avoid BLAS kernel launch?
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
  PetscInt     lda;

  const PetscScalar *r_array;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayReadAndMemType(R, &r_array, &memtype_r));
  PetscCall(VecGetArrayWriteAndMemType(x, &x_array, &memtype_x));
  PetscCall(MatDenseGetLDA(R, &lda));
  PetscAssert(memtype_x == memtype_r, comm, PETSC_ERR_PLIB, "Incompatible device pointers");

  switch (lbfgs->strategy) {
  case MAT_LBFGS_CD_REORDER:
    {
      switch (memtype_x) {
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

          PetscCall(PetscCUBLASGetHandle(&handle));
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
      PetscCall(VecRestoreArrayWriteAndMemType(x, &x_array));
      PetscCall(MatDenseRestoreArrayReadAndMemType(R, &r_array));
    }
    break;
  case MAT_LBFGS_CD_INPLACE:
      /* Shift x vector TODO there are x,b vecs. but actually we only need one here... */
    //TODO what is this???
#if 0 
      PetscCall(VecGetArrayRead(x, &array_read));
      PetscCall(VecGetSize(x, &N));
      PetscCall(PetscMalloc1(N, &buffer));
      PetscCall(PetscMemcpy(buffer, &array_read[index], (N - index)*sizeof(PetscScalar)));
      if (index != 0 ) {
        PetscCall(PetscMemcpy(&buffer[N - index], array_read, (index)*sizeof(PetscScalar)));
      }
      PetscCall(VecRestoreArrayReadAndMemType(x, &array_read));
      PetscCall(PetscFree(buffer));
#endif      
    switch (memtype_x) {
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
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &idx_blas, &idx_blas, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
          /* Applying B: x' = x - BC^-1 y' */
          PetscCallBLAS("BLASgemv", BLASgemv_("N",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying A: x' = A^-1 (x - BC^-1 y) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &diff_blas, &diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          /* Upper Triangular Transpose Case: 
           * Below, C,A are UT.
           * [ C | 0 ]^-T [y] => [C^-T(y - B^T A^-T x)]
           * [ B | A ]    [x]    [A^-T x              ] */
          /* Applying A: x' = A^-T x */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &diff_blas, &diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          /* Applying B: y' = y - B^T A^-T x */
          PetscCallBLAS("BLASgemv", BLASgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, &x_array[idx_blas], &one, &Alpha, x_array, &one));
          /* Applying C: y' = C^-T (y - B^T A^-T x) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &idx_blas, &idx_blas, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
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

        PetscCall(PetscCUBLASGetHandle(&handle));
        PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        //PetscAssert(PetscDefined(BLAS)...));
        PetscCuBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscCuBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
        PetscCall(PetscCuBLASIntCast(lmvm->k + 1 - lowest_index, &diff_blas));
        PetscCuBLASInt ldb_blas = lda_blas;

        switch (tri_type) {
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, idx_blas, idx_blas, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_N, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, x_array, one, &Alpha, &x_array[idx_blas], one));
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, diff_blas, diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, diff_blas, diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          PetscCallCUBLAS(cublasDgemv(handle, CUBLAS_OP_T, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
          PetscCallCUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, idx_blas, idx_blas, &Alpha, r_array, lda_blas, x_array, ldb_blas));
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
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, idx_blas, idx_blas, &Alpha, r_array, lda_blas, x_array, ldb_blas));
          PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_N, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, x_array, one, &Alpha, &x_array[idx_blas], one));
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, diff_blas, diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, diff_blas, diff_blas, &Alpha, &r_array[idx_blas*(lda_blas+1)], lda_blas, &x_array[idx_blas], ldb_blas));
          PetscCallHIPBLAS(hipblasXgemv(handle, HIPBLAS_OP_T, diff_blas, idx_blas, &neg_one, &r_array[idx_blas], lda_blas, &x_array[idx_blas], one, &Alpha, x_array, one));
          PetscCallHIPBLAS(hipblasXtrsm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, idx_blas, idx_blas, &Alpha, r_array, lda_blas, x_array, ldb_blas));
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
      PetscCall(PetscDeviceRegisterMemory(x_array, memtype_x, N*sizeof(*x_array)));
      //TODO dctx
      if (lmvm->k != lmvm->m -1) {
        PetscCall(PetscDeviceArrayZero(NULL, &x_array[lmvm->k+1], lmvm->m - lmvm->k -1));
      }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented L-BFGS strategy");
    }
    PetscCall(VecRestoreArrayAndMemType(X, &x_array));
  } 
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for 
 * [ I | S R^{-T} ] [   I  | 0 ] [ H_0 | 0 ] [ I | -Y ] [     I      ]
 *                  [ -Y^T | I ] [  0  | D ] [ 0 |  I ] [ R^{-1} S^T ]  */

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscInt index;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (lmvm->k == -1) {
    PetscCall(MatCDBFGSApplyJ0Inv(H, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  if (lbfgs->idx_begin == -1) {
    index = 0;
  } else {
    index = lbfgs->idx_begin;
  }
  /* Start with reusable part: rwork1 = R^-1 S^T F */
  PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->rwork1));
  PetscCall(Vec_Truncate(H,lbfgs->rwork1));

  /* Reordering rwork1, as STY is in canonical order, while S is in recycled order */
  PetscCall(VecRightward_Shift(H, lbfgs->rwork1, -lbfgs->idx_rplc));      
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, index, lbfgs->rwork1, MAT_CDBFGS_UPPER_TRIANGULAR));
  PetscCall(VecRightward_Shift(H, lbfgs->rwork1, lbfgs->idx_rplc));      
  PetscCall(Vec_Truncate(H,lbfgs->rwork1));

  /* lwork1 :H_0 (F - Y R^{-1} S^T X) */
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork1, lbfgs->lwork1));
  PetscCall(VecWAXPY(lbfgs->lwork2, -1., lbfgs->lwork1, F));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->lwork2, lbfgs->lwork1));
  PetscCall(VecCopy(lbfgs->lwork1, dX));

  /* -S R^{-T} ( Y^T lwork1 - D rwork1 ) */
  PetscCall(VecPointwiseMult(lbfgs->rwork1, lbfgs->diag_vec, lbfgs->rwork1));
  PetscCall(MatMultTranspose(lbfgs->Yfull, lbfgs->lwork1, lbfgs->rwork2));
  PetscCall(Vec_Truncate(H,lbfgs->rwork2));
  PetscCall(VecAXPY(lbfgs->rwork2, -1., lbfgs->rwork1));

  /* Reordering rwork2, as STY is in canonical order, while S is in recycled order */
  PetscCall(VecRightward_Shift(H, lbfgs->rwork2, -lbfgs->idx_rplc));      
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, index, lbfgs->rwork2, MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE));
  PetscCall(VecRightward_Shift(H, lbfgs->rwork2, lbfgs->idx_rplc));      
  PetscCall(Vec_Truncate(H,lbfgs->rwork2));
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(VecAXPY(dX, -1., lbfgs->lwork1));
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
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Adds LDL^T to J mat */

static PetscErrorCode MatAdd_LDLT(Mat B)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  const PetscScalar *r_array;
  PetscScalar       *x_array, *buffer;
  PetscInt           i, j, k, query_idx_i, query_idx_j, query_idx_k, index;
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
  PetscCall(VecGetArrayRead(lbfgs->diag_vec, &r_array));
  for (i=0; i<lmvm->m-1; i++) {
    query_idx_i = (index + i) % lmvm->m;
    PetscCall(MatDenseGetColumnVecRead(lbfgs->StYfull, query_idx_i, &workvec1));

    /* Copying to emulate strictly lower triangular */
    PetscCall(VecCopy(workvec1, lbfgs->rwork1));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->StYfull, query_idx_i, &workvec1));
    PetscCall(VecGetArray(lbfgs->rwork1, &x_array));
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
    PetscCall(VecRestoreArray(lbfgs->rwork1, &x_array));

    for (j=0, k=i+1; k<lmvm->m; k++, j++) {
      query_idx_j = (index + j) % lmvm->m;
      query_idx_k = (index + k) % lmvm->m;
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
      PetscCall(VecAXPY(workvec2, buffer[j], lbfgs->rwork1));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->J, query_idx_k, &workvec2));
    }
    PetscCall(PetscFree(buffer));
  }
  PetscCall(VecRestoreArrayRead(lbfgs->diag_vec, &r_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatRotate_STY_CCW(Mat R)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)R->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  PetscScalar  *x_array, *buffer1, *buffer2;
  PetscInt     a,b,i,N;
  Vec          workvec;
  Mat          B,D,B_work,D_work;
  PetscMemType memtype_vec;

  PetscFunctionBegin;

  if (lbfgs->idx_rplc == -1) {
    lbfgs->idx_rplc++;
    PetscFunctionReturn(PETSC_SUCCESS);
    //When STY becomes full for first time, this routine shouldn't be run.
  }

  //TODO is there more elegant way to do this?
  if (lbfgs->idx_rplc == 0) {
    lbfgs->idx_rplc++;
  } else {
    lbfgs->idx_rplc = (lbfgs->idx_rplc ) % lmvm->m + 1;
  }

  a = lmvm->k + 1 - lbfgs->idx_rplc;
  b = lbfgs->idx_rplc;
  /* Currently STY is in permuted format. Need to rotate anti-clockwise twice.                  *
   * diff: k + 1 - idx_rplc                                                                     *
   * A : [diff x diff], B : [diff x idx_rplc], C : [idx_rplc x idx_rplc], D : [idx_rplc x diff] *
   * [ C | D ]  ->  [ A | B ]                                                                   *
   * [ B | A ]      [ D | C ]                                                                   */

  /* Copy CB */
  if (b != 0) {
    /* Cannot use MatConvert here, as B,D_work is MEMTYPE_HOST TODO can we fix this? */
    PetscCall(MatDenseGetSubMatrix(lbfgs->StYfull, 0, lmvm->m, 0, b, &B));
    PetscCall(VecCreateMatDense(lmvm->Xprev, PETSC_DECIDE, PETSC_DECIDE, lmvm->m, b, NULL, &B_work));
    PetscCall(MatCopy(B, B_work, SAME_NONZERO_PATTERN));
    PetscCall(MatDenseRestoreSubMatrix(lbfgs->StYfull, &B));
  }

  if (a != 0) {
    /* Cannot use MatConvert here, as B,D_work is MEMTYPE_HOST TODO can we fix this? */
    PetscCall(MatDenseGetSubMatrix(lbfgs->StYfull, 0, lmvm->m, b, lmvm->m, &D));
    PetscCall(VecCreateMatDense(lmvm->Xprev, PETSC_DECIDE, PETSC_DECIDE, lmvm->m, lmvm->m - b, NULL, &D_work));
    PetscCall(MatCopy(D, D_work, SAME_NONZERO_PATTERN));
    PetscCall(MatDenseRestoreSubMatrix(lbfgs->StYfull, &D));
  }

  //TODO cuda doesnt like 0 x 0 memcopy... 
  if (b != 0) {
    /* Turn [C;B] to [B;C] */
    //TODO prob can unify buffer1,2 creation into one.
    for (i=0; i<b; i++) {
      PetscCall(MatDenseGetColumnVecRead(B_work, i, &workvec));
      PetscCall(VecCopy(workvec, lbfgs->rwork1));
      PetscCall(VecGetSize(workvec,&N));
      PetscCall(MatDenseRestoreColumnVecRead(B_work, i, &workvec));
      PetscCall(VecGetArrayAndMemType(lbfgs->rwork1, &x_array, &memtype_vec));
      PetscCall(PetscDeviceRegisterMemory(x_array, memtype_vec, N*sizeof(*x_array)));
      //TODO dctx?
      //TODO somehow I still need to call PetscDeviceRegisterMemory,
      //even after getting array via VecGetArrayAndMemType....
      PetscCall(PetscDeviceMalloc(NULL, memtype_vec, b, &buffer1));
      PetscCall(PetscDeviceMalloc(NULL, memtype_vec, a, &buffer2));
      PetscCall(PetscDeviceArrayCopy(NULL, buffer1, x_array, b));
      /* Shift */
      PetscCall(PetscDeviceArrayCopy(NULL, buffer2, &x_array[b], a));
      PetscCall(PetscDeviceArrayCopy(NULL, x_array, buffer2, a));
      /* Paste C part */
      PetscCall(PetscDeviceArrayCopy(NULL, &x_array[a], buffer1, b));
      PetscCall(PetscDeviceFree(NULL, buffer1));
      PetscCall(PetscDeviceFree(NULL, buffer2));
      PetscCall(VecRestoreArrayAndMemType(lbfgs->rwork1, &x_array));
    
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->StYfull, a+i, &workvec));
      PetscCall(VecCopy(lbfgs->rwork1, workvec));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StYfull, a+i, &workvec));
    }
    PetscCall(MatDestroy(&B_work));
  }

  /* Copy DA */
  /* Skip if size of DA is zero */
  if (a != 0) {
    /* Turn [D;A] to [A;D] */
    for (i=0; i<a; i++) {
      PetscCall(MatDenseGetColumnVecRead(D_work, i, &workvec));
      PetscCall(VecCopy(workvec, lbfgs->rwork1));
      PetscCall(MatDenseRestoreColumnVecRead(D_work, i, &workvec));
      PetscCall(VecGetArrayAndMemType(lbfgs->rwork1, &x_array, &memtype_vec));
      PetscCall(PetscDeviceRegisterMemory(x_array, memtype_vec, N*sizeof(*x_array)));
      PetscCall(PetscDeviceMalloc(NULL, memtype_vec,  a, &buffer1));
      PetscCall(PetscDeviceMalloc(NULL, memtype_vec,  b, &buffer2));
      PetscCall(PetscDeviceArrayCopy(NULL, buffer1, &x_array[b], a));
      /* Shift */
      PetscCall(PetscDeviceArrayCopy(NULL, buffer2, x_array, b));
      PetscCall(PetscDeviceArrayCopy(NULL, &x_array[a], buffer2, b));
      /* Paste A part */
      PetscCall(PetscDeviceArrayCopy(NULL, x_array, buffer1, a));
      PetscCall(PetscDeviceFree(NULL, buffer1));
      PetscCall(PetscDeviceFree(NULL, buffer2));
      PetscCall(VecRestoreArrayAndMemType(lbfgs->rwork1, &x_array));
    
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->StYfull, i, &workvec));
      PetscCall(VecCopy(lbfgs->rwork1, workvec));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StYfull, i, &workvec));
    }
    PetscCall(MatDestroy(&D_work));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;
 
  PetscScalar curvature, ststmp;
  PetscReal   curvtol;
  PetscInt    N, n, low, high, i;
  //MPI_Comm    comm = PetscObjectComm((PetscObject)B);
  Vec         workvec1, workvec2;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Test if the updates can be accepted */
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp));
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {

      lbfgs->iter_count++;//TODO for debugging purpose. delete later

      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetSize(lmvm->Xprev, &N));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));

      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        PetscCall(MatLMVMUpdate(lbfgs->diag_bfgs, X, F));
      }

      if (lmvm->k != lmvm->m-1) {
        lmvm->k = lmvm->k + 1;
        lbfgs->idx_cols = lmvm->k;
      }

      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Sfull, lbfgs->idx_cols, &workvec1));
      PetscCall(VecCopy(lmvm->Xprev, workvec1));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Sfull, lbfgs->idx_cols, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec1));
      PetscCall(VecCopy(lmvm->Fprev, workvec1));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec1));

      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));

      /* TODO perhaps unify idx? too many idx floating around rn */
      switch (lbfgs->strategy) {
      case MAT_LBFGS_CD_INPLACE:
        //TODO technically can make MTMMult to be just updating idx_col row and col of new MTMMult
        if (lmvm->k == lmvm->m-1) {
          lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
          lbfgs->idx_cols = lbfgs->idx_begin;
        }
        break;
      case MAT_LBFGS_CD_REORDER:
        if (lmvm->k == lmvm->m - 1) {
          lbfgs->idx_b_r = (lbfgs->idx_b_r+ 1) % lmvm->m;
          lbfgs->idx_cols = lbfgs->idx_b_r;
          PetscCall(MatRotate_STY_CCW(B));
        }
      case MAT_LBFGS_BASIC:
      default:
        //TODO seterrq here? do we even need basic flag?
        break;
      }

      PetscCall(MatGetDiagonal(lbfgs->StYfull, lbfgs->diag_vec));
      PetscCall(VecRightward_Shift(B, lbfgs->diag_vec, lbfgs->idx_rplc));

      if (lmvm->user_scale) {
        /* Update B_0 S matrix */
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec1));
        PetscCall(MatCDBFGSApplyJ0Fwd(B, lmvm->Xprev, workvec1));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec1));
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

      //Now, SBS is in shifted order in all strategies.
      /* Compute S^T B S + L D^{-1} L^T
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

    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
      lmvm->k = lmvm->k - 1;
    }
  } else {
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      /* No previous updates have been set, so we just update the diagonal with an initial scalar */
      dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
      dctx = (Mat_DiagBrdn*)dbase->ctx;
      PetscCall(VecSet(dctx->invD, lbfgs->delta));
    }
  }
  
  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      PetscCall(MatLMVMReset(lbfgs->diag_bfgs, PETSC_FALSE));
    }
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
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
  //TODO this is dirty, but cant think of anything else rn? also goes against actual description in lmvmutil...
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
      /* Create iteration storage matrices */
      PetscCall(VecCreateMatDense(X, n, lmvm->m, N, lmvm->m, NULL, &lbfgs->Sfull));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->Yfull));
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->BS));
      PetscCall(MatZeroEntries(lbfgs->Sfull));
      PetscCall(MatZeroEntries(lbfgs->Yfull));
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->J));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->J_work));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->diag_vec, NULL));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork3, &lbfgs->rwork4));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork1));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork2));
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
  if (lbfgs->allocated) {
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Sfull));
    PetscCall(MatDestroy(&lbfgs->Yfull));
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
  PetscCall(MatGetSize(B, &M, &N));
  if (M == 0 && N == 0) SETERRQ(comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
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
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), NULL, "Compact dense BFGS method (MATLMVMCDBFGS)", NULL);
  PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLBFGSType", MatLBFGSTypes, (PetscEnum)lbfgs->strategy, (PetscEnum *)&lbfgs->strategy, NULL);
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
  lbfgs->idx_begin       = -1;
  lbfgs->idx_b_r         = -1;
  lbfgs->idx_rplc        = -1;
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  lbfgs->strategy        = MAT_LBFGS_CD_REORDER;

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
   it may be faster on GPUs for large enough problems (note: requires CUDA).

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
