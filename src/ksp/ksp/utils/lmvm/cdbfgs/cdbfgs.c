
#include <../src/ksp/ksp/utils/lmvm/cdbfgs/cdbfgs.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>

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

/* MatMult for strictly lower triangular part of StYfull matrix.  */

PetscErrorCode MatLowerTriangularMult(Mat B, Vec X, TriangularTypes tri_type)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

  PetscInt     lda;
  PetscScalar *x_array;
  PetscBLASInt m_blas, lda_blas, one = 1;
  PetscMemType memtype_r, memtype_x;
  MPI_Comm     comm = PetscObjectComm((PetscObject)B);

  const PetscScalar *r_array;
  
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(lmvm->m-1, &m_blas));
  PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
  PetscCall(PetscBLASIntCast(lda, &lda_blas));
  PetscCall(VecGetArrayWriteAndMemType(X, &x_array, &memtype_x));
  PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->StYfull, &r_array, &memtype_r));
  //TODO mat and input vec size check assert

  switch (tri_type) {
  case MAT_CDBFGS_LOWER_TRIANGULAR:
    PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Normal", "NotUnitTriangular", &m_blas, &r_array[1], &lda_blas, x_array, &one));
    PetscCall(PetscArraymove(&x_array[1],&x_array[0], lmvm->m-1));
    x_array[0] = 0;
    break;
  case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
    PetscCallBLAS("BLAStrmv", BLAStrmv_("Lower", "Transpose", "NotUnitTriangular", &m_blas, &r_array[1], &lda_blas, &x_array[1], &one));
    PetscCall(PetscArraymove(&x_array[0],&x_array[1], lmvm->m-1));
    x_array[lmvm->m-1] = 0;
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

/*------------------------------------------------------------*/

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
  PetscScalar *buffer, *x_array;
  PetscInt     lda, N;

  const PetscScalar *array_read, *r_array;

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
            PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
            PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          }  
        }
        break;
      case PETSC_MEMTYPE_CUDA:
      case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
        {
          PetscCuBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
          PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
          PetscCuBLASInt ldb_blas = lda_blas;
          switch (tri_type) {
          case MAT_CDBFGS_UPPER_TRIANGULAR:
            PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
            PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR:
            PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
            PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          }
        }
#endif
        break;
      case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
        {
          PetscHIPBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscHIPBLASIntCast(lmvm->k, &m_blas));
          PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
          PetscHIPBLASInt ldb_blas = lda_blas;
          switch (tri_type) {
          case MAT_CDBFGS_UPPER_TRIANGULAR:
            PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
            PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR:
            PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
          case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
            PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &m_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));
            break;
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
      PetscCall(VecGetArrayRead(x, &array_read));
      PetscCall(VecGetSize(x, &N));
      PetscCall(PetscMalloc1(N, &buffer));//TODO free buffer
      PetscCall(PetscMemcpy(buffer, &array_read[lbfgs->idx_begin], (N - lbfgs->idx_begin)*sizeof(PetscScalar)));
      if (lbfgs->idx_begin != 0 ) {
        PetscCall(PetscMemcpy(&buffer[N - lbfgs->idx_begin], array_read, (lbfgs->idx_begin)*sizeof(PetscScalar)));
      }
      PetscCall(VecRestoreArrayReadAndMemType(x, &array_read));
    switch (memtype_x) {
    case PETSC_MEMTYPE_HOST:
      {
        //PetscAssert(PetscDefined(BLAS)...));
        PetscBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscBLASIntCast(lda, &lda_blas));
        PetscCall(PetscBLASIntCast(lmvm->k - lowest_index, &diff_blas));
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
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          /* Applying B: x' = x - BC^-1 y' */                                                                                                                                                                
          PetscCallBLAS("BLASgemv", BLASgemv_("N",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying A: x' = A^-1 (x - BC^-1 y) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          /* Upper Triangular Transpose Case: 
           * Below, C,A are UT.
           * [ C | 0 ]^-T [y] => [C^-T(y - B^T A^-T x)] 
           * [ B | A ]    [x]    [A^-T x              ] */
          /* Applying A: x' = A^-T x */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          /* Applying B: y' = y - B^T A^-T x */
          PetscCallBLAS("BLASgemv", BLASgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying C: y' = C^-T (y - B^T A^-T x) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
          /* Lower Triangular Case: 
           * Below, C,A are LT.
           * [ C | D ]^-1 [y] => [C^-1(y - D A^-1 x)] 
           * [ 0 | A ]    [x]    [A^-1 x            ] */
          /* Applying A: x' = A^-1 x */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          /* Applying D: y' = y - D A^-1 x */
          PetscCallBLAS("BLASgemv", BLASgemv_("N",  &idx_blas, &diff_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, &x_array[idx_blas], &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying C: y' = C^-1 (y - D A^-1 x) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          /* Lower Triangular Transpose Case: 
           * Below, C,A are LT.
           * [ C | D ]^-T [y] => [C^-T                 ] 
           * [ 0 | A ]    [x]    [A^-T (x - D^T C^-T y)] */
          /* Applying C: y' = C^-T y */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          /* Applying D: x' = x - D^T C^-T y */                                                                                                                                                                   
          PetscCallBLAS("BLASgemv", BLASgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          /* Applying A: x' = A^-T (x - D^T C^-T y) */
          PetscCallBLAS("BLAStrsm", BLAStrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        }  
      }
    break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
#if defined(PETSC_HAVE_CUDA)
      {
        //PetscAssert(PetscDefined(BLAS)...));
        PetscCuBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscCuBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscCuBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscCuBLASIntCast(lda, &lda_blas));
        PetscCall(PetscCuBLASIntCast(lmvm->k - lowest_index, &diff_blas));
        PetscCuBLASInt ldb_blas = lda_blas;

        switch (tri_type) {//TODO cublasDgemv, or cublasSgemv?
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          PetscCallCUBLAS("cublasDgemv", cublasDgemv_("N",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          PetscCallCUBLAS("cublasDgemv", cublasDgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          PetscCallCUBLAS("cublasDgemv", cublasDgemv_("N",  &idx_blas, &diff_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, &x_array[idx_blas], &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          PetscCallCUBLAS("cublasDgemv", cublasDgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallCUBLAS("cublastrsm", cublastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        }  
      }
#endif      
      break;
    case PETSC_MEMTYPE_HIP:
#if defined(PETSC_HAVE_HIP)
      {
        //PetscAssert(PetscDefined(BLAS)...));
        PetscHIPBLASInt m_blas, idx_blas, lda_blas, diff_blas, one = 1;
        PetscCall(PetscHIPBLASIntCast(lmvm->k, &m_blas));
        PetscCall(PetscHIPBLASIntCast(lowest_index, &idx_blas));
        PetscCall(PetscHIPBLASIntCast(lda, &lda_blas));
        PetscCall(PetscHIPBLASIntCast(lmvm->k - lowest_index, &diff_blas));
        PetscHIPBLASInt ldb_blas = lda_blas;

        switch (tri_type) {//TODO hipblasDgemv, or hipblasSgemv?
        case MAT_CDBFGS_UPPER_TRIANGULAR:
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          PetscCallHIPBLAS("hipblasDgemv", hipblasDgemv_("N",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
        case MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE:
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          PetscCallHIPBLAS("hipblasDgemv", hipblasDgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Upper", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR:
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          PetscCallHIPBLAS("hipblasDgemv", hipblasDgemv_("N",  &idx_blas, &diff_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, &x_array[idx_blas], &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Normal", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          break;
        case MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE:
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &idx_blas, &one, &Alpha, r_array, &lda_blas, x_array, &ldb_blas));//what if idx_blas=0. does this still work? TODO
          PetscCallHIPBLAS("hipblasDgemv", hipblasDgemv_("T",  &diff_blas, &idx_blas, &neg_one, &r_array[idx_blas*m_blas], &lda_blas, x_array, &one, &Alpha, &x_array[idx_blas], &one));
          PetscCallHIPBLAS("hipblastrsm", hipblastrsm_("Left", "Lower", "Transpose", "NotUnitTriangular", &diff_blas, &one, &Alpha, &r_array[idx_blas*(m_blas+1)], &lda_blas, &x_array[idx_blas], &ldb_blas));
          break;
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


/* Solves for 
 * H_0 - [H_0 Y | S] [   0   |         -R^-1        ] [Y^T H_0]
 *                   [ -R^-T | R^-T(D+Y^T H_0 Y)R^-1] [  S^T  ] */

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM*)H->data;
  Mat_CDBFGS  *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (lmvm->k == -1) {
    PetscCall(MatCDBFGSApplyJ0Inv(H, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }
  /* Start with reusable part: rwork1 = R^-1 S^T F */
  PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->rwork1));
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork1, MAT_CDBFGS_UPPER_TRIANGULAR));

  /* lwork1 :H_0 (F - Y R^{-1} S^T X) */
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork1, lbfgs->lwork1));
  PetscCall(VecWAXPY(lbfgs->lwork2, -1., lbfgs->lwork1, F));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->lwork2, lbfgs->lwork1));
  PetscCall(VecCopy(lbfgs->lwork1, dX));

  /* -S R^{-T} ( Y^T lwork1 - D rwork1 ) */
  PetscCall(VecPointwiseMult(lbfgs->rwork1, lbfgs->diag_vec, lbfgs->rwork1));
  PetscCall(MatMultTranspose(lbfgs->Yfull, lbfgs->lwork1, lbfgs->rwork2));
  PetscCall(VecAXPY(lbfgs->rwork2, -1., lbfgs->rwork1));
  PetscCall(MatSolveTriangular(H, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork2, MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE));
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

  /* Common part: rwork2:  J^{-T} J^{-1} (L D^{-1} Y^T X + S^T B_0 X) */
  PetscCall(VecPointwiseDivide(lbfgs->rwork2,lbfgs->rwork3, lbfgs->diag_vec));
  PetscCall(MatLowerTriangularMult(B, lbfgs->rwork2, MAT_CDBFGS_LOWER_TRIANGULAR));
  PetscCall(VecAXPY(lbfgs->rwork2, 1., lbfgs->rwork4));

  PetscCall(ISCreateStride(PETSC_COMM_WORLD, lmvm->k+1, 0, 1, &temp_is));
  PetscCall(VecGetSubVector(lbfgs->rwork1, temp_is, &temp_1));
  PetscCall(VecGetSubVector(lbfgs->rwork2, temp_is, &temp_2));

  /* TODO LU factor actually doesn't do anything now... MatSolve(J_solve) is actually just MatSolve(JJ^T) */
  PetscCall(MatSolve(lbfgs->J_solve, temp_2, temp_1));
  PetscCall(MatSolveTranspose(lbfgs->J_solve, temp_1, temp_2));
  PetscCall(VecRestoreSubVector(lbfgs->rwork1, temp_is, &temp_1));
  PetscCall(VecRestoreSubVector(lbfgs->rwork2, temp_is, &temp_2));

  /* Bottom part:  - B_0 S rwork2 */
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(Z, -1., lbfgs->lwork2));

  /* Top part: + Y D^{-1} ( Y^T X - D^{-1} L^T rwork2 ) */
  PetscCall(MatLowerTriangularMult(B, lbfgs->rwork2, MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE));
  PetscCall(VecPointwiseDivide(lbfgs->rwork4,lbfgs->rwork2, lbfgs->diag_vec));
  PetscCall(VecAXPY(lbfgs->rwork3, -1., lbfgs->rwork4));
  PetscCall(VecPointwiseDivide(lbfgs->rwork4,lbfgs->rwork3, lbfgs->diag_vec));
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork3, lbfgs->lwork1));
  PetscCall(VecAXPY(Z, 1., lbfgs->lwork1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/
//Currently only for Replace version...
static PetscErrorCode MatUpdate_C_Matrix(Mat_LMVM *lmvm, Mat_CDBFGS *lbfgs, Mat C, Mat BS_or_HY, Mat S_or_Y, Vec Z)
{
  const PetscScalar *array_read, *array_read_sty;
  PetscScalar       *array_ptr;
  PetscInt           i, lda;
  Vec                workvec;

  PetscFunctionBegin;
  /* Update C first */
  PetscCall(MatDenseGetArray(C, &array_ptr));
  if (lmvm->k == lmvm->m-1) {
    /* C is full.  Shifting matrix. */
    PetscCall(PetscArraymove(array_ptr, &array_ptr[lmvm->m], (lmvm->m-1)*(lmvm->m-1)));
  }
  /* Note:  here, all the vecs and mat are full, which means wasting some flops and mem for first m iter. but that saves few kernel launches for destory and create */
  /* Construct last row */
  PetscCall(MatMultTranspose(BS_or_HY, Z, lbfgs->rwork1));
  PetscCall(VecGetArrayRead(lbfgs->rwork1, &array_read));
  /* Iterate over to push in last row */
  for (i=0; i <lmvm->k; i++) {
    array_ptr[i*(lmvm->k + 1) + lmvm->k] = array_read[i];//TODO fix indexing here for inplace version
  }
  /* Push in last column  */
  PetscCall(MatDenseGetColumnVecRead(BS_or_HY, lbfgs->idx_cols, &workvec));
  PetscCall(MatMultTranspose(S_or_Y, workvec, lbfgs->rwork2));
  PetscCall(MatDenseRestoreColumnVecRead(BS_or_HY, lbfgs->idx_cols, &workvec));

  PetscCall(MatDenseGetColumnVecWrite(C, lmvm->k, &workvec));
  PetscCall(VecCopy(lbfgs->rwork2, workvec));
  PetscCall(MatDenseRestoreColumnVecWrite(C, lbfgs->idx_cols, &workvec));

  /* Adding (k,k) diagonal element */
  PetscCall(MatDenseGetLDA(lbfgs->StYfull, &lda));
  PetscCall(MatDenseGetArrayRead(lbfgs->StYfull, &array_read_sty));
  array_ptr[lda*lmvm->k + lmvm->k] += array_read_sty[lda*lmvm->k + lmvm->k];
  PetscCall(MatDenseRestoreArrayRead(lbfgs->StYfull,&array_read_sty));
  PetscCall(VecRestoreArrayRead(lbfgs->rwork1, &array_read));
  PetscCall(MatDenseRestoreArray(C, &array_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;
 
  PetscScalar       curvature, ststmp, *array_ptr;
  PetscReal         curvtol;
  PetscInt          N, n, low, high, i, j, k;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);

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

      lbfgs->iter_count++;      
      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetSize(lmvm->Xprev, &N));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));

      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
      }

      //TODO temp to debug. delete later
      dbase = (Mat_LMVM*)lbfgs->diag_bfgs->data;
      dctx = (Mat_DiagBrdn*)dbase->ctx;

      switch (lbfgs->strategy) {
      case (MAT_LBFGS_CD_REORDER):
        if (lmvm->k == lmvm->m-1) {
          /* S Matrix is full. Shift matrix */
          PetscCall(MatDenseGetArray(lbfgs->Sfull, &array_ptr));
          PetscCall(PetscArraymove(array_ptr, &array_ptr[N], (lmvm->m - 1)*N));
          PetscCall(MatDenseRestoreArray(lbfgs->Sfull, &array_ptr));
  
          /* Y Matrix is full. Shift matrix */
          PetscCall(MatDenseGetArray(lbfgs->Yfull, &array_ptr));
          PetscCall(PetscArraymove(array_ptr, &array_ptr[N], (lmvm->m - 1)*N));
          PetscCall(MatDenseRestoreArray(lbfgs->Yfull, &array_ptr));

          if (lmvm->user_scale) {
            /* Static J0. More efficient to precompute these */
            /* H_0 Y Matrix is full. Shift matrix */
            PetscCall(MatDenseGetArray(lbfgs->HY, &array_ptr));
            PetscCall(PetscArraymove(array_ptr, &array_ptr[N], (lmvm->m - 1)*N));
            PetscCall(MatDenseRestoreArray(lbfgs->HY, &array_ptr));

            /* B_0 S Matrix is full. Shift matrix */
            PetscCall(MatDenseGetArray(lbfgs->BS, &array_ptr));
            PetscCall(PetscArraymove(array_ptr, &array_ptr[N], (lmvm->m - 1)*N));
            PetscCall(MatDenseRestoreArray(lbfgs->BS, &array_ptr));
          }
        } else {
          lmvm->k = lmvm->k + 1;
        }
        lbfgs->idx_cols = lmvm->k;
        break;
      case (MAT_LBFGS_CD_INPLACE):
        /* Inplace doesn't move memory, but rather only finds index of oldest memory */
        if (lmvm->k == lmvm->m-1) {
          lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
          lbfgs->idx_cols = lbfgs->idx_begin;
        } else {
          lmvm->k = lmvm->k + 1;
          lbfgs->idx_cols = lmvm->k;
        }
        break;
      case (MAT_LBFGS_BASIC):
        SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented CDLBFGS Method");
        break;
      }

      /* First update the S^T matrix */
      Vec workvec;
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Sfull, lbfgs->idx_cols, &workvec));
      PetscCall(VecCopy(lmvm->Xprev, workvec));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Sfull, lbfgs->idx_cols, &workvec));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec));
      PetscCall(VecCopy(lmvm->Fprev, workvec));
      PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->Yfull, lbfgs->idx_cols, &workvec));

      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatGetDiagonal(lbfgs->StYfull, lbfgs->diag_vec));

      if (lmvm->user_scale) {
        /* Update H_0 Y matrix */
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->HY, lbfgs->idx_cols, &workvec));
        PetscCall(MatCDBFGSApplyJ0Inv(B, lmvm->Fprev, workvec));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->HY, lbfgs->idx_cols, &workvec));
  
        /* Update B_0 S matrix */
        PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec));
        PetscCall(MatCDBFGSApplyJ0Fwd(B, lmvm->Xprev, workvec));
        PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, lbfgs->idx_cols, &workvec));
        /* Reordering C matrices. C_H: D+ Y^T H_0 Y, C_B: S^T B_0 S */
        switch (lbfgs->strategy) {
        case (MAT_LBFGS_CD_REORDER):
          if (lmvm->k == 0) {
            PetscScalar c_b1, c_h1, diag00;
            PetscCall(MatCDBFGSApplyJ0Inv(B, lmvm->Fprev, lbfgs->lwork1));
            PetscCall(VecDot(lmvm->Fprev, lbfgs->lwork1, &c_h1));

            PetscCall(MatCDBFGSApplyJ0Fwd(B, lmvm->Xprev, lbfgs->lwork1));
            PetscCall(VecDot(lmvm->Xprev, lbfgs->lwork1, &c_b1));
            /* Adding diagonal element at (0,0) for C_H */
            PetscCall(MatGetValue(lbfgs->StYfull,0,0,&diag00));
            c_h1 += diag00;
            PetscCall(MatSetValue(lbfgs->C_B,0,0,c_b1, INSERT_VALUES));
            PetscCall(MatSetValue(lbfgs->C_H,0,0,c_h1, INSERT_VALUES));

            PetscCall(MatAssemblyBegin(lbfgs->StYfull, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyBegin(lbfgs->C_B, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyBegin(lbfgs->C_H, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(lbfgs->StYfull, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(lbfgs->C_B, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(lbfgs->C_H, MAT_FINAL_ASSEMBLY));
            break; //TODO is this safe?
          }
          /* Update C_H first */
          PetscCall(MatUpdate_C_Matrix(lmvm, lbfgs, lbfgs->C_H, lbfgs->HY, lbfgs->Yfull, lmvm->Fprev));
          /* Update C_B now. */
          PetscCall(MatUpdate_C_Matrix(lmvm, lbfgs, lbfgs->C_B, lbfgs->BS, lbfgs->Sfull, lmvm->Xprev));
          break;
        case (MAT_LBFGS_CD_INPLACE):
          break;
        case (MAT_LBFGS_BASIC):
          SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented CDLBFGS Method");
          break;                  
        }
      }

      /* Compute S^T B S + L D^{-1} L^T
       * J = S^T B S + L D^{-1} L^T */
      Vec workvec1, workvec2;
      const PetscScalar *r_array;
      PetscScalar *x_array, *buffer;

      /* J0 is non-static. Need to manually compute BS matrix. */
      if (!(lmvm->user_scale)) {
        for (i=0; i<lmvm->m; i++) {
          PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, i, &workvec1));
          PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, i, &workvec2));
          PetscCall(MatCDBFGSApplyJ0Fwd(B, workvec1, workvec2));
          PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, i, &workvec1));
          PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, i, &workvec2));
        }
      }
      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->BS, MAT_REUSE_MATRIX, PETSC_DEFAULT, &lbfgs->J));
    
      /* L D^{-1} L^T :  (L_i is ith column of strictly low tri mat. Below, multiply is pointwise mult.
       * [ 0 | L_0*L_0[1]/d_0 | L_0*L_0[2]/d_0 + L_1*L_1[2]/d_1 | ... ].  */
      PetscCall(VecGetArrayRead(lbfgs->diag_vec, &r_array));
      for (i=0; i<lmvm->m-1; i++) {
        PetscCall(MatDenseGetColumnVecRead(lbfgs->StYfull, i, &workvec1));
    
        /* Copying to emulate strictly lower triangular */
        PetscCall(VecCopy(workvec1, lbfgs->rwork1));
        PetscCall(MatDenseRestoreColumnVecRead(lbfgs->StYfull, i, &workvec1));
        PetscCall(VecGetArray(lbfgs->rwork1, &x_array));
        for (j=0; j<i+1; j++) {
          x_array[j] = 0;
        }
    
        /* Creating array for scale = L_i[i+1]/d_0 */
        PetscCall(PetscCalloc1(lmvm->m-i-1, &buffer));
        for (j=0; j < lmvm->m-i-1; j++) {
           //TODO technically we could do adaptive size for k<m, but later..
          if (r_array[i] != 0) {      
            buffer[j] = x_array[i+j+1]/r_array[i];
          } else {
            buffer[j] = 0;
          }
        }
        PetscCall(VecRestoreArray(lbfgs->rwork1, &x_array));
    
        for (j=0, k=i+1; k<lmvm->m; k++, j++) {
          PetscCall(MatDenseGetColumnVecWrite(lbfgs->J, k, &workvec2));
          PetscCall(VecAXPY(workvec2, buffer[j], lbfgs->rwork1));
          PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->J, k, &workvec2));
        }
        PetscCall(PetscFree(buffer));
      }
      PetscCall(VecRestoreArrayRead(lbfgs->diag_vec, &r_array));

      /* Cholesky factorization */
      IS            perm;
      PetscCall(VecGetOwnershipRange(X, &low, &high));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD, low, high, 1, &perm));

      /*TODO LU factor doesn't do anything... WHY??? */
      if (lbfgs->iter_count >= 7) {
        VecSetValue(lbfgs->rwork1,0,-0.0367335,INSERT_VALUES);
        VecSetValue(lbfgs->rwork1,1,0.2606,INSERT_VALUES);
        VecSetValue(lbfgs->rwork1,2,0.0285366,INSERT_VALUES);
        VecSetValue(lbfgs->rwork1,3,-0.0254939,INSERT_VALUES);
        VecSetValue(lbfgs->rwork1,4,-0.0341416,INSERT_VALUES);
      }

      if (lmvm->k == lmvm->m - 1) {
        PetscCall(MatDestroy(&lbfgs->J_solve));
        PetscCall(MatConvert(lbfgs->J, MATSAME, MAT_INITIAL_MATRIX, &lbfgs->J_solve));
        //MatDuplicate(lbfgs->J, MAT_COPY_VALUES, &lbfgs->J_solve);
        PetscCall(MatLUFactor(lbfgs->J_solve,NULL,NULL,NULL));
      } else {
        PetscCall(MatDenseGetSubMatrix(lbfgs->J, 0, lmvm->k+1, 0, lmvm->k+1, &lbfgs->J_work));
        PetscCall(MatDestroy(&lbfgs->J_solve));
        PetscCall(MatDuplicate(lbfgs->J_work, MAT_COPY_VALUES, &lbfgs->J_solve));
        PetscCall(MatLUFactor(lbfgs->J_solve,NULL,NULL,NULL));
        PetscCall(MatDenseRestoreSubMatrix(lbfgs->J, &lbfgs->J_work));
      }

      if (lbfgs->iter_count >= 7) {
        //PetscCall(MatLUFactor(lbfgs->J,NULL,NULL,NULL));//Tried LU on just J.. still nope
        PetscCall(MatSolve(lbfgs->J_solve, lbfgs->rwork1, lbfgs->rwork2));
        MatSetUnfactored(lbfgs->J_solve);
      }
      PetscCall(ISDestroy(&perm)); //TODO tried making regular perm to test LU. doesnt seem to matter. delete later

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
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *blbfgs = (Mat_CDBFGS*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_CDBFGS        *mlbfgs = (Mat_CDBFGS*)mdata->ctx;
  
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
    PetscCall(MatLMVMReset(lbfgs->diag_bfgs, destructive));
  }
  if (lbfgs->allocated && destructive) {
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Yfull));
    PetscCall(MatDestroy(&lbfgs->Sfull));
    PetscCall(MatDestroy(&lbfgs->C_H));
    PetscCall(MatDestroy(&lbfgs->C_B));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->J_work));
    PetscCall(MatDestroy(&lbfgs->HY));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->rwork5));
    PetscCall(VecDestroy(&lbfgs->rwork6));//TODO temp need to delete later
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
    PetscCall(VecDestroy(&lbfgs->diag_vec));
    lbfgs->allocated = PETSC_FALSE;
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscBool         same, allocate = PETSC_FALSE;
  VecType           vec_type;
  PetscInt          m, n, M, N;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  
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
      PetscCall(MatDuplicate(lbfgs->Sfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->HY));
      PetscCall(MatZeroEntries(lbfgs->Sfull));
      PetscCall(MatZeroEntries(lbfgs->Yfull));
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->C_H));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->C_B));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->J));
      PetscCall(MatDuplicate(lbfgs->StYfull, MAT_DO_NOT_COPY_VALUES, &lbfgs->J_work));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->diag_vec, NULL));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork3, &lbfgs->rwork4));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork5, &lbfgs->rwork6));//TODO temp. need to delete later
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork1));
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->lwork2));
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  if (lbfgs->allocated) {
    PetscCall(MatDestroy(&lbfgs->StYfull));
    PetscCall(MatDestroy(&lbfgs->Sfull));
    PetscCall(MatDestroy(&lbfgs->Yfull));
    PetscCall(MatDestroy(&lbfgs->C_H));
    PetscCall(MatDestroy(&lbfgs->C_B));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->J_work));
    PetscCall(MatDestroy(&lbfgs->HY));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->rwork5));
    PetscCall(VecDestroy(&lbfgs->rwork6));//TODO temp need to delete later
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  
  PetscInt          m, n, M, N;
  PetscMPIInt       size;
  MPI_Comm          comm = PetscObjectComm((PetscObject)B);
  Vec               Xtmp, Ftmp;

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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscBool         isascii;

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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_CDBFGS        *lbfgs = (Mat_CDBFGS*)lmvm->ctx;

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
  Mat_LMVM          *lmvm;
  Mat_CDBFGS        *lbfgs;

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
  lbfgs->idx_begin       = 0;
  lbfgs->watchdog        = 0;
  lbfgs->delta           = 1.0;
  lbfgs->delta_min       = 1e-7;
  lbfgs->delta_max       = 100.0;
  lbfgs->max_seq_rejects = lmvm->m/2;
  lbfgs->strategy        = MAT_LBFGS_CD_REORDER;

  lbfgs->iter_count = 0;//TODO remove later
  
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
