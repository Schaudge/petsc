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

/*------------------------------------------------------------*/

/* Solves triangular matrix, stored in recycled order.
 * One can solve for lower triangle, transpose of lower triangle, 
 * upper triangle, and transpoe of upper triangle. 
 * It assumes the input matrix is square matrix, n x n, in recycled order.
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
static PetscErrorCode MatSolveTriangularRecycleOrder(Mat_LMVM *lmvm, Mat_CDBFGS *lbfgs, Mat R, PetscInt lowest_index, Vec b, Vec x, TriangularTypes tri_type)
{
  MPI_Comm     comm  = PetscObjectComm((PetscObject)R);
  PetscScalar  Alpha = 1.0, neg_one = -1.;
  PetscMemType memtype_r, memtype_x;
  PetscScalar *buffer, *array_write, *x_array;
  PetscInt     lda;

  const PetscScalar *array_read, *r_array;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayReadAndMemType(R, &r_array, &memtype_r));
  PetscCall(VecGetArrayWriteAndMemType(x, &x_array, &memtype_x));
  PetscCall(MatDenseGetLDA(R, &lda));
  PetscAssert(memtype_x == memtype_r, comm, PETSC_ERR_PLIB, "Incompatible device pointers");

  switch (lbfgs->strategy) {
  case MAT_LBFGS_CD_REORDER:
    {
      /* Shift b vector */
      PetscCall(VecGetArrayRead(b, &array_read));
      PetscCall(PetscMalloc1(lmvm->m, &buffer));
      PetscCall(PetscMemcpy(buffer, &array_read[lbfgs->idx_begin], (lmvm->m - lbfgs->idx_begin)*sizeof(PetscScalar)));
      if (lbfgs->idx_begin != 0 ) {
        PetscCall(PetscMemcpy(&buffer[lmvm->m - lbfgs->idx_begin], array_read, (lbfgs->idx_begin)*sizeof(PetscScalar)));
      }
      PetscCall(VecRestoreArrayReadAndMemType(b, &array_read));
  
      PetscCall(VecGetArrayWrite(b, &array_write));
      PetscCall(PetscMemcpy(array_write, &buffer, (lmvm->m)*sizeof(PetscScalar)));
      PetscCall(VecRestoreArrayWriteAndMemType(b, &array_write));
      PetscCall(PetscFree(buffer));

      switch (memtype_x) {
      case PETSC_MEMTYPE_HOST:
        /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
        {
          //PetscAssert(PetscDefined(BLAS)...));
          PetscBLASInt m_blas, lda_blas, one = 1;
          PetscCall(PetscBLASIntCast(lmvm->k, &m_blas));
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
  /* Start with the H0 term */
  PetscCall(MatCDBFGSApplyJ0Inv(H, F, dX));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }


  /* Start with upper part : -H Y R^-1 S^T F */
  /* Start with reusable part: rwork1 = S^T F, rwork2 = R^-1 S^T F */
  PetscCall(MatMultTranspose(lbfgs->Sfull, F, lbfgs->rwork1));
  PetscCall(MatSolveTriangularRecycleOrder(lmvm, lbfgs, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork1, lbfgs->rwork2, MAT_CDBFGS_UPPER_TRIANGULAR));
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(dX, -1., lbfgs->lwork2));

  /* Start bottom part: S(-R^-T Y^T H S^T F + R^-T(D + Y^T H Y) R^-1 S^T F)
   * rwork3: D R^-1 S^T F, rwork1: Y^T H Y R^-1 S^T F                       */
  PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->diag_vec));
  PetscCall(MatGetDiagonal(lbfgs->StYfull, lbfgs->diag_vec));
  PetscCall(VecPointwiseMult(lbfgs->rwork3, lbfgs->diag_vec, lbfgs->rwork2));
  PetscCall(VecDestroy(&lbfgs->diag_vec));

  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Inv(H, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(MatMultTranspose(lbfgs->Yfull, lbfgs->lwork2, lbfgs->rwork1));
  PetscCall(VecAXPY(lbfgs->rwork1, 1., lbfgs->rwork3));

  PetscCall(MatCDBFGSApplyJ0Inv(H, F, lbfgs->lwork1));
  PetscCall(MatMultTranspose(lbfgs->Yfull, lbfgs->lwork1, lbfgs->rwork2));
  PetscCall(VecAXPY(lbfgs->rwork1, -1., lbfgs->rwork2));
  PetscCall(MatSolveTriangularRecycleOrder(lmvm, lbfgs, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork1, lbfgs->rwork2, MAT_CDBFGS_UPPER_TRIANGULAR_TRANSPOSE));
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(VecAXPY(dX, -1., lbfgs->lwork1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for 
 * B_0 - [B_S S | Y] [  0   |           L^-1          ] [S^T B]
 *                   [ L^-T | -L^-T (S^T B S + D) L^-1] [ Y^T ] */

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM   *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  
  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Start with the B0 term */
  PetscCall(MatCDBFGSApplyJ0Fwd(B, X, Z));
  if (lmvm->k == -1) {
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  /* rwork4 = S^T B X. Using the fact that Z is still B X */
  PetscCall(MatMultTranspose(lbfgs->Sfull, Z, lbfgs->rwork4));

  /* Upper half:  B S L^-1 Y^T X
   * rwork1: Y^T X, rwork2: L^-1 Y^T X           */
  PetscCall(MatMultTranspose(lbfgs->Yfull, X, lbfgs->rwork1));
  PetscCall(MatSolveTriangularRecycleOrder(lmvm, lbfgs, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork1, lbfgs->rwork2, MAT_CDBFGS_LOWER_TRIANGULAR));
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(VecAXPY(Z, -1., lbfgs->lwork2));

  /* Bottom half                               *
   * Y L^-T (S^T B - (S^T B S + D) L^-1 Y^T)X  *
   * rwork1: (S^T B S)L^-1 Y^T X               */
  PetscCall(MatMult(lbfgs->Sfull, lbfgs->rwork2, lbfgs->lwork1));
  PetscCall(MatCDBFGSApplyJ0Fwd(B, lbfgs->lwork1, lbfgs->lwork2));
  PetscCall(MatMultTranspose(lbfgs->Sfull, lbfgs->lwork2, lbfgs->rwork1));

  /* rwork2: D L^-1 Y^T X */
  PetscCall(VecDuplicate(lbfgs->rwork1, &lbfgs->diag_vec));
  PetscCall(MatGetDiagonal(lbfgs->StYfull, lbfgs->diag_vec));
  PetscCall(VecPointwiseMult(lbfgs->rwork3, lbfgs->diag_vec, lbfgs->rwork2));
  PetscCall(VecDestroy(&lbfgs->diag_vec));
  /* Adding them all up into rwork4 */
  PetscCall(VecAXPBYPCZ(lbfgs->rwork4, -1., -1., 1., lbfgs->rwork1, lbfgs->rwork2));

  PetscCall(MatSolveTriangularRecycleOrder(lmvm, lbfgs, lbfgs->StYfull, lbfgs->idx_begin, lbfgs->rwork4, lbfgs->rwork1, MAT_CDBFGS_LOWER_TRIANGULAR_TRANSPOSE));
  PetscCall(MatMult(lbfgs->Yfull, lbfgs->rwork1, lbfgs->lwork1));

  PetscCall(VecAXPY(Z, -1., lbfgs->lwork1));
  //TODO can't tell whether i should reorder solution vec here, or elsewhere?
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM*)B->data;
  Mat_CDBFGS   *lbfgs = (Mat_CDBFGS*)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;
  
  const PetscScalar *xx, *ff, *array_read;
  PetscScalar       curvature, ststmp, *buffer, *array_write;
  PetscReal         curvtol;
  PetscInt          n, low, high, i;
  PetscMemType      memtype_sy;
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
      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      PetscCall(VecGetLocalSize(lmvm->Xprev, &n));
      PetscCall(VecGetOwnershipRange(lmvm->Xprev, &low, &high));
      PetscCall(PetscMalloc2(n, &lbfgs->idx_rows, 1, &lbfgs->idx_cols));
      for (i=low; i<high; i++) {
        lbfgs->idx_rows[i] = i;
      }

      switch (lbfgs->strategy) {
      case (MAT_LBFGS_CD_REORDER):
        if (lmvm->k == lmvm->m-1) {
          /* S Matrix is full. Shift matrix via memcpy */
          PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Sfull, &array_read, &memtype_sy));
          // Assert(lda == m)
          PetscCall(PetscMalloc1(lmvm->m*lmvm->m - lmvm->m - 1, &buffer));
          PetscCall(PetscMemcpy(buffer, &array_read[lmvm->m+1], (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Sfull, &array_read));
  
          PetscCall(MatDenseGetArrayWriteAndMemType(lbfgs->Sfull, &array_write, &memtype_sy));
          PetscCall(PetscMemcpy(array_write, &buffer, (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayWriteAndMemType(lbfgs->Sfull, &array_write));
          PetscCall(PetscFree(buffer));

          /* Y Matrix is full. Shift matrix via memcpy */
          PetscCall(MatDenseGetArrayReadAndMemType(lbfgs->Yfull, &array_read, &memtype_sy));
          // Assert(lda == m)
          PetscCall(PetscMalloc1(lmvm->m*lmvm->m - lmvm->m - 1, &buffer));
          PetscCall(PetscMemcpy(buffer, &array_read[lmvm->m+1], (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayReadAndMemType(lbfgs->Yfull, &array_read));
  
          PetscCall(MatDenseGetArrayWriteAndMemType(lbfgs->Yfull, &array_write, &memtype_sy));
          PetscCall(PetscMemcpy(array_write, &buffer, (lmvm->m*lmvm->m - lmvm->m - 1)*sizeof(memtype_sy)));
          PetscCall(MatDenseRestoreArrayWriteAndMemType(lbfgs->Yfull, &array_write));
          PetscCall(PetscFree(buffer));
        } else {
          lmvm->k = lmvm->k + 1;
        }
        lbfgs->idx_cols[0] = lmvm->k;
        break;
      case (MAT_LBFGS_CD_INPLACE):
        /* Inplace doesn't move memory, but rather only finds index of oldest memory */
        if (lmvm->k == lmvm->m-1) {
          lbfgs->idx_begin = (lbfgs->idx_begin + 1) % lmvm->m;
          lbfgs->idx_cols[0] = lbfgs->idx_begin;
        } else {
          lmvm->k = lmvm->k + 1;
          lbfgs->idx_cols[0] = lmvm->k;
        }
        break;
      case (MAT_LBFGS_BASIC):
        SETERRQ(comm, PETSC_ERR_SUP, "Unimplemented CDLBFGS Method");
        break;
      }

      /* First update the S^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Xprev, &xx));
      PetscCall(MatSetValues(lbfgs->Sfull, n, lbfgs->idx_rows, 1, lbfgs->idx_cols, xx, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->Sfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Sfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &xx));
      /* Now repeat update for the Y^T matrix */
      PetscCall(VecGetArrayRead(lmvm->Fprev, &ff));
      PetscCall(MatSetValues(lbfgs->Yfull, n, lbfgs->idx_rows, 1, lbfgs->idx_cols, ff, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(lbfgs->Yfull, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(lbfgs->Yfull, MAT_FINAL_ASSEMBLY));
      PetscCall(VecRestoreArrayRead(lmvm->Xprev, &ff));
      /* Clean up unnecessary arrays */
      PetscCall(PetscFree2(lbfgs->idx_rows, lbfgs->idx_cols));

      PetscCall(MatDestroy(&lbfgs->StYfull));
      PetscCall(MatTransposeMatMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
//      PetscCall(MatConvert(lbfgs->StYfull, lbfgs->dense_type, MAT_INPLACE_MATRIX, &lbfgs->StYfull));//TODO is this needed, or done internally already?
      /* Clear out the previously formed submatrices and work vectors */
      PetscCall(VecDestroy(&lbfgs->rwork1));
      PetscCall(VecDestroy(&lbfgs->rwork2));
      PetscCall(VecDestroy(&lbfgs->rwork3));
      PetscCall(VecDestroy(&lbfgs->rwork4));

      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork1, &lbfgs->rwork2));
      PetscCall(MatCreateVecs(lbfgs->StYfull, &lbfgs->rwork3, &lbfgs->rwork4));
// TODO : make clear whether workvectors are L or R

      /* Update the diagonal H0 if it exists */
      if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) {
        PetscCall(MatLMVMUpdate(lbfgs->diag_bfgs, X, F));
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
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
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
      PetscCall(MatZeroEntries(lbfgs->Sfull));
      PetscCall(MatZeroEntries(lbfgs->Yfull));
      /* Create intermediate (sequential and small) matrices */
      //TODO: NOTE: "MMTM: This routine is currently only implemented for pairs of MATSEQAIJ matrices, for the MATSEQDENSE class, and for pairs of MATMPIDENSE matrices."
      PetscCall(MatMatTransposeMult(lbfgs->Sfull, lbfgs->Yfull, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lbfgs->StYfull));
//      PetscCall(MatZeroEntries(lbfgs->StYfull));//THIS CHANGES DENSE TO REGULAR.... FROM SEQDENSE TO SEQ
    }
      MatType stype, ytype, stytype;
      MatGetType(lbfgs->StYfull, &stytype);
      MatGetType(lbfgs->Sfull, &stype);
      MatGetType(lbfgs->Yfull, &ytype);
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
    PetscCall(VecDestroy(&lbfgs->rwork1));
    PetscCall(VecDestroy(&lbfgs->rwork2));
    PetscCall(VecDestroy(&lbfgs->rwork3));
    PetscCall(VecDestroy(&lbfgs->rwork4));
    PetscCall(VecDestroy(&lbfgs->lwork1));
    PetscCall(VecDestroy(&lbfgs->lwork2));
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
