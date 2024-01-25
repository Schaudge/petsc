#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory DFP method.
*/

typedef struct {
  Mat       diag_dfp;                                   /* diagonalized Hessian init */

  PetscInt  num_updates;
  PetscInt  num_mult_updates;
  Mat       Sfull, Yfull, HY; // Stored in recycled order
  Mat       StY_triu;        // triu(StY) is the R matrix
  Vec       StFprev;
  Vec       YtXprev;
  Mat       YtS_triu_strict; // strict_triu(YtS) is the L^T matrix
  Mat       LDLt;
  Mat       YtHY;
  Mat       J;
  Mat       temp_mat;
  Vec       diag_vec;
  Vec       diag_vec_recycle_order;
  Vec       inv_diag_vec;
  Vec       column_work, rwork1, rwork2, rwork3, rwork4;
  Vec       rwork2_local, rwork3_local;
  Vec       local_work_vec, local_work_vec_copy;
  Vec       cyclic_work_vec;
  MatType   dense_type;
  MatLMVMCompactDenseType strategy;
  MatLMVMSymBroydenScaleType scale_type;

  PetscInt         S_count, St_count, Y_count, Yt_count;
  PetscInt         watchdog, max_seq_rejects;                   /* tracker to reset after a certain # of consecutive rejects */
  PetscBool        allocated;
  PetscBool        mult_type;
  Vec              Fprev_ref;
  PetscObjectState Fprev_state;
  Vec              Xprev_ref;
  PetscObjectState Xprev_state;
} Mat_CDDFP;

PETSC_INTERN PetscErrorCode MatView_LMVMCDDFP(Mat, PetscViewer);

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace_CUPM(PetscBool, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt);
PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlaceCyclic_CUPM(PetscBool, PetscInt, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt);
PETSC_INTERN PETSC_UNUSED PetscErrorCode MatMultColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultTransposeColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultTransposeAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode VecCyclicShift(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode VecRecycleOrderToHistoryOrder(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode VecHistoryOrderToRecycleOrder(Mat, Vec, PetscInt, Vec);
PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace(Mat, Mat, Vec, PetscBool, PetscInt, MatLMVMCompactDenseType);
PETSC_INTERN PetscErrorCode MatMove_LR3(Mat, Mat, PetscInt, Mat);
