#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_dfp;                                   /* diagonalized Hessian init */

  PetscInt  num_updates;
  PetscInt  num_mult_updates;
  Mat       Sfull, Yfull, BS; // Stored in recycled order
  Mat       StY_triu;        // triu(StY) is the R matrix
  Vec       StFprev;
  Vec       YtXprev;
  Mat       YtS_triu_strict; // strict_triu(YtS) is the L^T matrix
  Mat       LDLt;
  Mat       StBS;
  Mat       J;
  Mat       temp_mat;
  Vec       diag_vec;
  Vec       diag_vec_recycle_order;
  Vec       inv_diag_vec;
  Vec       column_work, rwork1, rwork2, rwork3;
  Vec       rwork2_local, rwork3_local;
  Vec       local_work_vec, local_work_vec_copy;
  Vec       cyclic_work_vec;
  MatType   dense_type;
  MatLDFPType strategy;
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
