#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_bfgs;                                   /* diagonalized Hessian init */

  PetscInt  idx_begin;                                   // index of the oldest colums in Sfull and Yfull
  Mat       Sfull, Yfull, StYfull;                                // Stored in recycled order
  Mat       Q; // H_0 Y
  Mat       LDLT, BS; //S^T B_0 S (m x m)                 
  Mat       L, J, J_work, J_solve, J_temp_copy;
  Vec       diag_vec;
  Vec       lwork1, lwork2, rwork1, rwork2, rwork3, rwork4;
  Vec       rwork5,rwork6; //temp. need to delete later
  Vec       s_in_S, y_in_Y, q_in_Q;
  MatType   dense_type;
  MatLBFGSType strategy;

  PetscInt  watchdog, max_seq_rejects;                   /* tracker to reset after a certain # of consecutive rejects */
  PetscInt  idx_cols;
  PetscBool allocated;
  PetscReal delta, delta_min, delta_max;
} Mat_CDBFGS;

PETSC_INTERN PetscErrorCode MatView_LMVMCDBFGS(Mat, PetscViewer);
