#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_bfgs;                                   /* diagonalized Hessian init */

  PetscInt  idx_begin;                                   // index of the oldest colums in Sfull and Yfull
  Mat       Sfull, Yfull, StYfull;                                // Stored in recycled order
  Mat       Wfull;                                       // J_0 Y
  Mat       StY;
  Mat       C;                                           // diag(S^T Y) + Y^T J_0 Y)
  Vec       work_0, work_1, work_2;
  MatType   dense_type;
  MatLBFGSType strategy;

  PetscInt  watchdog, max_seq_rejects;                   /* tracker to reset after a certain # of consecutive rejects */
  PetscInt *idx_cols, *idx_rows;
  PetscBool allocated;
  PetscReal delta, delta_min, delta_max;
} Mat_CDBFGS;

PETSC_INTERN PetscErrorCode MatView_LMVMCDBFGS(Mat, PetscViewer);
