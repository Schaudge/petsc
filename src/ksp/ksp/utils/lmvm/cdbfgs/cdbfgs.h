#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_bfgs;                                   /* diagonalized Hessian init */

  PetscInt  idx_begin;                                   // index of the oldest colums in Sfull and Yfull
  Mat       Sfull, Yfull;                                // Stored in recycled order
  Mat       Wfull;                                       // J_0 Y
  Mat       Rbar;                                        // logically upper-triangular (wrt history ordering) of -S^T Y
  Mat       C;                                           // diag(S^T Y) + Y^T J_0 Y)
  PetscInt  watchdog, max_seq_rejects;                   /* tracker to reset after a certain # of consecutive rejects */
  Vec       work_0, work_1, work_2;
  PetscBool allocated;
  PetscReal delta, delta_min, delta_max;
  MatType   dense_type;
} Mat_CDBFGS;

PETSC_INTERN PetscErrorCode MatView_LMVMCDBFGS(Mat, PetscViewer);
