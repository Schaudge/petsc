#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

typedef enum {
  MATLBFGS_BASIC, // vector by vector
  MATLBFGS_CD_REORDER, // compact dense, reordering for a single trsm
  MATLBFGS_CD_INPLACE, // compact dense, simulate trsm in recycle order
}
MatLBFGSType;
PETSC_INTERN const char *const MatLBFGSTypes[];

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_bfgs;                                   /* diagonalized Hessian init */

  PetscInt  idx_begin;                                   // index of the oldest colums in S and Y
  Mat       S, Y, H0_Y;                                  // Stored in recycled order
  Mat       Q;                                           // H0 * Y
  Mat       StY;
  Mat       C;                                           // diag(S^T Y) + Y^T H_0 Y)
  Vec       work_0, work_1, work_2;
  MatType   dense_type;
  PetscScalar *alphas;
  PetscScalar *betas;
  PetscScalar *StYdiag;
  MatLBFGSType strategy;

  PetscInt  watchdog, max_seq_rejects;                   /* tracker to reset after a certain # of consecutive rejects */
  PetscBool allocated;
  PetscReal delta, delta_min, delta_max;
} Mat_CDBFGS;

PETSC_INTERN PetscErrorCode MatView_LMVMCDBFGS(Mat, PetscViewer);
