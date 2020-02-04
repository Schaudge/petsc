#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Compact dense representation for the limited-memory BFGS method.
*/

typedef struct {
  Mat       diag_bfgs;                        /* diagonalized Hessian init */
  Mat       STfull, YTfull;                   /* large matrices (m x n) */
  Mat       StYfull, Lfull, Dfull, Rfull;     /* small matrices (m x m) */
  Mat       ST, YT, StY, L, D, R, Rinv;       /* submatrices that span only the existing updates */
  Vec       rwork1, rwork2, rwork3, rwork4;   /* small work vectors (m) matching submatrices */
  Vec       lwork1, lwork2;                   /* large work vectors (n) */
  PetscInt  watchdog, max_seq_rejects;        /* tracker to reset after a certain # of consecutive rejects */
  PetscBool allocated;
  PetscInt *idx_cols;
  PetscInt *idx_rows;
  PetscReal delta, delta_min, delta_max;
  MatType   dense_type;
} Mat_CDBFGS;

PETSC_INTERN PetscErrorCode MatView_LMVMCDBFGS(Mat, PetscViewer);
