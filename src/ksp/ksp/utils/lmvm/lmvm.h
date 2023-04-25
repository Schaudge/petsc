#ifndef __LMVM_H
#define __LMVM_H
#include <petscksp.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>

/*
  MATLMVM format - a base matrix-type that represents Limited-Memory
  Variable Metric (LMVM) approximations of a Jacobian.

  LMVM approximations can be symmetric, symmetric positive-definite,
  rectangular, or otherwise square with no determinable properties.
  Each derived LMVM type should automatically set its matrix properties
  if its construction can guarantee symmetry (MAT_SYMMETRIC) or symmetric
  positive-definiteness (MAT_SPD).
*/

typedef struct _MatOps_LMVM *MatOps_LMVM;
struct _MatOps_LMVM {
  PetscErrorCode (*update)(Mat, Vec, Vec);
  PetscErrorCode (*allocate)(Mat, Vec, Vec);
  PetscErrorCode (*reset)(Mat, PetscBool);
  PetscErrorCode (*mult)(Mat, Vec, Vec);
  PetscErrorCode (*copy)(Mat, Mat, MatStructure);
};

// Limited Memory Window of Vectors
typedef struct _LMWindowVecs *LMWindowVecs;
struct _LMWindowVecs  {
  PetscInt m; // Number of vectors in the limited memory window
  PetscInt k; // Index of the history-order next vector to be inserted
  Mat vecs;   // Dense matrix backing storage of vectors
  Vec single;
  PetscScalar *write_placed_array;
  const PetscScalar *read_placed_array;
  PetscInt placed_idx;
  PetscMemType memtype;
  PetscObjectState op_state;
};

// LMWindowVecs in LMVM
typedef enum {
  LMVM_WINDOWVECS_Y          = 0x01,
  LMVM_WINDOWVECS_P          = 0x02, // J_0 S
  LMVM_WINDOWVECS_YMP        = 0x04, // Y - P
  LMVM_WINDOWVECS_YTYPE      = 0x07, // vector types like the range of J0
  LMVM_WINDOWVECS_S          = 0x08,
  LMVM_WINDOWVECS_Q          = 0x10, // J_0^{-1} Y
  LMVM_WINDOWVECS_SMQ        = 0x20, // S - Q
  LMVM_WINDOWVECS_STYPE      = 0x38, // vector types like the domain of J0
  LMVM_WINDOWVECS_PAIR_LIMIT = 0x25  // 1 greater than the largest YTYPE | STYPE
} LMVMWindowVecsType;

// Dot product types to keep in LMWindowVecs
typedef enum {
  LMVM_RVDOT_DIAG         = 0x1,
  LMVM_RVDOT_STRICT_LOWER = 0x2,
  LMVM_RVDOT_STRICT_UPPER = 0x4,
  LMVM_RVDOT_LOWER        = 0x3,
  LMVM_RVDOT_UPPER        = 0x5,
  LMVM_RVDOT_ALL          = 0x7,
} LMVMWindowVecsDotType;

PETSC_INTERN PetscErrorCode LMWindowVecsCreate(Vec, PetscInt, LMWindowVecs *);
PETSC_INTERN PetscErrorCode LMWindowVecsDestroy(LMWindowVecs *);

PETSC_INTERN PetscErrorCode LMWindowVecsGetNextWrite(LMWindowVecs, Vec *);
PETSC_INTERN PetscErrorCode LMWindowVecsRestoreNextWrite(LMWindowVecs, Vec *);
PETSC_INTERN PetscErrorCode LMWindowVecsGetSingleRead(LMWindowVecs, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMWindowVecsRestoreSingleRead(LMWindowVecs, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMWindowVecsGetSingleWrite(LMWindowVecs, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMWindowVecsRestoreSingleWrite(LMWindowVecs, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMVMWindowVecsRestoreSingleWrite(LMWindowVecs, PetscInt, Vec *);

typedef struct {
  /* Core data structures for stored updates */
  struct _MatOps_LMVM ops[1];
  PetscBool allocated, prev_set;
  PetscInt k_next;                                               // History index of the next vector to be inserted
  PetscInt  m_old, m, nupdates, nrejects, nresets;
  LMVMWindowVecsType   auto_vecs;                                // bit-mask of automatically updated windows
  LMWindowVecs         S, Y, Q, P, SmQ, YmP;
  PetscScalar            (*rv_dots)[LMVM_WINDOWVECS_PAIR_LIMIT]; // automatically updated dot products
  LMVMWindowVecsDotType dot_types[LMVM_WINDOWVECS_PAIR_LIMIT];   // which dot products in the history to perform
  Vec       Xprev, Fprev;

  /* User-defined initial Jacobian tools */
  PetscObjectState J0_state;
  Mat              J0;
  KSP              J0ksp;

  /* Miscellenous parameters */
  PetscBool square; /* flag for defining the LMVM approximation as a square matrix */
  PetscReal eps;    /* (default: PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0)) */
  void     *ctx;    /* implementation specific context */
} Mat_LMVM;

/* Shared internal functions for LMVM matrices */
PETSC_INTERN PetscErrorCode MatUpdateKernel_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatUpdate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatAllocate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatReset_LMVM(Mat, PetscBool);
PETSC_INTERN PetscErrorCode MatLMVMGetWindow(Mat, PetscInt *, PetscInt *);

/* LMVM implementations of core Mat functionality */
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVM(Mat, PetscOptionItems *PetscOptionsObject);
PETSC_INTERN PetscErrorCode MatSetUp_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatView_LMVM(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatDestroy_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatCreate_LMVM(Mat);

/* Create functions for derived LMVM types */
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDFP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBFGS(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSR1(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDiagBrdn(Mat);

/* Solve functions for derived LMVM types (necessary only for DFP and BFGS for re-use under SymBrdn) */
PETSC_INTERN PetscErrorCode MatSolve_LMVMDFP(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSolve_LMVMBFGS(Mat, Vec, Vec);

/* Mult functions for derived LMVM types (necessary only for DFP and BFGS for re-use under SymBrdn) */
PETSC_INTERN PetscErrorCode MatMult_LMVMDFP(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatMult_LMVMBFGS(Mat, Vec, Vec);

#endif
