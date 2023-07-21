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
  PetscErrorCode (*multht)(Mat, Vec, Vec);
  PetscErrorCode (*solve)(Mat, Vec, Vec);
  PetscErrorCode (*solveht)(Mat, Vec, Vec);
  PetscErrorCode (*copy)(Mat, Mat, MatStructure);
};

typedef struct _VecLink *VecLink;

struct _VecLink {
  Vec     vec;
  VecLink next;
};

typedef struct _RowLink *RowLink;

struct _RowLink {
  PetscScalar *row;
  RowLink      next;
};

// Limited Memory Basis
typedef struct _LMBasis *LMBasis;
struct _LMBasis {
  PetscInt         m;              // Number of vectors in the limited memory window
  PetscInt         k;              // Index of the history-order next vector to be inserted
  Mat              vecs;           // Dense matrix backing storage of vectors
  PetscObjectId    operator_id;    // If these vecs include the output of an operator (like B0 * S), the id of the operator
  PetscObjectState operator_state; // If these vecs include the output of an operator (like B0 * S), the id of the operator
  VecLink          work_vecs_available;
  VecLink          work_vecs_in_use;
  RowLink          work_rows_available;
  RowLink          work_rows_in_use;
};

PETSC_INTERN PetscErrorCode LMBasisCreate(Vec, PetscInt, LMBasis *);
PETSC_INTERN PetscErrorCode LMBasisDestroy(LMBasis *);
PETSC_INTERN PetscErrorCode LMBasisReset(LMBasis);

PETSC_INTERN PetscErrorCode LMBasisGetNextVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreNextVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisGetVec(LMBasis, PetscInt, PetscMemoryAccessMode, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreVec(LMBasis, PetscInt, PetscMemoryAccessMode, Vec *);
PETSC_INTERN PetscErrorCode LMBasisCopy(LMBasis, LMBasis);

PETSC_INTERN PetscErrorCode LMBasisGetWorkRow(LMBasis, PetscScalar **);
PETSC_INTERN PetscErrorCode LMBasisRestoreWorkRow(LMBasis, PetscScalar **);
PETSC_INTERN PetscErrorCode LMBasisGetWorkVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreWorkVec(LMBasis, Vec *);

PETSC_INTERN PetscErrorCode LMBasisGetRange(LMBasis, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode LMBasisGEMV(PetscScalar, LMBasis, PetscInt, PetscInt, PetscScalar[], PetscScalar, Vec);
PETSC_INTERN PetscErrorCode LMBasisGEMVH(PetscScalar, LMBasis, PetscInt, PetscInt, Vec, PetscScalar, PetscScalar[]);

// Refers to blocks of LMGramians
typedef enum {
  LMBLOCK_DIAGONAL,
  LMBLOCK_LOWER_TRIANGLE,
  LMBLOCK_UPPER_TRIANGLE,
  LMBLOCK_ALL,
  LMBLOCK_FACTORIZATION,
  LMBLOCK_END,
  LMBLOCK_STRICT_LOWER_TRIANGLE,
  LMBLOCK_STRICT_UPPER_TRIANGLE,
} LMBlockType;

typedef enum {
  LMSOLVE_LU,
  LMSOLVE_UPPER_TRIANGLE,
  LMSOLVE_LOWER_TRIANGLE,
  LMSOLVE_HPD_UPPER,
  LMSOLVE_HPD_LOWER,
  LMSOLVE_HERMITIAN_INDEFINITE_UPPER,
  LMSOLVE_HERMITIAN_INDEFINITE_LOWER,
  LMSOLVE_DIAGONAL
} LMSolveType;

static inline PETSC_UNUSED LMBlockType LMBlockTypeFromSolveType(LMSolveType s)
{
  // clang-format off
  switch(s) {
  case LMSOLVE_LU:               return LMBLOCK_ALL;
  case LMSOLVE_HPD_UPPER:
  case LMSOLVE_HERMITIAN_INDEFINITE_UPPER:
  case LMSOLVE_UPPER_TRIANGLE:   return LMBLOCK_UPPER_TRIANGLE;
  case LMSOLVE_HPD_LOWER:
  case LMSOLVE_HERMITIAN_INDEFINITE_LOWER:
  case LMSOLVE_LOWER_TRIANGLE:   return LMBLOCK_LOWER_TRIANGLE;
  case LMSOLVE_DIAGONAL:         return LMBLOCK_DIAGONAL;
  }
  // clang-format on
  return LMBLOCK_ALL;
}

typedef struct _LMBlockStatus {
  PetscInt         next; // one after the last entry that has been computed
  PetscObjectId    operator_id;
  PetscObjectState operator_state;
} LMBlockStatus;

// (cross-)Gramians of inner products of LMBasis vectors
typedef struct _LMGramian *LMGramian;
struct _LMGramian {
  PetscInt      m;
  PetscInt      k;
  PetscInt      lda;
  PetscScalar  *full;
  PetscScalar  *diagonal;
  PetscScalar  *diagonal_copy;
  PetscScalar  *factorization;
  PetscBLASInt *pivots;
  PetscBLASInt  lwork;
  PetscScalar  *work;
  LMBlockStatus status[LMBLOCK_END];
};

PETSC_INTERN PetscErrorCode LMGramianCreate(PetscInt, LMGramian *);
PETSC_INTERN PetscErrorCode LMGramianDestroy(LMGramian *);
PETSC_INTERN PetscErrorCode LMGramianReset(LMGramian);
PETSC_INTERN PetscErrorCode LMGramianUpdateNextIndex(LMGramian, PetscInt);
PETSC_INTERN PetscErrorCode LMGramianInsertDiagonalValue(LMGramian, PetscInt, PetscScalar);
PETSC_INTERN PetscErrorCode LMGramianGetDiagonalValue(LMGramian, PetscInt, PetscScalar *);
PETSC_INTERN PetscErrorCode LMGramianUpdateBlock(LMGramian, LMBasis, LMBasis, LMBlockType);
PETSC_INTERN PetscErrorCode LMGramianForceUpdateBlock(LMGramian, LMBasis, LMBasis, LMBlockType, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode LMGramianGetBlock(LMGramian, LMBasis, LMBasis, LMBlockType, const PetscScalar *[], PetscInt *);
PETSC_INTERN PetscErrorCode LMGramianBlockRead(LMGramian, LMBlockType, const PetscScalar *[], PetscInt *);
PETSC_INTERN PetscErrorCode LMGramianCopy(LMGramian, LMGramian);
PETSC_INTERN PetscErrorCode LMGramianAXPY(LMGramian, LMBlockType, PetscScalar, LMGramian, LMBlockType);
PETSC_INTERN PetscErrorCode LMGramianSolve(LMGramian, PetscInt, PetscInt, LMSolveType, PetscScalar[], PetscBool);

typedef enum {
  LMBASIS_S           = 0,
  LMBASIS_Y           = 1,
  LMBASIS_H0Y         = 2,
  LMBASIS_B0S         = 3,
  LMBASIS_S_MINUS_H0Y = 4,
  LMBASIS_Y_MINUS_B0S = 5,
  LMBASIS_END
} MatLMVMBasisType;

typedef enum {
  MATLMVM_MODE_PRIMAL = 0,
  MATLMVM_MODE_DUAL   = 1,
} MatLMVMMode;

#define LMVMModeMap(a, mode) ((a) ^ mode)

static inline PETSC_UNUSED MatLMVMBasisType MatLMVMBasisSizeOf(MatLMVMBasisType type)
{
  return (type & LMBASIS_Y);
}

typedef enum {
  MATLMVM_MATVEC_RECURSIVE,
  MATLMVM_MATVEC_COMPACT_DENSE,
} MatLMVMMatvecType;

typedef enum {
  MATLMVM_ORDER_HISTORY,
  MATLMVM_ORDER_RECYCLED,
} MatLMVMOrderType;

extern const char *const MatLMVMMatvecTypes[];

typedef struct {
  /* Core data structures for stored updates */
  struct _MatOps_LMVM ops[1];
  PetscBool           allocated, prev_set;
  PetscInt            m, k, nupdates, nrejects, nresets;
  LMBasis             basis[LMBASIS_END];
  LMGramian           gramian[LMBASIS_END][LMBASIS_END];
  Vec                 Xprev, Fprev;

  /* User-defined initial Jacobian tools */
  PetscScalar shift;
  Mat         J0;
  KSP         J0ksp;
  PetscBool   disable_ksp_viewers;

  /* Miscellenous parameters */
  PetscBool         square; /* flag for defining the LMVM approximation as a square matrix */
  PetscReal         eps;    /* (default: PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0)) */
  MatLMVMMatvecType matvec_type;
  MatLMVMOrderType  order;
  PetscBool         do_not_cache_J0_products;
  void             *ctx; /* implementation specific context */
} Mat_LMVM;

/* Shared internal functions for LMVM matrices */
PETSC_INTERN PetscErrorCode MatUpdateKernel_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatUpdate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatAllocate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatReset_LMVM(Mat, PetscBool);

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

PETSC_INTERN PetscErrorCode MatLMVMGetJ0InvDiag(Mat, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMRestoreJ0InvDiag(Mat, Vec *);

PETSC_INTERN PetscErrorCode MatLMVMGetVecsRead_Internal(Mat, PetscInt, ...);
PETSC_INTERN PetscErrorCode MatLMVMRestoreVecsRead_Internal(Mat, PetscInt, ...);
PETSC_INTERN PetscErrorCode MatLMVMGetRange(Mat, PetscInt *, PetscInt *);

#define MatLMVMGetVecsRead(B, idx, ...)     MatLMVMGetVecsRead_Internal(B, idx, __VA_ARGS__, LMBASIS_END)
#define MatLMVMRestoreVecsRead(B, idx, ...) MatLMVMRestoreVecsRead_Internal(B, idx, __VA_ARGS__, LMBASIS_END)

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0HermitianTranspose(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvHermitianTranspose(Mat, Vec, Vec);

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedBasis(Mat, MatLMVMBasisType, LMBasis *);
PETSC_INTERN PetscErrorCode MatLMVMBasisGetWorkRow(Mat, MatLMVMBasisType, PetscScalar **);
PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreWorkRow(Mat, MatLMVMBasisType, PetscScalar **);
PETSC_INTERN PetscErrorCode MatLMVMBasisGetWorkVec(Mat, MatLMVMBasisType, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreWorkVec(Mat, MatLMVMBasisType, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMBasisMultHermitianTranspose(Mat, MatLMVMBasisType, PetscInt, PetscInt, Vec, Vec, PetscScalar[]);
PETSC_INTERN PetscErrorCode MatLMVMBasisMultAdd(Mat, MatLMVMBasisType, PetscInt, PetscInt, PetscScalar[], Vec);
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMVH(Mat, MatLMVMBasisType, PetscInt, PetscInt, PetscScalar, Vec, Vec, PetscScalar, PetscScalar[]);
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMV(Mat, MatLMVMBasisType, PetscInt, PetscInt, PetscScalar, PetscScalar[], PetscScalar, Vec);

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedGramian(Mat, MatLMVMBasisType, MatLMVMBasisType, LMBlockType block_type, LMGramian *);
PETSC_INTERN PetscErrorCode MatLMVMGramianUpdate(Mat, MatLMVMBasisType, MatLMVMBasisType, LMBlockType block_type);
PETSC_INTERN PetscErrorCode MatLMVMGramianSolve(Mat, PetscInt, PetscInt, MatLMVMBasisType, MatLMVMBasisType, LMSolveType, PetscScalar *, PetscBool);
PETSC_INTERN PetscErrorCode MatLMVMGramianInsertDiagonalValue(Mat, MatLMVMBasisType, MatLMVMBasisType, PetscInt, PetscScalar);
PETSC_INTERN PetscErrorCode MatLMVMGramianGetDiagonalValue(Mat, MatLMVMBasisType, MatLMVMBasisType, PetscInt, PetscScalar *);
#endif
