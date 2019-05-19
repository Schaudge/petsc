
#if !defined(__FAIJ_H)
#define __FAIJ_H
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/aij/seq/aij.h>

/* This header is shared by the SeqSFAIJ matrix */
#define SEQFAIJHEADER \
  PetscInt    bs2;                      /*  square of block size */                                  \
  PetscInt    mbs,nbs;               /* rows/bs, columns/bs */                                       \
  PetscScalar *mult_work;            /* work array for matrix vector product*/                       \
  PetscScalar *sor_workt;            /* work array for SOR */                                        \
  PetscScalar *sor_work;             /* work array for SOR */                                        \
  MatScalar   *saved_values;                                                                    \
  MatScalar   *idiag;            /* inverse of block diagonal  */                                \
  PetscBool   idiagvalid         /* if above has correct/current values */

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQFAIJHEADER;
} Mat_SeqFAIJ;

PETSC_INTERN PetscErrorCode MatSeqFAIJSetPreallocation_SeqFAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqFAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultHermitianTranspose_SeqFAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqFAIJ(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeAdd_SeqFAIJ(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatScale_SeqFAIJ(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatNorm_SeqFAIJ(Mat,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode MatEqual_SeqFAIJ(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqFAIJ(Mat,Vec);
PETSC_INTERN PetscErrorCode MatGetInfo_SeqFAIJ(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatZeroEntries_SeqFAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqFAIJ(Mat);
PETSC_INTERN PetscErrorCode MatAssemblyEnd_SeqFAIJ(Mat,MatAssemblyType);
PETSC_INTERN PetscErrorCode MatMult_SeqFAIJ_N(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqFAIJ_N(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSetValues_SeqFAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
PETSC_INTERN PetscErrorCode MatSetValuesBlocked_SeqFAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

#endif
