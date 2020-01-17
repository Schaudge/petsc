/*
  Defines basic operations for the MATSEQAIJZ matrix class, providing select operations that are good for streaming of
  compressed matrix data.
*/

#include <../src/mat/impls/aij/seq/aij.h>

typedef struct {
  PetscInt  *colindices;
} Mat_SeqAIJZ;

static PetscErrorCode MatDestroy_SeqAIJZ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJZ    *aijz = (Mat_SeqAIJZ*) A->spptr;

  PetscFunctionBegin;

  /* If MatHeaderMerge() was used, then this SeqAIJZ matrix will not have an spptr pointer. */
  if (aijz) {
    /* Clean up everything in the Mat_SeqAIJZ data structure, then free A->spptr. */
    ierr = PetscFree(aijz->colindices);CHKERRQ(ierr);
    ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  }

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ()
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that
   * is only to be called when *building* a matrix.  I could be wrong, but
   * that is how things work for the SuperLU matrix class. */
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Build or update the shadow matrix */
static PetscErrorCode MatSeqAIJZ_update(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *aij  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJZ    *aijz = (Mat_SeqAIJZ*)A->spptr;

  PetscFunctionBegin;
  ierr = PetscFree(aijz->colindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(aij->nz,&aijz->colindices);CHKERRQ(ierr);
  ierr = PetscArraycpy(aijz->colindices,aij->j,aij->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqAIJZ(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode ierr;
  Mat_SeqAIJZ *aijz;
  Mat_SeqAIJZ *aijz_dest;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,op,M);CHKERRQ(ierr);
  aijz      = (Mat_SeqAIJZ*) A->spptr;
  aijz_dest = (Mat_SeqAIJZ*) (*M)->spptr;
  ierr = PetscArraycpy(aijz_dest,aijz,1);CHKERRQ(ierr);
  ierr = MatSeqAIJZ_update(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJZ(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Since a MATSEQAIJZ matrix is really just a MATSEQAIJ with some
   * extra information and some different methods, call the AssemblyEnd
   * routine for a MATSEQAIJ. */
  ierr = MatAssemblyEnd_SeqAIJ(A, mode);CHKERRQ(ierr);

  ierr = MatSeqAIJZ_update(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqAIJZ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *aij          = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJZ       *aijz         = (Mat_SeqAIJZ*)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aa = aij->a;
  PetscErrorCode    ierr;
  PetscInt          m = A->rmap->n,i,j;
  const PetscInt    *ai = aij->i,*aj = aijz->colindices;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar sum = 0;
    for (j=ai[i]; j<ai[i+1]; j++) {
      sum += aa[j] * x[aj[j]];
    }
    y[i] = sum;
  }
  ierr = PetscLogFlops(2.0*aij->nz - aij->nonzerorowcnt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJZ converts a SeqAIJ matrix into a
 * SeqAIJZ matrix.  This routine is called by the MatCreate_SeqAIJZ()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJZ one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJZ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqAIJ     *b;
  Mat_SeqAIJZ *aijz;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)A,type,&sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr     = PetscNewLog(B,&aijz);CHKERRQ(ierr);
  b        = (Mat_SeqAIJ*)B->data;
  B->spptr = (void*)aijz;

  /* Disable use of the inode routines so that the AIJZ ones will be used instead. */
  b->inode.use = PETSC_FALSE;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate        = MatDuplicate_SeqAIJZ;
  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJZ;
  B->ops->destroy          = MatDestroy_SeqAIJZ;

  /* If A has already been assembled and eager shadowing is specified, build the shadow matrix. */
  if (A->assembled) {
    ierr = MatSeqAIJZ_update(A);CHKERRQ(ierr);
  }

  B->ops->mult             = MatMult_SeqAIJZ;

  ierr    = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJZ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJZ - MATSEQAIJZ = "seqaijz" - A sequential sparse matrix type that uses streaming of compressed matrix
   entries for select operations and falls back to MATSEQAIJ for other operations.

   Options Database Keys:
+ -mat_type seqaijz - sets the matrix type to "seqaijz" during a call to MatSetFromOptions()
- -dm_mat_type seqaijz - makes DMCreateMatrix() produce matrices of type "seqaijz"

   Level: intermediate

.seealso: MatCreateSeqAIJ(), MatSetFromOptions(), MatSetType(), MatCreate(), MatType
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJZ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJZ(A,MATSEQAIJZ,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
