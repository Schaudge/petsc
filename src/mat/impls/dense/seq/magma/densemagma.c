/*
     Defines the matrix operations for sequential dense with Magma
*/
#include <petscpkg_version.h>
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsccublas.h>

/* cublas definitions are here */
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#include <magma.h>

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool   user_alloc;
  PetscScalar *unplacedarray; /* if one called MatCUDADensePlaceArray(), this is where it stashed the original */
  PetscBool   unplaced_user_alloc;
  /* factorization support */
  int         *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_work; /* device workspace */
  int         fact_lwork;
  int         *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseMagma;

PETSC_EXTERN PetscErrorCode MatSeqDenseMagmaInvertFactors_Private(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseMagma(Mat A,Mat B,Mat X)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseMagma_Private(Mat A,Vec xx,Vec yy,PetscBool trans)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseMagma(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseMagma_Private(A,xx,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseMagma(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseMagma_Private(A,xx,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseMagma(Mat A,IS rperm,IS cperm,const MatFactorInfo *factinfo)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  A->ops->solve          = MatSolve_SeqDenseMagma;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseMagma;
  A->ops->matsolve       = MatMatSolve_SeqDenseMagma;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMAGMA,&A->solvertype);CHKERRQ(ierr);
  A->factortype = MAT_FACTOR_LU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseMagma(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseMagma *dA = (Mat_SeqDenseMagma*)A->spptr;
  PetscScalar       *da;
  int               n,lda;
  cudaError_t       ccer;
  magma_int_t       merr,info;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"Cholesky factor %d x %d on Magma backend\n",n,n);CHKERRQ(ierr);
  if (A->spd) {
    ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
    merr = magma_dpotrf_gpu(MagmaUpper,n,da,lda,&info);CHKERRMAGMA(merr);

    //    if (!dA->fact_lwork) {
    //    cerr = cusolverDnXpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
    //    ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
    // }
    // if (!dA->d_fact_info) {
    //    ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
    //  }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    //  cerr = cusolverDnXpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    ccer = WaitForGPU();CHKERRCUDA(ccer);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
    //#if defined(PETSC_USE_DEBUG)
                                             //ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
    //    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    //    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
    //#endif
  }
  A->ops->solve          = MatSolve_SeqDenseMagma;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseMagma;
  A->ops->matsolve       = MatMatSolve_SeqDenseMagma;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMAGMA,&A->solvertype);CHKERRQ(ierr);
  A->factortype = MAT_FACTOR_CHOLESKY;
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma_Private(Mat A,Mat B,Mat C,PetscBool tA,PetscBool tB)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseMagma_SeqDenseMagma(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma_Private(A,B,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma_Private(A,B,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseMagma_SeqDenseMagma(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma_Private(A,B,C,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqDenseMagma_Private(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (yy && yy != zz) { /* mult add */
    ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) {
    if (!yy) { /* mult only */
      ierr = VecSet_SeqCUDA(zz,0.0);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo2(A,"Matrix-vector product %d x %d on backend\n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseMagma(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseMagma_Private(A,xx,yy,zz,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseMagma(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseMagma_Private(A,xx,yy,zz,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseMagma(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseMagma_Private(A,xx,NULL,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseMagma(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseMagma_Private(A,xx,NULL,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseMagma(Mat A)
{
  Mat_SeqDenseMagma *dA = (Mat_SeqDenseMagma*)A->spptr;
  cudaError_t      cerr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dA) {
    if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseMagmaResetArray() must be called first");
    if (!dA->user_alloc) { cerr = cudaFree(dA->d_v);CHKERRCUDA(cerr); }
    ierr = VecDestroy(&dA->workvec);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDenseMagma_SeqDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat              B;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatConvert_SeqDenseCUDA_SeqDense(M,type,reuse,newmat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseMagma(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) { A->offloadmask = PETSC_OFFLOAD_CPU; }
  ierr = MatConvert_SeqDenseMagma_SeqDense(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatDestroy_SeqDense(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/vecimpl.h>

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_magma(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,MATSEQDENSEMAGMA);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERMAGMA,&(*fact)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatBindToCPU_SeqDenseCUDA(Mat,PetscBool);

static PetscErrorCode MatBindToCPU_SeqDenseMagma(Mat A,PetscBool flg)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatBindToCPU_SeqDenseCUDA(A,flg);CHKERRQ(ierr);
  if (!flg) {
    PetscBool iscuda;
    A->ops->mult                    = MatMult_SeqDenseMagma;
    A->ops->multadd                 = MatMultAdd_SeqDenseMagma;
    A->ops->multtranspose           = MatMultTranspose_SeqDenseMagma;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDenseMagma;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDenseMagma_SeqDenseMagma;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDenseMagma_SeqDenseMagma;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDenseMagma_SeqDenseMagma;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDenseMagma;
    A->ops->lufactor                = MatLUFactor_SeqDenseMagma;
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat,MatType,MatReuse,Mat *);

PetscErrorCode MatConvert_SeqDense_SeqDenseMagma(Mat M,MatType type,MatReuse reuse,Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatConvert_SeqDense_SeqDenseCUDA(M,type,reuse,B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)*B,"MatConvert_seqdensemagma_seqdense_C",MatConvert_SeqDenseMagma_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*B,MATSEQDENSEMAGMA);CHKERRQ(ierr);
  ierr = MatBindToCPU_SeqDenseMagma(*B,PETSC_FALSE);CHKERRQ(ierr);
  (*B)->ops->bindtocpu = MatBindToCPU_SeqDenseMagma;
  (*B)->ops->destroy  = MatDestroy_SeqDenseMagma;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqDenseMagma - Creates a sequential matrix in dense format using Magma.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of rows
.  n - number of columns
-  data - optional location of GPU matrix data.  Set data=NULL for PETSc
   to control matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqDense(), MatCreateSeqDenseCUDA()
@*/
PetscErrorCode  MatCreateSeqDenseMagma(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Invalid communicator size %d",size);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSEMAGMA);CHKERRQ(ierr);
  ierr = MatSeqDenseCUDASetPreallocation(*A,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSEMAGMA - MATSEQDENSEMAGMA = "seqdensemagma" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensemagma - sets the matrix type to "seqdensemagma" during a call to MatSetFromOptions()

  Level: beginner

.seealso:   MATSEQDENSE, MATSEQDENSECUDA
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseMagma(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqDense(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqDense_SeqDenseMagma(B,MATSEQDENSEMAGMA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
