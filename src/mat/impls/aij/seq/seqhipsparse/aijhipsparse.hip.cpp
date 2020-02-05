/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the HIPSPARSE library,
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>

const char *const MatHIPSPARSEStorageFormats[] = {"CSR","ELL","HYB","MatHIPSPARSEStorageFormat","MAT_HIPSPARSE_",0};

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatSolve_SeqAIJHIHIPPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat);
static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec);

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix**);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct**);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct**,MatHIPSPARSEStorageFormat);
static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors**);
static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE**);

PetscErrorCode MatHIPSPARSESetStream(Mat A,const hipStream_t stream)
{
  hipsparseStatus_t   stat;
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  hipsparsestruct->stream = stream;
  stat = hipsparseSetStream(hipsparsestruct->handle,hipsparsestruct->stream);CHKERRHIPSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSESetHandle(Mat A,const hipsparseHandle_t handle)
{
  hipsparseStatus_t   stat;
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (hipsparsestruct->handle != handle) {
    if (hipsparsestruct->handle) {
      stat = hipsparseDestroy(hipsparsestruct->handle);CHKERRHIPSPARSE(stat);
    }
    hipsparsestruct->handle = handle;
  }
  stat = hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSEClearHandle(Mat A)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  PetscFunctionBegin;
  if (hipsparsestruct->handle)
    hipsparsestruct->handle = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_hipsparse(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHIPSPARSE;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERHIPSPARSE = "hipsparse" - A matrix type providing triangular solvers for seq matrices
  on a single GPU of type, seqaijhipsparse, aijhipsparse, or seqaijhipsp, aijhipsp. Currently supported
  algorithms are ILU(k) and ICC(k). Typically, deeper factorizations (larger k) results in poorer
  performance in the triangular solves. Full LU, and Cholesky decompositions can be solved through the
  HIPSPARSE triangular solve algorithm. However, the performance can be quite poor and thus these
  algorithms are not recommended. This class does NOT support direct solver operations.

  Level: beginner

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatCreateSeqAIJHIPSPARSE(), MATAIJHIPSPARSE, MatCreateAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJHIPSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJHIPSPARSE;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJHIPSPARSE;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for HIPSPARSE Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_hipsparse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatHIPSPARSESetFormat_SeqAIJHIPSPARSE(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT:
    hipsparsestruct->format = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparsestruct->format = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatHIPSPARSEFormatOperation. MAT_HIPSPARSE_MULT and MAT_HIPSPARSE_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

/*@
   MatHIPSPARSESetFormat - Sets the storage format of HIPSPARSE matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for MPIAIJHIPSPARSE matrices.
   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJHIPSPARSE
.  op - MatHIPSPARSEFormatOperation. SEQAIJHIPSPARSE matrices support MAT_HIPSPARSE_MULT and MAT_HIPSPARSE_ALL. MPIAIJHIPSPARSE matrices support MAT_HIPSPARSE_MULT_DIAG, MAT_HIPSPARSE_MULT_OFFDIAG, and MAT_HIPSPARSE_ALL.
-  format - MatHIPSPARSEStorageFormat (one of MAT_HIPSPARSE_CSR, MAT_HIPSPARSE_ELL, MAT_HIPSPARSE_HYB. The latter two require CUDA 4.2)

   Output Parameter:

   Level: intermediate

.seealso: MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
@*/
PetscErrorCode MatHIPSPARSESetFormat(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  ierr = PetscTryMethod(A, "MatHIPSPARSESetFormat_C",(Mat,MatHIPSPARSEFormatOperation,MatHIPSPARSEStorageFormat),(A,op,format));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;
  MatHIPSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_SeqAIJHIPSPARSE       *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJHIPSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_hipsparse_mult_storage_format","sets storage format of (seq)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsEnum("-mat_hipsparse_storage_format","sets storage format of (seq)aijhipsparse gpu matrices for SpMV and TriSolve",
                          "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_ALL,format);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatICCFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  hipsparseStatus_t                  stat;
  const PetscInt                    *ai = a->i,*aj = a->j,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiLo, *AjLo;
  PetscScalar                       *AALo;
  PetscInt                          i,nz, nzLower, offset, rowOffset;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];

      /* Allocate Space for the lower triangular matrix */
      cerr = hipHostMalloc((void**) &AiLo, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AjLo, nzLower*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AALo, nzLower*sizeof(PetscScalar));CHKERRCUDA(cerr);

      /* Fill the lower triangular matrix */
      AiLo[0]  = (PetscInt) 0;
      AiLo[n]  = nzLower;
      AjLo[0]  = (PetscInt) 0;
      AALo[0]  = (MatScalar) 1.0;
      v        = aa;
      vi       = aj;
      offset   = 1;
      rowOffset= 1;
      for (i=1; i<n; i++) {
        nz = ai[i+1] - ai[i];
        /* additional 1 for the term on the diagonal */
        AiLo[i]    = rowOffset;
        rowOffset += nz+1;

        ierr = PetscArraycpy(&(AjLo[offset]), vi, nz);CHKERRQ(ierr);
        ierr = PetscArraycpy(&(AALo[offset]), v, nz);CHKERRQ(ierr);

        offset      += nz;
        AjLo[offset] = (PetscInt) i;
        AALo[offset] = (MatScalar) 1.0;
        offset      += 1;

        v  += nz;
        vi += nz;
      }

      /* allocate space for the triangular factor information */
      loTriFactor = new Mat_SeqAIJHIPSPARSETriFactorStruct;

      /* Create the matrix description */
      stat = hipsparseCreateMatDescr(&loTriFactor->descr);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatIndexBase(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatType(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatFillMode(loTriFactor->descr, HIPSPARSE_FILL_MODE_LOWER);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatDiagType(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT);CHKERRHIPSPARSE(stat);

      /* Create the solve analysis information */
      stat = hipsparseCreateSolveAnalysisInfo(&loTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* set the operation */
      loTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

      /* set the matrix */
      loTriFactor->csrMat = new CsrMatrix;
      loTriFactor->csrMat->num_rows = n;
      loTriFactor->csrMat->num_cols = n;
      loTriFactor->csrMat->num_entries = nzLower;

      loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
      loTriFactor->csrMat->row_offsets->assign(AiLo, AiLo+n+1);

      loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzLower);
      loTriFactor->csrMat->column_indices->assign(AjLo, AjLo+nzLower);

      loTriFactor->csrMat->values = new THRUSTARRAY(nzLower);
      loTriFactor->csrMat->values->assign(AALo, AALo+nzLower);

      /* perform the solve analysis */
      stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp,
                               loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                               loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                               loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo);CHKERRhipSPARSE(stat);

      /* assign the pointer. Is this really necessary? */
      ((Mat_SeqAIJhipSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;

      cerr = hipHostFree(AiLo);CHKERRCUDA(cerr);
      cerr = hipHostFree(AjLo);CHKERRCUDA(cerr);
      cerr = hipHostFree(AALo);CHKERRCUDA(cerr);
      ierr = PetscLogCpuToGpu((n+1+nzLower)*sizeof(int)+nzLower*sizeof(PetscScalar));CHKERRQ(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  hipsparseStatus_t                  stat;
  const PetscInt                    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiUp, *AjUp;
  PetscScalar                       *AAUp;
  PetscInt                          i,nz, nzUpper, offset;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];

      /* Allocate Space for the upper triangular matrix */
      cerr = hipHostMalloc((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);

      /* Fill the upper triangular matrix */
      AiUp[0]=(PetscInt) 0;
      AiUp[n]=nzUpper;
      offset = nzUpper;
      for (i=n-1; i>=0; i--) {
        v  = aa + adiag[i+1] + 1;
        vi = aj + adiag[i+1] + 1;

        /* number of elements NOT on the diagonal */
        nz = adiag[i] - adiag[i+1]-1;

        /* decrement the offset */
        offset -= (nz+1);

        /* first, set the diagonal elements */
        AjUp[offset] = (PetscInt) i;
        AAUp[offset] = (MatScalar)1./v[nz];
        AiUp[i]      = AiUp[i+1] - (nz+1);

        ierr = PetscArraycpy(&(AjUp[offset+1]), vi, nz);CHKERRQ(ierr);
        ierr = PetscArraycpy(&(AAUp[offset+1]), v, nz);CHKERRQ(ierr);
      }

      /* allocate space for the triangular factor information */
      upTriFactor = new Mat_SeqAIJHIPSPARSETriFactorStruct;

      /* Create the matrix description */
      stat = hipsparseCreateMatDescr(&upTriFactor->descr);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatIndexBase(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatType(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatFillMode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatDiagType(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT);CHKERRHIPSPARSE(stat);

      /* Create the solve analysis information */
      stat = hipsparseCreateSolveAnalysisInfo(&upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* set the operation */
      upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

      /* set the matrix */
      upTriFactor->csrMat = new CsrMatrix;
      upTriFactor->csrMat->num_rows = n;
      upTriFactor->csrMat->num_cols = n;
      upTriFactor->csrMat->num_entries = nzUpper;

      upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
      upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+n+1);

      upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzUpper);
      upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+nzUpper);

      upTriFactor->csrMat->values = new THRUSTARRAY(nzUpper);
      upTriFactor->csrMat->values->assign(AAUp, AAUp+nzUpper);

      /* perform the solve analysis */
      stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp,
                               upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                               upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                               upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* assign the pointer. Is this really necessary? */
      ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;

      cerr = hipHostFree(AiUp);CHKERRCUDA(cerr);
      cerr = hipHostFree(AjUp);CHKERRCUDA(cerr);
      cerr = hipHostFree(AAUp);CHKERRCUDA(cerr);
      ierr = PetscLogCpuToGpu((n+1+nzUpper)*sizeof(int)+nzUpper*sizeof(PetscScalar));CHKERRQ(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  IS                           isrow = a->row,iscol = a->icol;
  PetscBool                    row_identity,col_identity;
  const PetscInt               *r,*c;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(A);CHKERRQ(ierr);

  /* TODO: SEK - need to fix THRUST references */
  hipsparseTriFactors->workVector = new THRUSTARRAY(n);
  hipsparseTriFactors->nnz=a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH;
  /* lower triangular indices */
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity) {
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(r, r+n);
  }
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  /* upper triangular indices */
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity) {
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(c, c+n);
  }

  if (!row_identity && !col_identity) {
    ierr = PetscLogCpuToGpu(2*n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if(!row_identity) {
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if(!col_identity) {
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  hipsparseStatus_t                  stat;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;
  PetscInt                          *AiUp, *AjUp;
  PetscScalar                       *AAUp;
  PetscScalar                       *AALo;
  PetscInt                          nzUpper = a->nz,n = A->rmap->n,i,offset,nz,j;
  Mat_SeqSBAIJ                      *b = (Mat_SeqSBAIJ*)A->data;
  const PetscInt                    *ai = b->i,*aj = b->j,*vj;
  const MatScalar                   *aa = b->a,*v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* Allocate Space for the upper triangular matrix */
      cerr = hipHostMalloc((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);
      cerr = hipHostMalloc((void**) &AALo, nzUpper*sizeof(PetscScalar));CHKERRCUDA(cerr);

      /* Fill the upper triangular matrix */
      AiUp[0]=(PetscInt) 0;
      AiUp[n]=nzUpper;
      offset = 0;
      for (i=0; i<n; i++) {
        /* set the pointers */
        v  = aa + ai[i];
        vj = aj + ai[i];
        nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

        /* first, set the diagonal elements */
        AjUp[offset] = (PetscInt) i;
        AAUp[offset] = (MatScalar)1.0/v[nz];
        AiUp[i]      = offset;
        AALo[offset] = (MatScalar)1.0/v[nz];

        offset+=1;
        if (nz>0) {
          ierr = PetscArraycpy(&(AjUp[offset]), vj, nz);CHKERRQ(ierr);
          ierr = PetscArraycpy(&(AAUp[offset]), v, nz);CHKERRQ(ierr);
          for (j=offset; j<offset+nz; j++) {
            AAUp[j] = -AAUp[j];
            AALo[j] = AAUp[j]/v[nz];
          }
          offset+=nz;
        }
      }

      /* allocate space for the triangular factor information */
      upTriFactor = new Mat_SeqAIJHIPSPARSETriFactorStruct;

      /* Create the matrix description */
      stat = hipsparseCreateMatDescr(&upTriFactor->descr);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatIndexBase(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatType(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatFillMode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatDiagType(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT);CHKERRHIPSPARSE(stat);

      /* Create the solve analysis information */
      stat = hipsparseCreateSolveAnalysisInfo(&upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* set the operation */
      upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

      /* set the matrix */
      upTriFactor->csrMat = new CsrMatrix;
      upTriFactor->csrMat->num_rows = A->rmap->n;
      upTriFactor->csrMat->num_cols = A->cmap->n;
      upTriFactor->csrMat->num_entries = a->nz;

      upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

      upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
      upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

      upTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
      upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);

      /* perform the solve analysis */
      stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp,
                               upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                               upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                               upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* assign the pointer. Is this really necessary? */
      ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;

      /* allocate space for the triangular factor information */
      loTriFactor = new Mat_SeqAIJHIPSPARSETriFactorStruct;

      /* Create the matrix description */
      stat = hipsparseCreateMatDescr(&loTriFactor->descr);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatIndexBase(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatType(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatFillMode(loTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
      stat = hipsparseSetMatDiagType(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT);CHKERRHIPSPARSE(stat);

      /* Create the solve analysis information */
      stat = hipsparseCreateSolveAnalysisInfo(&loTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* set the operation */
      loTriFactor->solveOp = HIPSPARSE_OPERATION_TRANSPOSE;

      /* set the matrix */
      loTriFactor->csrMat = new CsrMatrix;
      loTriFactor->csrMat->num_rows = A->rmap->n;
      loTriFactor->csrMat->num_cols = A->cmap->n;
      loTriFactor->csrMat->num_entries = a->nz;

      loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      loTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

      loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
      loTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

      loTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
      loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);
      ierr = PetscLogCpuToGpu(2*(((A->rmap->n+1)+(a->nz))*sizeof(int)+(a->nz)*sizeof(PetscScalar)));CHKERRQ(ierr);

      /* perform the solve analysis */
      stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp,
                               loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                               loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                               loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo);CHKERRHIPSPARSE(stat);

      /* assign the pointer. Is this really necessary? */
      ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;

      A->offloadmask = PETSC_OFFLOAD_BOTH;
      cerr = hipHostFree(AiUp);CHKERRCUDA(cerr);
      cerr = hipHostFree(AjUp);CHKERRCUDA(cerr);
      cerr = hipHostFree(AAUp);CHKERRCUDA(cerr);
      cerr = hipHostFree(AALo);CHKERRCUDA(cerr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  IS                           ip = a->row;
  const PetscInt               *rip;
  PetscBool                    perm_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSEBuildICCTriMatrices(A);CHKERRQ(ierr);
  hipsparseTriFactors->workVector = new THRUSTARRAY(n);
  hipsparseTriFactors->nnz=(a->nz-n)*2 + n;

  /* lower triangular indices */
  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) {
    IS             iip;
    const PetscInt *irip;

    ierr = ISInvertPermutation(ip,PETSC_DECIDE,&iip);CHKERRQ(ierr);
    ierr = ISGetIndices(iip,&irip);CHKERRQ(ierr);
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(rip, rip+n);
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(irip, irip+n);
    ierr = ISRestoreIndices(iip,&irip);CHKERRQ(ierr);
    ierr = ISDestroy(&iip);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(2*n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row,iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);

  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity) {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  hipsparseStatus_t                  stat;
  hipsparseIndexBase_t               indexBase;
  hipsparseMatrixType_t              matrixType;
  hipsparseFillMode_t                fillMode;
  hipsparseDiagType_t                diagType;

  PetscFunctionBegin;

  /*********************************************/
  /* Now the Transpose of the Lower Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the lower triangular factor */
  loTriFactorT = new Mat_SeqAIJHIPSPARSETriFactorStruct;

  /* set the matrix descriptors of the lower triangular factor */
  matrixType = hipsparseGetMatType(loTriFactor->descr);
  indexBase = hipsparseGetMatIndexBase(loTriFactor->descr);
  fillMode = hipsparseGetMatFillMode(loTriFactor->descr)==HIPSPARSE_FILL_MODE_UPPER ?
    HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType = hipsparseGetMatDiagType(loTriFactor->descr);

  /* Create the matrix description */
  stat = hipsparseCreateMatDescr(&loTriFactorT->descr);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatIndexBase(loTriFactorT->descr, indexBase);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatType(loTriFactorT->descr, matrixType);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatFillMode(loTriFactorT->descr, fillMode);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatDiagType(loTriFactorT->descr, diagType);CHKERRHIPSPARSE(stat);

  /* Create the solve analysis information */
  stat = hipsparseCreateSolveAnalysisInfo(&loTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);

  /* set the operation */
  loTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the lower triangular factor*/
  loTriFactorT->csrMat = new CsrMatrix;
  loTriFactorT->csrMat->num_rows = loTriFactor->csrMat->num_rows;
  loTriFactorT->csrMat->num_cols = loTriFactor->csrMat->num_cols;
  loTriFactorT->csrMat->num_entries = loTriFactor->csrMat->num_entries;
  loTriFactorT->csrMat->row_offsets = new THRUSTINTARRAY32(loTriFactor->csrMat->num_rows+1);
  loTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(loTriFactor->csrMat->num_entries);
  loTriFactorT->csrMat->values = new THRUSTARRAY(loTriFactor->csrMat->num_entries);

  /* compute the transpose of the lower triangular factor, i.e. the CSC */
  stat = hipsparse_csr2csc(hipsparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                          loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                          loTriFactor->csrMat->values->data().get(),
                          loTriFactor->csrMat->row_offsets->data().get(),
                          loTriFactor->csrMat->column_indices->data().get(),
                          loTriFactorT->csrMat->values->data().get(),
                          loTriFactorT->csrMat->column_indices->data().get(),
                          loTriFactorT->csrMat->row_offsets->data().get(),
                          HIPSPARSE_ACTION_NUMERIC, indexBase);CHKERRHIPSPARSE(stat);

  /* perform the solve analysis on the transposed matrix */
  stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                           loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries,
                           loTriFactorT->descr, loTriFactorT->csrMat->values->data().get(),
                           loTriFactorT->csrMat->row_offsets->data().get(), loTriFactorT->csrMat->column_indices->data().get(),
                           loTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);

  /* assign the pointer. Is this really necessary? */
  ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->loTriFactorPtrTranspose = loTriFactorT;

  /*********************************************/
  /* Now the Transpose of the Upper Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the upper triangular factor */
  upTriFactorT = new Mat_SeqAIJHIPSPARSETriFactorStruct;

  /* set the matrix descriptors of the upper triangular factor */
  matrixType = hipsparseGetMatType(upTriFactor->descr);
  indexBase = hipsparseGetMatIndexBase(upTriFactor->descr);
  fillMode = hipsparseGetMatFillMode(upTriFactor->descr)==HIPSPARSE_FILL_MODE_UPPER ?
    HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType = hipsparseGetMatDiagType(upTriFactor->descr);

  /* Create the matrix description */
  stat = hipsparseCreateMatDescr(&upTriFactorT->descr);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatIndexBase(upTriFactorT->descr, indexBase);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatType(upTriFactorT->descr, matrixType);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatFillMode(upTriFactorT->descr, fillMode);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatDiagType(upTriFactorT->descr, diagType);CHKERRHIPSPARSE(stat);

  /* Create the solve analysis information */
  stat = hipsparseCreateSolveAnalysisInfo(&upTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);

  /* set the operation */
  upTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the upper triangular factor*/
  upTriFactorT->csrMat = new CsrMatrix;
  upTriFactorT->csrMat->num_rows = upTriFactor->csrMat->num_rows;
  upTriFactorT->csrMat->num_cols = upTriFactor->csrMat->num_cols;
  upTriFactorT->csrMat->num_entries = upTriFactor->csrMat->num_entries;
  upTriFactorT->csrMat->row_offsets = new THRUSTINTARRAY32(upTriFactor->csrMat->num_rows+1);
  upTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(upTriFactor->csrMat->num_entries);
  upTriFactorT->csrMat->values = new THRUSTARRAY(upTriFactor->csrMat->num_entries);

  /* compute the transpose of the upper triangular factor, i.e. the CSC */
  stat = hipsparse_csr2csc(hipsparseTriFactors->handle, upTriFactor->csrMat->num_rows,
                          upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                          upTriFactor->csrMat->values->data().get(),
                          upTriFactor->csrMat->row_offsets->data().get(),
                          upTriFactor->csrMat->column_indices->data().get(),
                          upTriFactorT->csrMat->values->data().get(),
                          upTriFactorT->csrMat->column_indices->data().get(),
                          upTriFactorT->csrMat->row_offsets->data().get(),
                          HIPSPARSE_ACTION_NUMERIC, indexBase);CHKERRHIPSPARSE(stat);

  /* perform the solve analysis on the transposed matrix */
  stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                           upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries,
                           upTriFactorT->descr, upTriFactorT->csrMat->values->data().get(),
                           upTriFactorT->csrMat->row_offsets->data().get(), upTriFactorT->csrMat->column_indices->data().get(),
                           upTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);

  /* assign the pointer. Is this really necessary? */
  ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtrTranspose = upTriFactorT;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEGenerateTransposeForMult(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
  Mat_SeqAIJHIPSPARSEMultStruct *matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  hipsparseStatus_t             stat;
  hipsparseIndexBase_t          indexBase;
  hipError_t                  err;
  PetscErrorCode               ierr;

  PetscFunctionBegin;

  /* allocate space for the triangular factor information */
  matstructT = new Mat_SeqAIJHIPSPARSEMultStruct;
  stat = hipsparseCreateMatDescr(&matstructT->descr);CHKERRHIPSPARSE(stat);
  indexBase = hipsparseGetMatIndexBase(matstruct->descr);
  stat = hipsparseSetMatIndexBase(matstructT->descr, indexBase);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatType(matstructT->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);

  /* set alpha and beta */
  err = hipMalloc((void **)&(matstructT->alpha),    sizeof(PetscScalar));CHKERRCUDA(err);
  err = hipMalloc((void **)&(matstructT->beta_zero),sizeof(PetscScalar));CHKERRCUDA(err);
  err = hipMalloc((void **)&(matstructT->beta_one), sizeof(PetscScalar));CHKERRCUDA(err);
  err = hipMemcpy(matstructT->alpha,    &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
  err = hipMemcpy(matstructT->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
  err = hipMemcpy(matstructT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
  stat = hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);

  if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
    CsrMatrix *matrix = (CsrMatrix*)matstruct->mat;
    CsrMatrix *matrixT= new CsrMatrix;
    thrust::device_vector<int> *rowoffsets_gpu;
    const int ncomprow = matstruct->cprowIndices->size();
    thrust::host_vector<int> cprow(ncomprow);
    cprow = *matstruct->cprowIndices; // GPU --> CPU
    matrixT->num_rows = A->cmap->n;
    matrixT->num_cols = A->rmap->n;
    matrixT->num_entries = a->nz;
    matrixT->row_offsets = new THRUSTINTARRAY32(matrixT->num_rows+1);
    matrixT->column_indices = new THRUSTINTARRAY32(a->nz);
    matrixT->values = new THRUSTARRAY(a->nz);
    PetscInt k,i;
    thrust::host_vector<int> rowst_host(A->rmap->n+1);

    /* expand compress rows, which is forced in constructor (MatSeqAIJHIPSPARSECopyToGPU) */
    rowst_host[0] = 0;
    for (k = 0, i = 0; i < A->rmap->n ; i++) {
      if (k < ncomprow && i==cprow[k]) {
	rowst_host[i+1] = a->i[i+1];
	k++;
      } else rowst_host[i+1] = rowst_host[i];
    }
    if (k!=ncomprow) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"%D != %D",k,ncomprow);
    rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1);
    *rowoffsets_gpu = rowst_host; // CPU --> GPU

    /* compute the transpose of the upper triangular factor, i.e. the CSC */
    indexBase = hipsparseGetMatIndexBase(matstruct->descr);
    stat = hipsparse_csr2csc(hipsparsestruct->handle, A->rmap->n,
                            A->cmap->n, matrix->num_entries,
                            matrix->values->data().get(),
                            rowoffsets_gpu->data().get(),
                            matrix->column_indices->data().get(),
                            matrixT->values->data().get(),
                            matrixT->column_indices->data().get(),
                            matrixT->row_offsets->data().get(),
                            HIPSPARSE_ACTION_NUMERIC, indexBase);CHKERRHIPSPARSE(stat);

    /* assign the pointer */
    matstructT->mat = matrixT;
    ierr = PetscLogCpuToGpu(((A->rmap->n+1)+(a->nz))*sizeof(int)+(3+a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else if (hipsparsestruct->format==MAT_HIPSPARSE_ELL || hipsparsestruct->format==MAT_HIPSPARSE_HYB) {
    /* First convert HYB to CSR */
    CsrMatrix *temp= new CsrMatrix;
    temp->num_rows = A->rmap->n;
    temp->num_cols = A->cmap->n;
    temp->num_entries = a->nz;
    temp->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
    temp->column_indices = new THRUSTINTARRAY32(a->nz);
    temp->values = new THRUSTARRAY(a->nz);


    stat = hipsparse_hyb2csr(hipsparsestruct->handle,
                            matstruct->descr, (hipsparseHybMat_t)matstruct->mat,
                            temp->values->data().get(),
                            temp->row_offsets->data().get(),
                            temp->column_indices->data().get());CHKERRHIPSPARSE(stat);

    /* Next, convert CSR to CSC (i.e. the matrix transpose) */
    CsrMatrix *tempT= new CsrMatrix;
    tempT->num_rows = A->rmap->n;
    tempT->num_cols = A->cmap->n;
    tempT->num_entries = a->nz;
    tempT->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
    tempT->column_indices = new THRUSTINTARRAY32(a->nz);
    tempT->values = new THRUSTARRAY(a->nz);

    stat = hipsparse_csr2csc(hipsparsestruct->handle, temp->num_rows,
                            temp->num_cols, temp->num_entries,
                            temp->values->data().get(),
                            temp->row_offsets->data().get(),
                            temp->column_indices->data().get(),
                            tempT->values->data().get(),
                            tempT->column_indices->data().get(),
                            tempT->row_offsets->data().get(),
                            HIPSPARSE_ACTION_NUMERIC, indexBase);CHKERRHIPSPARSE(stat);

    /* Last, convert CSC to HYB */
    hipsparseHybMat_t hybMat;
    stat = hipsparseCreateHybMat(&hybMat);CHKERRHIPSPARSE(stat);
    hipsparseHybPartition_t partition = hipsparsestruct->format==MAT_HIPSPARSE_ELL ?
      HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
    stat = hipsparse_csr2hyb(hipsparsestruct->handle, A->rmap->n, A->cmap->n,
                            matstructT->descr, tempT->values->data().get(),
                            tempT->row_offsets->data().get(),
                            tempT->column_indices->data().get(),
                            hybMat, 0, partition);CHKERRHIPSPARSE(stat);

    /* assign the pointer */
    matstructT->mat = hybMat;
    ierr = PetscLogCpuToGpu((2*(((A->rmap->n+1)+(a->nz))*sizeof(int)+(a->nz)*sizeof(PetscScalar)))+3*sizeof(PetscScalar));CHKERRQ(ierr);

    /* delete temporaries */
    if (tempT) {
      if (tempT->values) delete (THRUSTARRAY*) tempT->values;
      if (tempT->column_indices) delete (THRUSTINTARRAY32*) tempT->column_indices;
      if (tempT->row_offsets) delete (THRUSTINTARRAY32*) tempT->row_offsets;
      delete (CsrMatrix*) tempT;
    }
    if (temp) {
      if (temp->values) delete (THRUSTARRAY*) temp->values;
      if (temp->column_indices) delete (THRUSTINTARRAY32*) temp->column_indices;
      if (temp->row_offsets) delete (THRUSTINTARRAY32*) temp->row_offsets;
      delete (CsrMatrix*) temp;
    }
  }
  /* assign the compressed row indices */
  matstructT->cprowIndices = new THRUSTINTARRAY;
  matstructT->cprowIndices->resize(A->cmap->n);
  thrust::sequence(matstructT->cprowIndices->begin(), matstructT->cprowIndices->end());
  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSE*)A->spptr)->matTranspose = matstructT;
  PetscFunctionReturn(0);
}

/* Why do we need to analyze the tranposed matrix again? Can't we just use op(A) = HIPSPARSE_OPERATION_TRANSPOSE in MatSolve_SeqAIJHIPSPARSE? */
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat A,Vec bb,Vec xx)
{
  PetscInt                              n = xx->map->n;
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  hipsparseStatus_t                      stat;
  Mat_SeqAIJHIPSPARSETriFactors          *hihipparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                        ierr;
  hipError_t                           cerr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU+n, hipsparseTriFactors->rpermIndices->end()),
               xGPU);

  /* First, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        xarray, tempGPU->data().get());CHKERRHIPSPARSE(stat);

  /* Then, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(), xarray);CHKERRHIPSPARSE(stat);

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::make_permutation_iterator(xGPU, hipsparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(xGPU+n, hipsparseTriFactors->cpermIndices->end()),
               tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  hipsparseStatus_t                  stat;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        barray, tempGPU->data().get());CHKERRHIPSPARSE(stat);

  /* Then, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(), xarray);CHKERRHIPSPARSE(stat);

  /* restore */
  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  hipsparseStatus_t                      stat;
  Mat_SeqAIJHIPSPARSETriFactors          *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                        ierr;
  hipError_t                           cerr;

  PetscFunctionBegin;

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->end()),
               tempGPU->begin());

  /* Next, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        tempGPU->data().get(), xarray);CHKERRHIPSPARSE(stat);

  /* Then, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,
                        xarray, tempGPU->data().get());CHKERRHIPSPARSE(stat);

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->end()),
               xGPU);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  hipsparseStatus_t                  stat;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        barray, tempGPU->data().get());CHKERRHIPSPARSE(stat);

  /* Next, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows, &PETSC_HIPSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,
                        tempGPU->data().get(), xarray);CHKERRHIPSPARSE(stat);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  PetscInt                     m = A->rmap->n,*ii,*ridx;
  PetscErrorCode               ierr;
  hipsparseStatus_t             stat;
  hipError_t                  err;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (A->assembled && A->nonzerostate == hipsparsestruct->nonzerostate && hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      CsrMatrix *matrix = (CsrMatrix*)matstruct->mat;
      /* copy values only */
      matrix->values->assign(a->a, a->a+a->nz);
      ierr = PetscLogCpuToGpu((a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      MatSeqAIJHIPSPARSEMultStruct_Destroy(&matstruct,hipsparsestruct->format);
      try {
        hipsparsestruct->nonzerorow=0;
        for (int j = 0; j<m; j++) hipsparsestruct->nonzerorow += ((a->i[j+1]-a->i[j])>0);

        if (a->compressedrow.use) {
          m    = a->compressedrow.nrows;
          ii   = a->compressedrow.i;
          ridx = a->compressedrow.rindex;
        } else {
          /* Forcing compressed row on the GPU */
          int k=0;
          ierr = PetscMalloc1(hipsparsestruct->nonzerorow+1, &ii);CHKERRQ(ierr);
          ierr = PetscMalloc1(hipsparsestruct->nonzerorow, &ridx);CHKERRQ(ierr);
          ii[0]=0;
          for (int j = 0; j<m; j++) {
            if ((a->i[j+1]-a->i[j])>0) {
              ii[k]  = a->i[j];
              ridx[k]= j;
              k++;
            }
          }
          ii[hipsparsestruct->nonzerorow] = a->nz;
          m = hipsparsestruct->nonzerorow;
        }

        /* allocate space for the triangular factor information */
        matstruct = new Mat_SeqAIJHIPSPARSEMultStruct;
        stat = hipsparseCreateMatDescr(&matstruct->descr);CHKERRHIPSPARSE(stat);
        stat = hipsparseSetMatIndexBase(matstruct->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = hipsparseSetMatType(matstruct->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);

        err = hipMalloc((void **)&(matstruct->alpha),    sizeof(PetscScalar));CHKERRCUDA(err);
        err = hipMalloc((void **)&(matstruct->beta_zero),sizeof(PetscScalar));CHKERRCUDA(err);
        err = hipMalloc((void **)&(matstruct->beta_one), sizeof(PetscScalar));CHKERRCUDA(err);
        err = hipMemcpy(matstruct->alpha,    &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
        err = hipMemcpy(matstruct->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
        err = hipMemcpy(matstruct->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRCUDA(err);
        stat = hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);

        /* Build a hybrid/ellpack matrix if this option is chosen for the storage */
        if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
          /* set the matrix */
          CsrMatrix *matrix= new CsrMatrix;
          matrix->num_rows = m;
          matrix->num_cols = A->cmap->n;
          matrix->num_entries = a->nz;
          matrix->row_offsets = new THRUSTINTARRAY32(m+1);
          matrix->row_offsets->assign(ii, ii + m+1);

          matrix->column_indices = new THRUSTINTARRAY32(a->nz);
          matrix->column_indices->assign(a->j, a->j+a->nz);

          matrix->values = new THRUSTARRAY(a->nz);
          matrix->values->assign(a->a, a->a+a->nz);

          /* assign the pointer */
          matstruct->mat = matrix;

        } else if (hipsparsestruct->format==MAT_HIPSPARSE_ELL || hipsparsestruct->format==MAT_HIPSPARSE_HYB) {
          CsrMatrix *matrix= new CsrMatrix;
          matrix->num_rows = m;
          matrix->num_cols = A->cmap->n;
          matrix->num_entries = a->nz;
          matrix->row_offsets = new THRUSTINTARRAY32(m+1);
          matrix->row_offsets->assign(ii, ii + m+1);

          matrix->column_indices = new THRUSTINTARRAY32(a->nz);
          matrix->column_indices->assign(a->j, a->j+a->nz);

          matrix->values = new THRUSTARRAY(a->nz);
          matrix->values->assign(a->a, a->a+a->nz);

          hipsparseHybMat_t hybMat;
          stat = hipsparseCreateHybMat(&hybMat);CHKERRHIPSPARSE(stat);
          hipsparseHybPartition_t partition = hipsparsestruct->format==MAT_HIPSPARSE_ELL ?  HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
          stat = hipsparse_csr2hyb(hipsparsestruct->handle, matrix->num_rows, matrix->num_cols,
              matstruct->descr, matrix->values->data().get(),
              matrix->row_offsets->data().get(),
              matrix->column_indices->data().get(),
              hybMat, 0, partition);CHKERRHIPSPARSE(stat);
          /* assign the pointer */
          matstruct->mat = hybMat;

          if (matrix) {
            if (matrix->values) delete (THRUSTARRAY*)matrix->values;
            if (matrix->column_indices) delete (THRUSTINTARRAY32*)matrix->column_indices;
            if (matrix->row_offsets) delete (THRUSTINTARRAY32*)matrix->row_offsets;
            delete (CsrMatrix*)matrix;
          }
        }

        /* assign the compressed row indices */
        matstruct->cprowIndices = new THRUSTINTARRAY(m);
        matstruct->cprowIndices->assign(ridx,ridx+m);
        ierr = PetscLogCpuToGpu(((m+1)+(a->nz))*sizeof(int)+m*sizeof(PetscInt)+(3+(a->nz))*sizeof(PetscScalar));CHKERRQ(ierr);

        /* assign the pointer */
        hipsparsestruct->mat = matstruct;

        if (!a->compressedrow.use) {
          ierr = PetscFree(ii);CHKERRQ(ierr);
          ierr = PetscFree(ridx);CHKERRQ(ierr);
        }
        hipsparsestruct->workVector = new THRUSTARRAY(m);
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
      }
      hipsparsestruct->nonzerostate = A->nonzerostate;
    }
    err  = WaitForGPU();CHKERRCUDA(err);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
    ierr = PetscLogEventEnd(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct VecCUDAPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqAIJHIPSPARSE(A,xx,NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstructT;
  const PetscScalar            *xarray;
  PetscScalar                  *yarray;
  PetscErrorCode               ierr;
  hipError_t                  cerr;
  hipsparseStatus_t             stat;

  PetscFunctionBegin;
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  if (!matstructT) {
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  }
  ierr = VecCUDAGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(yy,&yarray);CHKERRQ(ierr);
  if (yy->map->n) {
    PetscInt                     n = yy->map->n;
    hipError_t                  err;
    err = hipMemset(yarray,0,n*sizeof(PetscScalar));CHKERRCUDA(err); /* hack to fix numerical errors from reading output vector yy, apparently */
  }

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
    CsrMatrix *mat = (CsrMatrix*)matstructT->mat;
    stat = hipsparse_csr_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             mat->num_rows, mat->num_cols,
                             mat->num_entries, matstructT->alpha, matstructT->descr,
                             mat->values->data().get(), mat->row_offsets->data().get(),
                             mat->column_indices->data().get(), xarray, matstructT->beta_zero,
                             yarray);CHKERRHIPSPARSE(stat);
  } else {
    hipsparseHybMat_t hybMat = (hipsparseHybMat_t)matstructT->mat;
    stat = hipsparse_hyb_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             matstructT->alpha, matstructT->descr, hybMat,
                             xarray, matstructT->beta_zero,
                             yarray);CHKERRHIPSPARSE(stat);
  }
  ierr = VecCUDARestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&yarray);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*a->nz - hipsparsestruct->nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct;
  const PetscScalar            *xarray;
  PetscScalar                  *zarray,*dptr,*beta;
  PetscErrorCode               ierr;
  hipError_t                  cerr;
  hipsparseStatus_t             stat;
  PetscBool                    cmpr; /* if the matrix has been compressed (zero rows) */

  PetscFunctionBegin;
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
  try {
    cmpr = (PetscBool)(hipsparsestruct->workVector->size() == (thrust::detail::vector_base<PetscScalar, thrust::device_malloc_allocator<PetscScalar> >::size_type)(A->rmap->n));
    ierr = VecCUDAGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    if (yy && !cmpr) { /* MatMultAdd with noncompressed storage -> need uptodate zz vector */
      ierr = VecCUDAGetArray(zz,&zarray);CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArrayWrite(zz,&zarray);CHKERRQ(ierr);
    }
    dptr = cmpr ? zarray : hipsparsestruct->workVector->data().get();
    beta = (yy == zz && dptr == zarray) ? matstruct->beta_one : matstruct->beta_zero;

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    /* csr_spmv is multiply add */
    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      /* here we need to be careful to set the number of rows in the multiply to the
         number of compressed rows in the matrix ... which is equivalent to the
         size of the workVector */
      CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
      stat = hipsparse_csr_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               mat->num_rows, mat->num_cols,
                               mat->num_entries, matstruct->alpha, matstruct->descr,
                               mat->values->data().get(), mat->row_offsets->data().get(),
                               mat->column_indices->data().get(), xarray, beta,
                               dptr);CHKERRHIPSPARSE(stat);
    } else {
      if (hipsparsestruct->workVector->size()) {
        hipsparseHybMat_t hybMat = (hipsparseHybMat_t)matstruct->mat;
        stat = hipsparse_hyb_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 matstruct->alpha, matstruct->descr, hybMat,
                                 xarray, beta,
                                 dptr);CHKERRHIPSPARSE(stat);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    if (yy) {
      if (dptr != zarray) {
        ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
      } else if (zz != yy) {
        ierr = VecAXPY_SeqCUDA(zz,1.0,yy);CHKERRQ(ierr);
      }
    } else if (dptr != zarray) {
      ierr = VecSet_SeqCUDA(zz,0);CHKERRQ(ierr);
    }
    /* scatter the data from the temporary into the full vector with a += operation */
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (dptr != zarray) {
      thrust::device_ptr<PetscScalar> zptr;

      zptr = thrust::device_pointer_cast(zarray);
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))),
                       thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))) + hipsparsestruct->workVector->size(),
                       VecCUDAPlusEquals());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    if (yy && !cmpr) {
      ierr = VecCUDARestoreArray(zz,&zarray);CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);
    }
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
  }
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ                      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSE              *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct    *matstructT;
  const PetscScalar               *xarray;
  PetscScalar                     *zarray,*dptr,*beta;
  PetscErrorCode                  ierr;
  hipError_t                     cerr;
  hipsparseStatus_t                stat;

  PetscFunctionBegin;
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  if (!matstructT) {
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  }

  try {
    ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(zz,&zarray);CHKERRQ(ierr);
    dptr = hipsparsestruct->workVector->size() == (thrust::detail::vector_base<PetscScalar, thrust::device_malloc_allocator<PetscScalar> >::size_type)(A->cmap->n) ? zarray : hipsparsestruct->workVector->data().get();
    beta = (yy == zz && dptr == zarray) ? matstructT->beta_one : matstructT->beta_zero;

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    /* multiply add with matrix transpose */
    if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
      CsrMatrix *mat = (CsrMatrix*)matstructT->mat;
      /* here we need to be careful to set the number of rows in the multiply to the
         number of compressed rows in the matrix ... which is equivalent to the
         size of the workVector */
      stat = hipsparse_csr_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               mat->num_rows, mat->num_cols,
                               mat->num_entries, matstructT->alpha, matstructT->descr,
                               mat->values->data().get(), mat->row_offsets->data().get(),
                               mat->column_indices->data().get(), xarray, beta,
                               dptr);CHKERRHIPSPARSE(stat);
    } else {
      hipsparseHybMat_t hybMat = (hipsparseHybMat_t)matstructT->mat;
      if (hipsparsestruct->workVector->size()) {
        stat = hipsparse_hyb_spmv(hipsparsestruct->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 matstructT->alpha, matstructT->descr, hybMat,
                                 xarray, beta,
                                 dptr);CHKERRHIPSPARSE(stat);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    if (dptr != zarray) {
      ierr = VecCopy_SeqCUDA(yy,zz);CHKERRQ(ierr);
    } else if (zz != yy) {
      ierr = VecAXPY_SeqCUDA(zz,1.0,yy);CHKERRQ(ierr);
    }
    /* scatter the data from the temporary into the full vector with a += operation */
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (dptr != zarray) {
      thrust::device_ptr<PetscScalar> zptr;

      zptr = thrust::device_pointer_cast(zarray);

      /* scatter the data from the temporary into the full vector with a += operation */
      thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstructT->cprowIndices->begin()))),
                       thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstructT->cprowIndices->begin()))) + A->cmap->n,
                       VecCUDAPlusEquals());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(zz,&zarray);CHKERRQ(ierr);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
  }
  cerr = WaitForGPU();CHKERRCUDA(cerr); /* is this needed? just for yy==0 in Mult */
  ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJHIPSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  }
  A->ops->mult             = MatMult_SeqAIJHIPSPARSE;
  A->ops->multadd          = MatMultAdd_SeqAIJHIPSPARSE;
  A->ops->multtranspose    = MatMultTranspose_SeqAIJHIPSPARSE;
  A->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@
   MatCreateSeqAIJHIPSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately pushed down
   to NVidia GPUs and use the HIPSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSEQAIJHIPSPARSE, MATAIJHIPSPARSE
@*/
PetscErrorCode  MatCreateSeqAIJHIPSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJHIPSPARSE(Mat A)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) {
      ierr = MatSeqAIJHIPSPARSE_Destroy((Mat_SeqAIJHIPSPARSE**)&A->spptr);CHKERRQ(ierr);
    }
  } else {
    ierr = MatSeqAIJHIPSPARSETriFactors_Destroy((Mat_SeqAIJHIPSPARSETriFactors**)&A->spptr);CHKERRQ(ierr);
  }
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqAIJHIPSPARSE(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;
  Mat C;
  hipsparseStatus_t stat;
  hipsparseHandle_t handle=0;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  C    = *B;
  ierr = PetscFree(C->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&C->defaultvectype);CHKERRQ(ierr);

  /* inject HIPSPARSE-specific stuff */
  if (C->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.
       now build a GPU matrix data structure */
    C->spptr = new Mat_SeqAIJHIPSPARSE;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->mat          = 0;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->matTranspose = 0;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->workVector   = 0;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->format       = MAT_HIPSPARSE_CSR;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->stream       = 0;
    stat = hipsparseCreate(&handle);CHKERRHIPSPARSE(stat);
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->handle       = handle;
    ((Mat_SeqAIJHIPSPARSE*)C->spptr)->nonzerostate = 0;
  } else {
    /* NEXT, set the pointers to the triangular factors */
    C->spptr = new Mat_SeqAIJHIPSPARSETriFactors;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->loTriFactorPtr          = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->upTriFactorPtr          = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->loTriFactorPtrTranspose = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->upTriFactorPtrTranspose = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->rpermIndices            = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->cpermIndices            = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->workVector              = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->handle                  = 0;
    stat = hipsparseCreate(&handle);CHKERRHIPSPARSE(stat);
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->handle                  = handle;
    ((Mat_SeqAIJHIPSPARSETriFactors*)C->spptr)->nnz                     = 0;
  }

  C->ops->assemblyend      = MatAssemblyEnd_SeqAIJHIPSPARSE;
  C->ops->destroy          = MatDestroy_SeqAIJHIPSPARSE;
  C->ops->setfromoptions   = MatSetFromOptions_SeqAIJHIPSPARSE;
  C->ops->mult             = MatMult_SeqAIJHIPSPARSE;
  C->ops->multadd          = MatMultAdd_SeqAIJHIPSPARSE;
  C->ops->multtranspose    = MatMultTranspose_SeqAIJHIPSPARSE;
  C->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJHIPSPARSE;
  C->ops->duplicate        = MatDuplicate_SeqAIJHIPSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)C,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);

  C->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  ierr = PetscObjectComposeFunction((PetscObject)C, "MatHIPSPARSESetFormat_C", MatHIPSPARSESetFormat_SeqAIJHIPSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat B)
{
  PetscErrorCode ierr;
  hipsparseStatus_t stat;
  hipsparseHandle_t handle=0;

  PetscFunctionBegin;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);

  if (B->factortype==MAT_FACTOR_NONE) {
    /* you cannot check the inode.use flag here since the matrix was just created.
       now build a GPU matrix data structure */
    B->spptr = new Mat_SeqAIJHIPSPARSE;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->mat          = 0;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->matTranspose = 0;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->workVector   = 0;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->format       = MAT_HIPSPARSE_CSR;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->stream       = 0;
    stat = hipsparseCreate(&handle);CHKERRHIPSPARSE(stat);
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->handle       = handle;
    ((Mat_SeqAIJHIPSPARSE*)B->spptr)->nonzerostate = 0;
  } else {
    /* NEXT, set the pointers to the triangular factors */
    B->spptr = new Mat_SeqAIJHIPSPARSETriFactors;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->loTriFactorPtr          = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->upTriFactorPtr          = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->loTriFactorPtrTranspose = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->upTriFactorPtrTranspose = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->rpermIndices            = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->cpermIndices            = 0;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->workVector              = 0;
    stat = hipsparseCreate(&handle);CHKERRHIPSPARSE(stat);
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->handle                  = handle;
    ((Mat_SeqAIJHIPSPARSETriFactors*)B->spptr)->nnz                     = 0;
  }

  B->ops->assemblyend      = MatAssemblyEnd_SeqAIJHIPSPARSE;
  B->ops->destroy          = MatDestroy_SeqAIJHIPSPARSE;
  B->ops->setfromoptions   = MatSetFromOptions_SeqAIJHIPSPARSE;
  B->ops->mult             = MatMult_SeqAIJHIPSPARSE;
  B->ops->multadd          = MatMultAdd_SeqAIJHIPSPARSE;
  B->ops->multtranspose    = MatMultTranspose_SeqAIJHIPSPARSE;
  B->ops->multtransposeadd = MatMultTransposeAdd_SeqAIJHIPSPARSE;
  B->ops->duplicate        = MatDuplicate_SeqAIJHIPSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);

  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  ierr = PetscObjectComposeFunction((PetscObject)B, "MatHIPSPARSESetFormat_C", MatHIPSPARSESetFormat_SeqAIJHIPSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJHIPSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJHIPSPARSE(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJHIPSPARSE - MATAIJHIPSPARSE = "(seq)aijhipsparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the HIPSPARSE library.

   Options Database Keys:
+  -mat_type aijhipsparse - sets the matrix type to "seqaijhipsparse" during a call to MatSetFromOptions()
.  -mat_hipsparse_storage_format csr - sets the storage format of matrices (for MatMult and factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_mult_storage_format csr - sets the storage format of matrices (for MatMult) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

.seealso: MatCreateSeqAIJHIPSPARSE(), MATAIJHIPSPARSE, MatCreateAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat,MatFactorType,Mat*);


PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_HIPSPARSE(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_LU,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_CHOLESKY,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_ILU,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_ICC,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE **hipsparsestruct)
{
  hipsparseStatus_t stat;
  hipsparseHandle_t handle;

  PetscFunctionBegin;
  if (*hipsparsestruct) {
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->mat,(*hipsparsestruct)->format);
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->matTranspose,(*hipsparsestruct)->format);
    delete (*hipsparsestruct)->workVector;
    if (handle = (*hipsparsestruct)->handle) {
      stat = hipsparseDestroy(handle);CHKERRHIPSPARSE(stat);
    }
    delete *hipsparsestruct;
    *hipsparsestruct = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix **mat)
{
  PetscFunctionBegin;
  if (*mat) {
    delete (*mat)->values;
    delete (*mat)->column_indices;
    delete (*mat)->row_offsets;
    delete *mat;
    *mat = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct **trifactor)
{
  hipsparseStatus_t stat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (*trifactor) {
    if ((*trifactor)->descr) { stat = hipsparseDestroyMatDescr((*trifactor)->descr);CHKERRHIPSPARSE(stat); }
    if ((*trifactor)->solveInfo) { stat = hipsparseDestroySolveAnalysisInfo((*trifactor)->solveInfo);CHKERRHIPSPARSE(stat); }
    ierr = CsrMatrix_Destroy(&(*trifactor)->csrMat);CHKERRQ(ierr);
    delete *trifactor;
    *trifactor = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct **matstruct,MatHIPSPARSEStorageFormat format)
{
  CsrMatrix        *mat;
  hipsparseStatus_t stat;
  hipError_t      err;

  PetscFunctionBegin;
  if (*matstruct) {
    if ((*matstruct)->mat) {
      if (format==MAT_HIPSPARSE_ELL || format==MAT_HIPSPARSE_HYB) {
        hipsparseHybMat_t hybMat = (hipsparseHybMat_t)(*matstruct)->mat;
        stat = hipsparseDestroyHybMat(hybMat);CHKERRHIPSPARSE(stat);
      } else {
        mat = (CsrMatrix*)(*matstruct)->mat;
        CsrMatrix_Destroy(&mat);
      }
    }
    if ((*matstruct)->descr) { stat = hipsparseDestroyMatDescr((*matstruct)->descr);CHKERRHIPSPARSE(stat); }
    delete (*matstruct)->cprowIndices;
    if ((*matstruct)->alpha)     { err=hipFree((*matstruct)->alpha);CHKERRCUDA(err); }
    if ((*matstruct)->beta_zero) { err=hipFree((*matstruct)->beta_zero);CHKERRCUDA(err); }
    if ((*matstruct)->beta_one)  { err=hipFree((*matstruct)->beta_one);CHKERRCUDA(err); }
    delete *matstruct;
    *matstruct = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors** trifactors)
{
  hipsparseHandle_t handle;
  hipsparseStatus_t stat;

  PetscFunctionBegin;
  if (*trifactors) {
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtr);
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtr);
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtrTranspose);
    MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtrTranspose);
    delete (*trifactors)->rpermIndices;
    delete (*trifactors)->cpermIndices;
    delete (*trifactors)->workVector;
    if (handle = (*trifactors)->handle) {
      stat = hipsparseDestroy(handle);CHKERRHIPSPARSE(stat);
    }
    delete *trifactors;
    *trifactors = 0;
  }
  PetscFunctionReturn(0);
}

