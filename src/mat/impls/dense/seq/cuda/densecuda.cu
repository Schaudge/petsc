
/*
     Defines the matrix operations for sequential dense with CUDA
*/

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

/* TODO: Move to a different include file */
#include <cusolverDn.h>
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnCpotrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnCpotrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnCpotrs((a),(b),(c),(d),(cuComplex*)(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnCsytrf((a),(b),(c),(cuComplex*)(d),(e),(f),(cuComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnCsytrf_bufferSize((a),(b),(cuComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnCgetrf((a),(b),(c),(cuComplex*)(d),(e),(cuComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnCgetrf_bufferSize((a),(b),(c),(cuComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnCgetrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j))
#else /* complex double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnZpotrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnZpotrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnZpotrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnZsytrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(f),(cuDoubleComplex*)(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnZsytrf_bufferSize((a),(b),(cuDoubleComplex*)(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnZgetrf((a),(b),(c),(cuDoubleComplex*)(d),(e),(cuDoubleComplex*)(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnZgetrf_bufferSize((a),(b),(c),(cuDoubleComplex*)(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnZgetrs((a),(b),(c),(d),(cuDoubleComplex*)(e),(f),(g),(cuDoubleComplex*)(h),(i),(j))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnSpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnSpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnSpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnSsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnSsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnSgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnSgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnSgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#else /* real double */
#define cusolverDnXpotrf(a,b,c,d,e,f,g,h)        cusolverDnDpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXpotrf_bufferSize(a,b,c,d,e,f) cusolverDnDpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXpotrs(a,b,c,d,e,f,g,h,i)      cusolverDnDpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf(a,b,c,d,e,f,g,h,i)      cusolverDnDsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define cusolverDnXsytrf_bufferSize(a,b,c,d,e)   cusolverDnDsytrf_bufferSize((a),(b),(c),(d),(e))
#define cusolverDnXgetrf(a,b,c,d,e,f,g,h)        cusolverDnDgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define cusolverDnXgetrf_bufferSize(a,b,c,d,e,f) cusolverDnDgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define cusolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    cusolverDnDgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#endif
#endif

/* copy and pasted from the CUBLAS implementation */
/* Where to place the stream ? */
/*
cudaStream_t stream;
ccer = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);CHKERRCUDA(ccer);
cerr = cusolverDnSetStream(handle,stream);CHKERRCUSOLVER(cerr);
ccer = cudaStreamDestroy(stream);CHKERRCUDA(ccer);
*/
#define CHKERRCUSOLVER(err) do {if (PetscUnlikely(err)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSOLVER error %d",err);} while(0)
static PetscErrorCode PetscCUSOLVERDnDestroyHandle();
static PetscErrorCode PetscCUSOLVERDnGetHandle_Private(cusolverDnHandle_t **handle)
{
  static cusolverDnHandle_t cusolverdnhandle = NULL;
  cusolverStatus_t          cerr;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  if (!cusolverdnhandle) {
    cerr = cusolverDnCreate(&cusolverdnhandle);CHKERRCUSOLVER(cerr);
    ierr = PetscRegisterFinalize(PetscCUSOLVERDnDestroyHandle);CHKERRQ(ierr);
  }
  *handle = &cusolverdnhandle;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnInitializeHandle(void)
{
  cusolverDnHandle_t *p_cusolverdnhandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscCUSOLVERDnGetHandle_Private(&p_cusolverdnhandle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  cusolverDnHandle_t *p_cusolverdnhandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr    = PetscCUSOLVERDnGetHandle_Private(&p_cusolverdnhandle);CHKERRQ(ierr);
  *handle = *p_cusolverdnhandle;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnDestroyHandle()
{
  cusolverDnHandle_t *p_cusolverdnhandle;
  cusolverStatus_t cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUSOLVERDnGetHandle_Private(&p_cusolverdnhandle);CHKERRQ(ierr);
  cerr = cusolverDnDestroy(*p_cusolverdnhandle);CHKERRCUSOLVER(cerr);
  *p_cusolverdnhandle = NULL;  /* Ensures proper reinitialization */
  PetscFunctionReturn(0);
}

//TODO
//static PetscErrorCode MatSeqDenseSymmetrize_Private(Mat A, PetscBool hermitian)
//{
//  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
//  PetscInt      j, k, n = A->rmap->n;
//
//  PetscFunctionBegin;
//  if (A->rmap->n != A->cmap->n) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot symmetrize a rectangular matrix");
//  if (!hermitian) {
//    for (k=0;k<n;k++) {
//      for (j=k;j<n;j++) {
//        mat->v[j*mat->lda + k] = mat->v[k*mat->lda + j];
//      }
//    }
//  } else {
//    for (k=0;k<n;k++) {
//      for (j=k;j<n;j++) {
//        mat->v[j*mat->lda + k] = PetscConj(mat->v[k*mat->lda + j]);
//      }
//    }
//  }
//  PetscFunctionReturn(0);
//}
//
//PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat A)
//{
//#if defined(PETSC_MISSING_LAPACK_POTRF)
//  PetscFunctionBegin;
//  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable.");
//#else
//  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
//  PetscErrorCode ierr;
//  PetscBLASInt   info,n;
//
//  PetscFunctionBegin;
//  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
//  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
//  if (A->factortype == MAT_FACTOR_LU) {
//    if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
//    if (!mat->fwork) {
//      mat->lfwork = n;
//      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
//      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
//    }
//    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
//    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
//    ierr = PetscFPTrapPop();CHKERRQ(ierr);
//    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr); /* TODO CHECK FLOPS */
//  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
//    if (A->spd) {
//      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
//      PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&n,mat->v,&mat->lda,&info));
//      ierr = PetscFPTrapPop();CHKERRQ(ierr);
//      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
//#if defined(PETSC_USE_COMPLEX)
//    } else if (A->hermitian) {
//      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
//      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
//      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
//      PetscStackCallBLAS("LAPACKhetri",LAPACKhetri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
//      ierr = PetscFPTrapPop();CHKERRQ(ierr);
//      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
//#endif
//    } else { /* symmetric case */
//      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
//      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
//      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
//      PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
//      ierr = PetscFPTrapPop();CHKERRQ(ierr);
//      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_FALSE);CHKERRQ(ierr);
//    }
//    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad Inversion: zero pivot in row %D",(PetscInt)info-1);
//    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr); /* TODO CHECK FLOPS */
//  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
//#endif
//
//  A->ops->solve             = NULL;
//  A->ops->matsolve          = NULL;
//  A->ops->solvetranspose    = NULL;
//  A->ops->matsolvetranspose = NULL;
//  A->ops->solveadd          = NULL;
//  A->ops->solvetransposeadd = NULL;
//  A->factortype             = MAT_FACTOR_NONE;
//  ierr                      = PetscFree(A->solvertype);CHKERRQ(ierr);
//  PetscFunctionReturn(0);
//}


#if 0
/* ------------------------------------------------------------------*/
static PetscErrorCode MatSOR_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *x,*v = mat->v,zero = 0.0,xt;
  const PetscScalar *b;
  PetscErrorCode    ierr;
  PetscInt          m = A->rmap->n,i;
  PetscBLASInt      o = 1,bm;

  PetscFunctionBegin;
  if (shift == -1) shift = 0.0; /* negative shift indicates do not error on zero diagonal; this code never zeros on zero diagonal */
  ierr = PetscBLASIntCast(m,&bm);CHKERRQ(ierr);
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdotu */
    ierr = VecSet(xx,zero);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  its  = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        PetscStackCallBLAS("BLASdotu",xt   = b[i] - BLASdotu_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i=m-1; i>=0; i--) {
        PetscStackCallBLAS("BLASdotu",xt   = b[i] - BLASdotu_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_SeqDense(Mat A,const PetscScalar array[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  a->unplacedarray       = a->v;
  a->unplaced_user_alloc = a->user_alloc;
  a->v                   = (PetscScalar*) array;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseResetArray_SeqDense(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  a->v             = a->unplacedarray;
  a->user_alloc    = a->unplaced_user_alloc;
  a->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMax_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) < PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMaxAbs_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  PetscReal      atmp;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = PetscAbsScalar(aa[i]);
    for (j=1; j<n; j++) {
      atmp = PetscAbsScalar(aa[i+m*j]);
      if (PetscAbsScalar(x[i]) < atmp) {x[i] = atmp; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMin_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) > PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_SeqDense(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscArraycpy(x,a->v+col*a->lda,A->rmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnNorms_SeqDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,m,n;
  const PetscScalar *a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscArrayzero(norms,n);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] += PetscAbsScalar(a[j]*a[j]);
      }
      a += m;
    }
  } else if (type == NORM_1) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] += PetscAbsScalar(a[j]);
      }
      a += m;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] = PetscMax(PetscAbsScalar(a[j]),norms[i]);
      }
      a += m;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown NormType");
  ierr = MatDenseRestoreArrayRead(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = PetscSqrtReal(norms[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumn_SeqDense(Mat A,PetscInt col,PetscScalar **vals)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  *vals = a->v+col*a->lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumn_SeqDense(Mat A,PetscScalar **vals)
{
  PetscFunctionBegin;
  *vals = 0; /* user cannot accidently use the array later */
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = { MatSetValues_SeqDense,
                                        MatGetRow_SeqDense,
                                        MatRestoreRow_SeqDense,
                                        MatMult_SeqDense,
                                /*  4*/ MatMultAdd_SeqDense,
                                        MatMultTranspose_SeqDense,
                                        MatMultTransposeAdd_SeqDense,
                                        0,
                                        0,
                                        0,
                                /* 10*/ 0,
                                        MatLUFactor_SeqDense,
                                        MatCholeskyFactor_SeqDense,
                                        MatSOR_SeqDense,
                                        MatTranspose_SeqDense,
                                /* 15*/ MatGetInfo_SeqDense,
                                        MatEqual_SeqDense,
                                        MatGetDiagonal_SeqDense,
                                        MatDiagonalScale_SeqDense,
                                        MatNorm_SeqDense,
                                /* 20*/ MatAssemblyBegin_SeqDense,
                                        MatAssemblyEnd_SeqDense,
                                        MatSetOption_SeqDense,
                                        MatZeroEntries_SeqDense,
                                /* 24*/ MatZeroRows_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 29*/ MatSetUp_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 34*/ MatDuplicate_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 39*/ MatAXPY_SeqDense,
                                        MatCreateSubMatrices_SeqDense,
                                        0,
                                        MatGetValues_SeqDense,
                                        MatCopy_SeqDense,
                                /* 44*/ MatGetRowMax_SeqDense,
                                        MatScale_SeqDense,
                                        MatShift_Basic,
                                        0,
                                        MatZeroRowsColumns_SeqDense,
                                /* 49*/ MatSetRandom_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 54*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 59*/ 0,
                                        MatDestroy_SeqDense,
                                        MatView_SeqDense,
                                        0,
                                        0,
                                /* 64*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 69*/ MatGetRowMaxAbs_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 74*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 79*/ 0,
                                        0,
                                        0,
                                        0,
                                /* 83*/ MatLoad_SeqDense,
                                        0,
                                        MatIsHermitian_SeqDense,
                                        0,
                                        0,
                                        0,
                                /* 89*/ MatMatMult_SeqDense_SeqDense,
                                        MatMatMultSymbolic_SeqDense_SeqDense,
                                        MatMatMultNumeric_SeqDense_SeqDense,
                                        MatPtAP_SeqDense_SeqDense,
                                        MatPtAPSymbolic_SeqDense_SeqDense,
                                /* 94*/ MatPtAPNumeric_SeqDense_SeqDense,
                                        MatMatTransposeMult_SeqDense_SeqDense,
                                        MatMatTransposeMultSymbolic_SeqDense_SeqDense,
                                        MatMatTransposeMultNumeric_SeqDense_SeqDense,
                                        0,
                                /* 99*/ 0,
                                        0,
                                        0,
                                        MatConjugate_SeqDense,
                                        0,
                                /*104*/ 0,
                                        MatRealPart_SeqDense,
                                        MatImaginaryPart_SeqDense,
                                        0,
                                        0,
                                /*109*/ 0,
                                        0,
                                        MatGetRowMin_SeqDense,
                                        MatGetColumnVector_SeqDense,
                                        MatMissingDiagonal_SeqDense,
                                /*114*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /*119*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /*124*/ 0,
                                        MatGetColumnNorms_SeqDense,
                                        0,
                                        0,
                                        0,
                                /*129*/ 0,
                                        MatTransposeMatMult_SeqDense_SeqDense,
                                        MatTransposeMatMultSymbolic_SeqDense_SeqDense,
                                        MatTransposeMatMultNumeric_SeqDense_SeqDense,
                                        0,
                                /*134*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /*139*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /*144*/ MatCreateMPIMatConcatenateSeqMat_SeqDense
};
#endif
///*@C
//   MatCreateSeqDenseCUDA - Creates a sequential dense matrix that
//   is stored in column major order (the usual Fortran 77 manner). Many
//   of the matrix operations use the BLAS and LAPACK routines.
//
//   Collective
//
//   Input Parameters:
//+  comm - MPI communicator, set to PETSC_COMM_SELF
//.  m - number of rows
//.  n - number of columns
//-  data - optional location of matrix data in column major order.  Set data=NULL for PETSc
//   to control all matrix memory allocation.
//
//   Output Parameter:
//.  A - the matrix
//
//   Notes:
//   The data input variable is intended primarily for Fortran programmers
//   who wish to allocate their own matrix memory space.  Most users should
//   set data=NULL.
//
//   Level: intermediate
//
//.seealso: MatCreate(), MatCreateDense(), MatSetValues()
//@*/
//PetscErrorCode  MatCreateSeqDenseCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
//{
//  PetscErrorCode ierr;
//
//  PetscFunctionBegin;
//  ierr = MatCreate(comm,A);CHKERRQ(ierr);
//  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
//  ierr = MatSetType(*A,MATSEQDENSECUDA);CHKERRQ(ierr);
//  ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
//  PetscFunctionReturn(0);
//}

//PetscErrorCode MatSeqDenseSetPreallocation_SeqDense(Mat B,PetscScalar *data)
//{
//  Mat_SeqDense   *b;
//  PetscErrorCode ierr;
//
//  PetscFunctionBegin;
//  B->preallocated = PETSC_TRUE;
//
//  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
//  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
//
//  b       = (Mat_SeqDense*)B->data;
//  b->Mmax = B->rmap->n;
//  b->Nmax = B->cmap->n;
//  if (b->lda <= 0 || b->changelda) b->lda = B->rmap->n;
//
//  ierr = PetscIntMultError(b->lda,b->Nmax,NULL);CHKERRQ(ierr);
//  if (!data) { /* petsc-allocated storage */
//    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
//    ierr = PetscCalloc1((size_t)b->lda*b->Nmax,&b->v);CHKERRQ(ierr);
//    ierr = PetscLogObjectMemory((PetscObject)B,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRQ(ierr);
//
//    b->user_alloc = PETSC_FALSE;
//  } else { /* user-allocated storage */
//    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
//    b->v          = data;
//    b->user_alloc = PETSC_TRUE;
//  }
//  B->assembled = PETSC_TRUE;
//  PetscFunctionReturn(0);
//}

typedef struct {
  PetscScalar *d_v;   /* pointer to the matrix on the GPU */
  /* factorization support */
  int         *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_work; /* device workspace */
  int         fact_lwork;
  int         *d_fact_info; /* device info */
} Mat_SeqDenseCUDA;

PetscErrorCode MatSeqDenseCUDACopyFromGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",A->valid_GPU_matrix == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (A->valid_GPU_matrix == PETSC_OFFLOAD_GPU) {
    ierr = PetscLogEventBegin(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    /* TODO, no lda? */
    cerr = cudaMemcpy(cA->v,dA->d_v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    ierr = PetscLogGpuToCpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);

    A->valid_GPU_matrix = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseCUDACopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (A->pinnedtocpu) PetscFunctionReturn(0);
  if (A->valid_GPU_matrix == PETSC_OFFLOAD_UNALLOCATED) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Unallocated device memory");
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",A->valid_GPU_matrix == PETSC_OFFLOAD_CPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (A->valid_GPU_matrix == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    /* TODO, no lda? */
    cerr = cudaMemcpy(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);

    A->valid_GPU_matrix = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDAGetArrayWrite(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;

  PetscFunctionBegin;
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  A->valid_GPU_matrix = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDAGetArrayRead(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDARestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDAGetArray(Mat A, PetscScalar **a)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseCUDARestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  A->valid_GPU_matrix = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDenseCUDA(Mat A,Mat B,Mat X)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense       *x = (Mat_SeqDense*)X->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *dx;
  cusolverDnHandle_t handle;
  PetscMPIInt        nrhs,info,n,lda,ldx;
  cusolverStatus_t   cerr;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = MatCopy(B,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  /* TODO: MatMatSolve does not have a dispatching mechanism, we may end up with a MATSEQDENSE here */
  ierr = MatDenseCUDAGetArrayWrite(X,&dx);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(X->cmap->n,&nrhs);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(x->lda,&ldx);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = cusolverDnXgetrs(handle,CUBLAS_OP_N,n,nrhs,da,lda,dA->d_fact_ipiv,dx,ldx,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
      cerr = cusolverDnXpotrs(handle,CUBLAS_FILL_MODE_LOWER,n,nrhs,da,lda,dx,ldx,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayWrite(X,&dx);CHKERRQ(ierr);
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  ierr = PetscLogGpuFlops(nrhs*(2.0*n*n - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Private(Mat A,Vec xx,Vec yy,PetscBool trans)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *y;
  cusolverDnHandle_t handle;
  PetscMPIInt        one = 1,info,n,lda;
  cusolverStatus_t   cerr;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = VecCopy(xx,yy);CHKERRQ(ierr);
  /* may not be of type cuda */
  ierr = VecCUDAGetArrayWrite(yy,&y);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = cusolverDnXgetrs(handle,trans ? CUBLAS_OP_T : CUBLAS_OP_N,n,one,da,lda,dA->d_fact_ipiv,y,n,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit cudaErrorNotReady (error 34) due to "device not ready" on CUDA API call to cudaEventQuery. */
      cerr = cusolverDnXpotrs(handle,CUBLAS_FILL_MODE_LOWER,n,one,da,lda,y,n,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"cusolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  ierr = PetscLogGpuFlops(2.0*n*n - n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseCUDA_Private(A,xx,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseCUDA_Private(A,xx,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseCUDA(Mat A,IS rperm,IS cperm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscMPIInt        m,n,lda,info;
  cusolverStatus_t   cerr;
  cusolverDnHandle_t handle;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"LU factor %d x %d on backend\n",m,n);CHKERRQ(ierr);
  if (!dA->d_fact_ipiv) {
    ccer = cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRCUDA(ccer);
  }
  if (!dA->fact_lwork) {
    cerr = cusolverDnXgetrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
    ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
  }
  if (!dA->d_fact_info) {
    ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = cusolverDnXgetrf(handle,m,n,da,lda,dA->d_fact_work,dA->d_fact_ipiv,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
  ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  A->factortype = MAT_FACTOR_LU;
  ierr = PetscLogGpuFlops(2.0*n*n*m/3.0);CHKERRQ(ierr);

  A->ops->solve          = MatSolve_SeqDenseCUDA;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseCUDA;
  A->ops->matsolve       = MatMatSolve_SeqDenseCUDA;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseCUDA(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscScalar        *da;
  PetscMPIInt        n,lda,info;
  cusolverStatus_t   cerr;
  cusolverDnHandle_t handle;
  cudaError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"Cholesky factor %d x %d on backend\n",n,n);CHKERRQ(ierr);
  if (A->spd) {
    ierr = MatDenseCUDAGetArray(A,&da);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
    if (!dA->fact_lwork) {
      cerr = cusolverDnXpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
      ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cusolverDnXpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = MatDenseCUDARestoreArray(A,&da);CHKERRQ(ierr);
    ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(int), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
    A->factortype = MAT_FACTOR_CHOLESKY;
    ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  } else { /* symmetric case */
#if 0
    /* at the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs and *hetr* routines
       instead of erroring , we implement these factorizations using *getr*
       The code below works, and it can be activated when *sytrs routines will be available */
    if (!dA->d_fact_ipiv) {
      ccer = cudaMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRCUDA(ccer);
    }
    if (!dA->fact_lwork) {
      cerr = cusolverDnXsytrf_bufferSize(handle,n,da,lda,&dA->fact_lwork);CHKERRCUSOLVER(cerr);
      ccer = cudaMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRCUDA(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = cudaMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRCUDA(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cusolverDnXsytrf(handle,CUBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_ipiv,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRCUSOLVER(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    ierr = MatLUFactor_SeqDenseCUDA(A,perm,perm,factinfo);CHKERRQ(ierr);
  }

  A->ops->solve          = MatSolve_SeqDenseCUDA;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseCUDA;
  A->ops->matsolve       = MatMatSolve_SeqDenseCUDA;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
static PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(Mat A,Mat B,Mat C,PetscBool tA, PetscBool tB)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense      *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense      *c = (Mat_SeqDense*)C->data;
  const PetscScalar *da,*db;
  PetscScalar       *dc;
  PetscScalar       one=1.0,zero=0.0;
  PetscMPIInt       m,n,k,alda,blda,clda;
  PetscErrorCode    ierr;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;

  PetscFunctionBegin;
  ierr = PetscMPIIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  if (tA) {
    ierr = PetscMPIIntCast(A->rmap->n,&k);CHKERRQ(ierr);
  } else {
    ierr = PetscMPIIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  }
  if (!m || !n || !k) PetscFunctionReturn(0);
  ierr = PetscInfo3(C,"Matrix-Matrix product %d x %d x %d on backend\n",m,k,n);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayWrite(C,&dc);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&alda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(b->lda,&blda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(c->lda,&clda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemm(cublasv2handle,tA ? CUBLAS_OP_T : CUBLAS_OP_N,tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,k,&one,da,alda,db,blda,&zero,dc,clda);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayWrite(C,&dc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA_Private(A,B,C,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* zz = op(A)*xx + yy
   if yy == NULL, only MatMult */
static PetscErrorCode MatMultAdd_SeqDenseCUDA_Private(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *xarray,*da;
  PetscScalar       *zarray;
  PetscScalar       one=1.0,zero=0.0;
  PetscMPIInt       m, n, lda; /* Use PetscMPIInt as it is typedef'ed to int */
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;
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
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(mat->lda,&lda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemv(cublasv2handle,trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                     m,n,&one,da,lda,xarray,1,(yy ? &one : &zero),zarray,1);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*A->rmap->n*A->cmap->n - (yy ? 0 : A->rmap->n));CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,yy,zz,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseCUDA(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseCUDA_Private(A,xx,NULL,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetArrayRead_SeqDenseCUDA(Mat A,const PetscScalar *array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatDenseGetArray_SeqDense(A,(PetscScalar**)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetArray_SeqDenseCUDA(Mat A,PetscScalar *array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatDenseGetArray_SeqDense(A,array);CHKERRQ(ierr);
  A->valid_GPU_matrix = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_SeqDenseCUDA(Mat A,PetscScalar *array[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseRestoreArray_SeqDense(A,array);CHKERRQ(ierr);
  ierr = MatSeqDenseCUDACopyToGPU(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseCUDA(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense*)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  PetscBLASInt      j,N,m,ldax,lday,one = 1;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    berr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseCUDAGetArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseCUDAGetArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDAGetArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  ierr = PetscBLASIntCast(X->rmap->n*X->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(x->lda,&ldax);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(y->lda,&lday);CHKERRQ(ierr);
  ierr = PetscInfo2(Y,"Performing AXPY %d x %d on backend\n",Y->rmap->n,Y->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      berr = cublasXaxpy(cublasv2handle,m,&alpha,dx+j*ldax,one,dy+j*lday,one);CHKERRCUBLAS(berr);
    }
  } else {
    berr = cublasXaxpy(cublasv2handle,N,&alpha,dx,one,dy,one);CHKERRCUBLAS(berr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*N-1,0));CHKERRQ(ierr);
  ierr = MatDenseCUDARestoreArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseCUDARestoreArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseCUDARestoreArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseCUDA(Mat A)
{
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  cudaError_t      cerr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dA) {
    cerr = cudaFree(dA->d_v);CHKERRCUDA(cerr);
    cerr = cudaFree(dA->d_fact_ipiv);CHKERRCUDA(cerr);
    cerr = cudaFree(dA->d_fact_info);CHKERRCUDA(cerr);
    cerr = cudaFree(dA->d_fact_work);CHKERRCUDA(cerr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatReset_SeqDenseCUDA(A);CHKERRQ(ierr);
  ierr = MatDestroy_SeqDense(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseSetPreallocation_SeqDenseCUDA(Mat B,PetscScalar *data)
{
  Mat_SeqDense      *b;
  Mat_SeqDenseCUDA* dB;
  cudaError_t       cerr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  b       = (Mat_SeqDense*)B->data;
  b->Mmax = B->rmap->n;
  b->Nmax = B->cmap->n;
  if (b->lda <= 0 || b->changelda) b->lda = B->rmap->n;
  if (b->lda < B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid lda %D < %D",b->lda,B->rmap->n);

  ierr = PetscIntMultError(b->lda,b->Nmax,NULL);CHKERRQ(ierr);

  ierr     = MatReset_SeqDenseCUDA(B);CHKERRQ(ierr);
  ierr     = PetscNewLog(B,&dB);CHKERRQ(ierr);
  B->spptr = dB;
  cerr     = cudaMalloc((void**)&dB->d_v,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRCUDA(cerr);

  if (!data) { /* petsc-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    ierr = PetscCalloc1((size_t)b->lda*b->Nmax,&b->v);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)B,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRQ(ierr);
    b->user_alloc       = PETSC_FALSE;
  } else { /* user-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    b->v                = data;
    b->user_alloc       = PETSC_TRUE;
  }
  B->valid_GPU_matrix = PETSC_OFFLOAD_CPU;
  B->preallocated     = PETSC_TRUE;
  B->assembled        = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*B,A,cpvalues);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES && A->valid_GPU_matrix != PETSC_OFFLOAD_CPU) {
    Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
    const PetscScalar *da;
    PetscScalar       *db;
    cudaError_t       cerr;

    ierr = MatDenseCUDAGetArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseCUDAGetArrayWrite(*B,&db);CHKERRQ(ierr);
    if (a->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* it can be done better */
        cerr = cudaMemcpy(db+j*m,da+j*a->lda,m*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
      }
    } else {
      cerr = cudaMemcpy(db,da,a->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);
    }
    ierr = MatDenseCUDARestoreArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseCUDARestoreArrayWrite(*B,&db);CHKERRQ(ierr);
    (*B)->valid_GPU_matrix = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,MATSEQDENSECUDA);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCUDA,&(*fact)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPinToCPU_SeqDenseCUDA(Mat A,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  A->pinnedtocpu = flg;
  if (!flg) {
    /* make sure we have an up-to-date copy on the CPU */
    ierr = MatSeqDenseCUDACopyFromGPU(A);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",           MatDenseGetArray_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",       MatDenseGetArrayRead_SeqDenseCUDA);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreArray_C",       MatDenseRestoreArray_SeqDenseCUDA);CHKERRQ(ierr);

    A->ops->destroy                 = MatDestroy_SeqDenseCUDA;
    A->ops->duplicate               = MatDuplicate_SeqDenseCUDA;
    A->ops->mult                    = MatMult_SeqDenseCUDA;
    A->ops->multadd                 = MatMultAdd_SeqDenseCUDA;
    A->ops->multtranspose           = MatMultTranspose_SeqDenseCUDA;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDenseCUDA;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA;
    A->ops->axpy                    = MatAXPY_SeqDenseCUDA;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDenseCUDA;
    A->ops->lufactor                = MatLUFactor_SeqDenseCUDA;
  } else {
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",           MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",       MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreArray_C",       MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);

    A->ops->destroy                 = MatDestroy_SeqDense;
    A->ops->duplicate               = MatDuplicate_SeqDense;
    A->ops->mult                    = MatMult_SeqDense;
    A->ops->multadd                 = MatMultAdd_SeqDense;
    A->ops->multtranspose           = MatMultTranspose_SeqDense;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDense;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDense_SeqDense;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDense_SeqDense;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDense_SeqDense;
    A->ops->axpy                    = MatAXPY_SeqDense;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDense;
    A->ops->lufactor                = MatLUFactor_SeqDense;
 }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSECUDA);CHKERRQ(ierr);

  ierr = MatPinToCPU_SeqDenseCUDA(B,PETSC_FALSE);CHKERRQ(ierr);
  B->ops->pintocpu = MatPinToCPU_SeqDenseCUDA;
  B->valid_GPU_matrix = PETSC_OFFLOAD_UNALLOCATED;

  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSECUDA - MATSEQDENSECUDA = "seqdensecuda" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensecuda - sets the matrix type to "seqdensecuda" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqDenseCuda()

M*/

PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqDense(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqDense_SeqDenseCUDA(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
