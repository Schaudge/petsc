/*
     Defines the matrix operations for sequential dense with HIP
*/
#include <petscpkg_version.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petschipblas.h>

/* hipblas definitions are here */
#include <../src/vec/vec/impls/seq/seqhip/hipvecimpl.h>

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define hipsolverDnXpotrf(a,b,c,d,e,f,g,h)        hipsolverDnCpotrf((a),(b),(c),(hipComplex*)(d),(e),(hipComplex*)(f),(g),(h))
#define hipsolverDnXpotrf_bufferSize(a,b,c,d,e,f) hipsolverDnCpotrf_bufferSize((a),(b),(c),(hipComplex*)(d),(e),(f))
#define hipsolverDnXpotrs(a,b,c,d,e,f,g,h,i)      hipsolverDnCpotrs((a),(b),(c),(d),(hipComplex*)(e),(f),(hipComplex*)(g),(h),(i))
#define hipsolverDnXpotri(a,b,c,d,e,f,g,h)        hipsolverDnCpotri((a),(b),(c),(hipComplex*)(d),(e),(hipComplex*)(f),(g),(h))
#define hipsolverDnXpotri_bufferSize(a,b,c,d,e,f) hipsolverDnCpotri_bufferSize((a),(b),(c),(hipComplex*)(d),(e),(f))
#define hipsolverDnXsytrf(a,b,c,d,e,f,g,h,i)      hipsolverDnCsytrf((a),(b),(c),(hipComplex*)(d),(e),(f),(hipComplex*)(g),(h),(i))
#define hipsolverDnXsytrf_bufferSize(a,b,c,d,e)   hipsolverDnCsytrf_bufferSize((a),(b),(hipComplex*)(c),(d),(e))
#define hipsolverDnXgetrf(a,b,c,d,e,f,g,h)        hipsolverDnCgetrf((a),(b),(c),(hipComplex*)(d),(e),(hipComplex*)(f),(g),(h))
#define hipsolverDnXgetrf_bufferSize(a,b,c,d,e,f) hipsolverDnCgetrf_bufferSize((a),(b),(c),(hipComplex*)(d),(e),(f))
#define hipsolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    hipsolverDnCgetrs((a),(b),(c),(d),(hipComplex*)(e),(f),(g),(hipComplex*)(h),(i),(j))
#else /* complex double */
#define hipsolverDnXpotrf(a,b,c,d,e,f,g,h)        hipsolverDnZpotrf((a),(b),(c),(hipDoubleComplex*)(d),(e),(hipDoubleComplex*)(f),(g),(h))
#define hipsolverDnXpotrf_bufferSize(a,b,c,d,e,f) hipsolverDnZpotrf_bufferSize((a),(b),(c),(hipDoubleComplex*)(d),(e),(f))
#define hipsolverDnXpotrs(a,b,c,d,e,f,g,h,i)      hipsolverDnZpotrs((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(hipDoubleComplex*)(g),(h),(i))
#define hipsolverDnXpotri(a,b,c,d,e,f,g,h)        hipsolverDnZpotri((a),(b),(c),(hipDoubleComplex*)(d),(e),(hipDoubleComplex*)(f),(g),(h))
#define hipsolverDnXpotri_bufferSize(a,b,c,d,e,f) hipsolverDnZpotri_bufferSize((a),(b),(c),(hipDoubleComplex*)(d),(e),(f))
#define hipsolverDnXsytrf(a,b,c,d,e,f,g,h,i)      hipsolverDnZsytrf((a),(b),(c),(hipDoubleComplex*)(d),(e),(f),(hipDoubleComplex*)(g),(h),(i))
#define hipsolverDnXsytrf_bufferSize(a,b,c,d,e)   hipsolverDnZsytrf_bufferSize((a),(b),(hipDoubleComplex*)(c),(d),(e))
#define hipsolverDnXgetrf(a,b,c,d,e,f,g,h)        hipsolverDnZgetrf((a),(b),(c),(hipDoubleComplex*)(d),(e),(hipDoubleComplex*)(f),(g),(h))
#define hipsolverDnXgetrf_bufferSize(a,b,c,d,e,f) hipsolverDnZgetrf_bufferSize((a),(b),(c),(hipDoubleComplex*)(d),(e),(f))
#define hipsolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    hipsolverDnZgetrs((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(g),(hipDoubleComplex*)(h),(i),(j))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define hipsolverDnXpotrf(a,b,c,d,e,f,g,h)        hipsolverDnSpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXpotrf_bufferSize(a,b,c,d,e,f) hipsolverDnSpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXpotrs(a,b,c,d,e,f,g,h,i)      hipsolverDnSpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverDnXpotri(a,b,c,d,e,f,g,h)        hipsolverDnSpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXpotri_bufferSize(a,b,c,d,e,f) hipsolverDnSpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXsytrf(a,b,c,d,e,f,g,h,i)      hipsolverDnSsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverDnXsytrf_bufferSize(a,b,c,d,e)   hipsolverDnSsytrf_bufferSize((a),(b),(c),(d),(e))
#define hipsolverDnXgetrf(a,b,c,d,e,f,g,h)        hipsolverDnSgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXgetrf_bufferSize(a,b,c,d,e,f) hipsolverDnSgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    hipsolverDnSgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#else /* real double */
#define hipsolverDnXpotrf(a,b,c,d,e,f,g,h)        hipsolverDnDpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXpotrf_bufferSize(a,b,c,d,e,f) hipsolverDnDpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXpotrs(a,b,c,d,e,f,g,h,i)      hipsolverDnDpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverDnXpotri(a,b,c,d,e,f,g,h)        hipsolverDnDpotri((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXpotri_bufferSize(a,b,c,d,e,f) hipsolverDnDpotri_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXsytrf(a,b,c,d,e,f,g,h,i)      hipsolverDnDsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverDnXsytrf_bufferSize(a,b,c,d,e)   hipsolverDnDsytrf_bufferSize((a),(b),(c),(d),(e))
#define hipsolverDnXgetrf(a,b,c,d,e,f,g,h)        hipsolverDnDgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverDnXgetrf_bufferSize(a,b,c,d,e,f) hipsolverDnDgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverDnXgetrs(a,b,c,d,e,f,g,h,i,j)    hipsolverDnDgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#endif
#endif

typedef struct {
  PetscScalar *d_v;   /* pointer to the matrix on the GPU */
  /* factorization support */
  int         *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_work; /* device workspace */
  int         fact_lwork;
  int         *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseHIP;

PetscErrorCode MatSeqDenseHIPCopyFromGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;
  hipError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr = PetscLogEventBegin(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* TODO: it can be done better */
        cerr = hipMemcpy(cA->v + j*cA->lda,dA->d_v + j*cA->lda,m*sizeof(PetscScalar),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
      }
    } else {
      cerr = hipMemcpy(cA->v,dA->d_v,cA->lda*sizeof(PetscScalar)*A->cmap->n,hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    }
    ierr = PetscLogGpuToCpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyFromGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseHIPCopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscBool        copy;
  PetscErrorCode   ierr;
  hipError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  if (A->pinnedtocpu) PetscFunctionReturn(0);
  if (!dA->d_v) {
    cerr = hipMalloc((void**)&dA->d_v,cA->lda*cA->Nmax*sizeof(PetscScalar));CHKERRHIP(cerr);
  }
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (copy) {
    ierr = PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* TODO: it can be done better */
        cerr = hipMemcpy(dA->d_v + j*cA->lda,cA->v + j*cA->lda,m*sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
      }
    } else {
      cerr = hipMemcpy(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,hipMemcpyHostToDevice);CHKERRHIP(cerr);
    }
    ierr = PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPGetArrayWrite(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  if (!dA->d_v) {
    Mat_SeqDense *cA = (Mat_SeqDense*)A->data;
    hipError_t  cerr;

    cerr = hipMalloc((void**)&dA->d_v,cA->lda*cA->Nmax*sizeof(PetscScalar));CHKERRHIP(cerr);
  }
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPRestoreArrayWrite(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPGetArrayRead(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPRestoreArrayRead(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPGetArray(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseHIPRestoreArray(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPInvertFactors_Private(Mat A)
{
#if PETSC_PKG_HIP_VERSION_GE(10,1,0)
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP   *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  PetscErrorCode     ierr;
  hipError_t        ccer;
  hipsolverStatus_t   cerr;
  hipsolverDnHandle_t handle;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDngetri not implemented");
  else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      int il;

      ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
      cerr = hipsolverDnXpotri_bufferSize(handle,HIPBLAS_FILL_MODE_LOWER,n,da,lda,&il);CHKERRHIPSOLVER(cerr);
      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        ccer = hipFree(dA->d_fact_work);CHKERRHIP(ccer);
        ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
      }
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = hipsolverDnXpotri(handle,HIPBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
      ccer = WaitForGPU();CHKERRHIP(ccer);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = MatDenseHIPRestoreArray(A,&da);CHKERRQ(ierr);
      /* TODO (write hip kernel) */
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDnsytri not implemented");
  }
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: leading minor of order %d is zero",info);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Upgrade to HIP version 10.1.0 or higher");
#endif
}

static PetscErrorCode MatMatSolve_SeqDenseHIP(Mat A,Mat B,Mat X)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense       *x = (Mat_SeqDense*)X->data;
  Mat_SeqDenseHIP   *dA = (Mat_SeqDenseHIP*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *dx;
  hipsolverDnHandle_t handle;
  PetscBool          iship;
  int                nrhs,n,lda,ldx;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipError_t        ccer;
  hipsolverStatus_t   cerr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&iship,VECSEQHIP,VECMPIHIP,"");CHKERRQ(ierr);
  if (X != B) {
    ierr = MatCopy(B,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
  /* MatMatSolve does not have a dispatching mechanism, we may end up with a MATSEQDENSE here */
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSEHIP,&iship);CHKERRQ(ierr);
  if (!iship) {
    ierr = MatConvert(X,MATSEQDENSEHIP,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
  ierr = MatDenseHIPGetArray(X,&dx);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(X->cmap->n,&nrhs);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(x->lda,&ldx);CHKERRQ(ierr);
  ierr = PetscHIPSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = hipsolverDnXgetrs(handle,HIPBLAS_OP_N,n,nrhs,da,lda,dA->d_fact_ipiv,dx,ldx,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit hipErrorNotReady (error 34) due to "device not ready" on HIP API call to hipEventQuery. */
      cerr = hipsolverDnXpotrs(handle,HIPBLAS_FILL_MODE_LOWER,n,nrhs,da,lda,dx,ldx,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ccer = WaitForGPU();CHKERRHIP(ccer);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(X,&dx);CHKERRQ(ierr);
  if (!iship) {
    ierr = MatConvert(X,MATSEQDENSE,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(nrhs*(2.0*n*n - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP_Private(Mat A,Vec xx,Vec yy,PetscBool trans)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP   *dA = (Mat_SeqDenseHIP*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *y;
  hipsolverDnHandle_t handle;
  int                one = 1,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipError_t        ccer;
  hipsolverStatus_t   cerr;
  PetscBool          iship;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  if (!dA->d_fact_work) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  /* MatSolve does not have a dispatching mechanism, we may end up with a VECSTANDARD here */
  ierr = PetscObjectTypeCompareAny((PetscObject)yy,&iship,VECSEQHIP,VECMPIHIP,"");CHKERRQ(ierr);
  if (iship) {
    ierr = VecCopy(xx,yy);CHKERRQ(ierr);
    ierr = VecHIPGetArray(yy,&y);CHKERRQ(ierr);
  } else {
    if (!dA->workvec) {
      ierr = MatCreateVecs(A,&dA->workvec,NULL);CHKERRQ(ierr);
    }
    ierr = VecCopy(xx,dA->workvec);CHKERRQ(ierr);
    ierr = VecHIPGetArray(dA->workvec,&y);CHKERRQ(ierr);
  }
  ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscHIPSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = hipsolverDnXgetrs(handle,trans ? HIPBLAS_OP_T : HIPBLAS_OP_N,n,one,da,lda,dA->d_fact_ipiv,y,n,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit hipErrorNotReady (error 34) due to "device not ready" on HIP API call to hipEventQuery. */
      cerr = hipsolverDnXpotrs(handle,HIPBLAS_FILL_MODE_LOWER,n,one,da,lda,y,n,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ccer = WaitForGPU();CHKERRHIP(ccer);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (iship) {
    ierr = VecHIPRestoreArray(yy,&y);CHKERRQ(ierr);
  } else {
    ierr = VecHIPRestoreArray(dA->workvec,&y);CHKERRQ(ierr);
    ierr = VecCopy(dA->workvec,yy);CHKERRQ(ierr);
  }
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(2.0*n*n - n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseHIP(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseHIP_Private(A,xx,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDenseHIP(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDenseHIP_Private(A,xx,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactor_SeqDenseHIP(Mat A,IS rperm,IS cperm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP   *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  int                m,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipsolverStatus_t   cerr;
  hipsolverDnHandle_t handle;
  hipError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"LU factor %d x %d on backend\n",m,n);CHKERRQ(ierr);
  if (!dA->d_fact_ipiv) {
    ccer = hipMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRHIP(ccer);
  }
  if (!dA->fact_lwork) {
    cerr = hipsolverDnXgetrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork);CHKERRHIPSOLVER(cerr);
    ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
  }
  if (!dA->d_fact_info) {
    ccer = hipMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRHIP(ccer);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = hipsolverDnXgetrf(handle,m,n,da,lda,dA->d_fact_work,dA->d_fact_ipiv,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
  ccer = WaitForGPU();CHKERRHIP(ccer);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
  A->factortype = MAT_FACTOR_LU;
  ierr = PetscLogGpuFlops(2.0*n*n*m/3.0);CHKERRQ(ierr);

  A->ops->solve          = MatSolve_SeqDenseHIP;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseHIP;
  A->ops->matsolve       = MatMatSolve_SeqDenseHIP;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERHIP,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseHIP(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP   *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipsolverStatus_t   cerr;
  hipsolverDnHandle_t handle;
  hipError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERDnGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"Cholesky factor %d x %d on backend\n",n,n);CHKERRQ(ierr);
  if (A->spd) {
    ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
    if (!dA->fact_lwork) {
      cerr = hipsolverDnXpotrf_bufferSize(handle,HIPBLAS_FILL_MODE_LOWER,n,da,lda,&dA->fact_lwork);CHKERRHIPSOLVER(cerr);
      ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = hipMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRHIP(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = hipsolverDnXpotrf(handle,HIPBLAS_FILL_MODE_LOWER,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    ccer = WaitForGPU();CHKERRHIP(ccer);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatDenseHIPRestoreArray(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"hipsolverDnsytrs unavailable. Use MAT_FACTOR_LU");
  A->ops->solve          = MatSolve_SeqDenseHIP;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseHIP;
  A->ops->matsolve       = MatMatSolve_SeqDenseHIP;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERHIP,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
static PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(Mat A,Mat B,Mat C,PetscBool tA, PetscBool tB)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense      *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense      *c = (Mat_SeqDense*)C->data;
  const PetscScalar *da,*db;
  PetscScalar       *dc;
  PetscScalar       one=1.0,zero=0.0;
  int               m,n,k,alda,blda,clda;
  PetscErrorCode    ierr;
  hipblasHandle_t    cublasv2handle;
  hipblasStatus_t    berr;
  hipError_t       cerr;

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
  ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayWrite(C,&dc);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&alda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(b->lda,&blda);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(c->lda,&clda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemm(cublasv2handle,tA ? HIPBLAS_OP_T : HIPBLAS_OP_N,tB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                     m,n,k,&one,da,alda,db,blda,&zero,dc,clda);CHKERRCUBLAS(berr);
  cerr = WaitForGPU();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayWrite(C,&dc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A,B,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A,B,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDenseHIP_SeqDenseHIP(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(A,B,C,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* zz = op(A)*xx + yy
   if yy == NULL, only MatMult */
static PetscErrorCode MatMultAdd_SeqDenseHIP_Private(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *xarray,*da;
  PetscScalar       *zarray;
  PetscScalar       one=1.0,zero=0.0;
  int               m, n, lda; /* Use PetscMPIInt as it is typedef'ed to int */
  hipblasHandle_t    cublasv2handle;
  hipblasStatus_t    berr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (yy && yy != zz) { /* mult add */
    ierr = VecCopy_SeqHIP(yy,zz);CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) {
    if (!yy) { /* mult only */
      ierr = VecSet_SeqHIP(zz,0.0);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo2(A,"Matrix-vector product %d x %d on backend\n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(mat->lda,&lda);CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = cublasXgemv(cublasv2handle,trans ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                     m,n,&one,da,lda,xarray,1,(yy ? &one : &zero),zarray,1);CHKERRCUBLAS(berr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*A->rmap->n*A->cmap->n - (yy ? 0 : A->rmap->n));CHKERRQ(ierr);
  ierr = VecHIPRestoreArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDenseHIP(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseHIP_Private(A,xx,yy,zz,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDenseHIP(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseHIP_Private(A,xx,yy,zz,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDenseHIP(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseHIP_Private(A,xx,NULL,yy,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqDenseHIP(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd_SeqDenseHIP_Private(A,xx,NULL,yy,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetArrayRead_SeqDenseHIP(Mat A,const PetscScalar *array[])
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetArray_SeqDenseHIP(Mat A,PetscScalar *array[])
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_SeqDenseHIP(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseHIP(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense*)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  int               j,N,m,ldax,lday,one = 1;
  hipblasHandle_t    cublasv2handle;
  hipblasStatus_t    berr;
  PetscErrorCode    ierr;
  hipError_t       cerr;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseHIPGetArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseHIPGetArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  ierr = PetscMPIIntCast(X->rmap->n*X->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(X->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(x->lda,&ldax);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(y->lda,&lday);CHKERRQ(ierr);
  ierr = PetscInfo2(Y,"Performing AXPY %d x %d on backend\n",Y->rmap->n,Y->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      berr = cublasXaxpy(cublasv2handle,m,&alpha,dx+j*ldax,one,dy+j*lday,one);CHKERRCUBLAS(berr);
    }
  } else {
    berr = cublasXaxpy(cublasv2handle,N,&alpha,dx,one,dy,one);CHKERRCUBLAS(berr);
  }
  cerr = WaitForGPU();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2.*N-1,0));CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(X,&dx);CHKERRQ(ierr);
  if (alpha != 0.0) {
    ierr = MatDenseHIPRestoreArray(Y,&dy);CHKERRQ(ierr);
  } else {
    ierr = MatDenseHIPRestoreArrayWrite(Y,&dy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_SeqDenseHIP(Mat A)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  hipError_t      cerr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (dA) {
    cerr = hipFree(dA->d_v);CHKERRHIP(cerr);
    cerr = hipFree(dA->d_fact_ipiv);CHKERRHIP(cerr);
    cerr = hipFree(dA->d_fact_info);CHKERRHIP(cerr);
    cerr = hipFree(dA->d_fact_work);CHKERRHIP(cerr);
    ierr = VecDestroy(&dA->workvec);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDenseHIP(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* prevent to copy back data if we own the data pointer */
  if (!a->user_alloc) { A->offloadmask = PETSC_OFFLOAD_CPU; }
  ierr = MatConvert_SeqDenseHIP_SeqDense(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatDestroy_SeqDense(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqDenseSetPreallocation_SeqDenseHIP(Mat B,PetscScalar *data)
{
  Mat_SeqDense     *b;
  Mat_SeqDenseHIP *dB;
  hipError_t      cerr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  b       = (Mat_SeqDense*)B->data;
  b->Mmax = B->rmap->n;
  b->Nmax = B->cmap->n;
  if (b->lda <= 0 || b->changelda) b->lda = B->rmap->n;
  if (b->lda < B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid lda %D < %D",b->lda,B->rmap->n);

  ierr = PetscIntMultError(b->lda,b->Nmax,NULL);CHKERRQ(ierr);

  ierr     = MatReset_SeqDenseHIP(B);CHKERRQ(ierr);
  ierr     = PetscNewLog(B,&dB);CHKERRQ(ierr);
  B->spptr = dB;
  cerr     = hipMalloc((void**)&dB->d_v,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRHIP(cerr);

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
  B->offloadmask = PETSC_OFFLOAD_CPU;
  B->preallocated     = PETSC_TRUE;
  B->assembled        = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDenseHIP(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*B,A,cpvalues);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) {
    Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
    const PetscScalar *da;
    PetscScalar       *db;
    hipError_t       cerr;

    ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseHIPGetArrayWrite(*B,&db);CHKERRQ(ierr);
    if (a->lda > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* it can be done better */
        cerr = hipMemcpy(db+j*m,da+j*a->lda,m*sizeof(PetscScalar),hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
      }
    } else {
      cerr = hipMemcpy(db,da,a->lda*sizeof(PetscScalar)*A->cmap->n,hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
    }
    ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseHIPRestoreArrayWrite(*B,&db);CHKERRQ(ierr);
    (*B)->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_hip(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,MATSEQDENSEHIP);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERHIP,&(*fact)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPinToCPU_SeqDenseHIP(Mat A,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  A->pinnedtocpu = flg;
  if (!flg) {
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",           MatDenseGetArray_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",       MatDenseGetArrayRead_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreArray_C",       MatDenseRestoreArray_SeqDenseHIP);CHKERRQ(ierr);

    A->ops->duplicate               = MatDuplicate_SeqDenseHIP;
    A->ops->mult                    = MatMult_SeqDenseHIP;
    A->ops->multadd                 = MatMultAdd_SeqDenseHIP;
    A->ops->multtranspose           = MatMultTranspose_SeqDenseHIP;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDenseHIP;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDenseHIP_SeqDenseHIP;
    A->ops->axpy                    = MatAXPY_SeqDenseHIP;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDenseHIP;
    A->ops->lufactor                = MatLUFactor_SeqDenseHIP;
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",           MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",       MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreArray_C",       MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);

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

PetscErrorCode MatConvert_SeqDenseHIP_SeqDense(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat              B;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    ierr = MatConvert_Basic(M,type,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  ierr = MatPinToCPU_SeqDenseHIP(B,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatReset_SeqDenseHIP(B);CHKERRQ(ierr);
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensehip_seqdense_C",NULL);CHKERRQ(ierr);

  B->ops->pintocpu    = NULL;
  B->ops->destroy     = MatDestroy_SeqDense;
  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqDense_SeqDenseHIP(Mat M,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat_SeqDenseHIP *dB;
  Mat              B;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    /* TODO these cases should be optimized */
    ierr = MatConvert_Basic(M,type,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECHIP,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSEHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensehip_seqdense_C",MatConvert_SeqDenseHIP_SeqDense);CHKERRQ(ierr);

  ierr     = MatReset_SeqDenseHIP(B);CHKERRQ(ierr);
  ierr     = PetscNewLog(B,&dB);CHKERRQ(ierr);
  B->spptr = dB;

  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  ierr = MatPinToCPU_SeqDenseHIP(B,PETSC_FALSE);CHKERRQ(ierr);
  B->ops->pintocpu = MatPinToCPU_SeqDenseHIP;
  B->ops->destroy  = MatDestroy_SeqDenseHIP;
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSEHIP - MATSEQDENSEHIP = "seqdensehip" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensehip - sets the matrix type to "seqdensehip" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqDenseHip()

M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseHIP(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqDense(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqDense_SeqDenseHIP(B,MATSEQDENSEHIP,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
