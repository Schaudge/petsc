/*
     Defines the matrix operations for sequential dense with HIP/ROCm
*/
#include <petscpkg_version.h>
#define PETSC_SKIP_IMMINTRIN_H_HIPWORKAROUND 1
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petsc/private/hipvecimpl.h> /* rocblas definitions are here */

/* Use hipsolver for our own namespace -- fundamental library is rocsolver */
/* NOTE ON potri:  Not in the 4.x versions of rocsolver (why?) */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define hipsolverXpotrf(a,b,c,d,e,f,g,h)        rocsolver_cpotrf((a),(b),(c),(rocblas_float_complex*)(d),(e),(rocblas_float_complex*)(f),(g),(h))
#define hipsolverXpotrf_bufferSize(a,b,c,d,e,f) rocsolver_cpotrf_bufferSize((a),(b),(c),(rocblas_float_complex*)(d),(e),(f))
#define hipsolverXpotrs(a,b,c,d,e,f,g,h,i)      rocsolver_cpotrs((a),(b),(c),(d),(rocblas_float_complex*)(e),(f),(rocblas_float_complex*)(g),(h),(i))
/* #define hipsolverXpotri(a,b,c,d,e,f,g,h)        rocsolver_cpotri((a),(b),(c),(rocblas_float_complex*)(d),(e),(rocblas_float_complex*)(f),(g),(h)) */
/* #define hipsolverXpotri_bufferSize(a,b,c,d,e,f) rocsolver_cpotri_bufferSize((a),(b),(c),(rocblas_float_complex*)(d),(e),(f)) */
#define hipsolverXsytrf(a,b,c,d,e,f,g,h,i)      rocsolver_csytrf((a),(b),(c),(rocblas_float_complex*)(d),(e),(f),(rocblas_float_complex*)(g),(h),(i))
#define hipsolverXsytrf_bufferSize(a,b,c,d,e)   rocsolver_csytrf_bufferSize((a),(b),(rocblas_float_complex*)(c),(d),(e))
#define hipsolverXgetrf(a,b,c,d,e,f,g,h)        rocsolver_cgetrf((a),(b),(c),(rocblas_float_complex*)(d),(e),(rocblas_float_complex*)(f),(g),(h))
#define hipsolverXgetrf_bufferSize(a,b,c,d,e,f) rocsolver_cgetrf_bufferSize((a),(b),(c),(rocblas_float_complex*)(d),(e),(f))
#define hipsolverXgetrs(a,b,c,d,e,f,g,h,i,j)    rocsolver_cgetrs((a),(b),(c),(d),(rocblas_float_complex*)(e),(f),(g),(rocblas_float_complex*)(h),(i),(j))
#else /* complex double */
#define hipsolverXpotrf(a,b,c,d,e,f,g,h)        rocsolver_zpotrf((a),(b),(c),(rocblas_double_complex*)(d),(e),(rocblas_double_complex*)(f),(g),(h))
#define hipsolverXpotrf_bufferSize(a,b,c,d,e,f) rocsolver_zpotrf_bufferSize((a),(b),(c),(rocblas_double_complex*)(d),(e),(f))
#define hipsolverXpotrs(a,b,c,d,e,f,g,h,i)      rocsolver_zpotrs((a),(b),(c),(d),(rocblas_double_complex*)(e),(f),(rocblas_double_complex*)(g),(h),(i))
/* #define hipsolverXpotri(a,b,c,d,e,f,g,h)        rocsolver_zpotri((a),(b),(c),(rocblas_double_complex*)(d),(e),(rocblas_double_complex*)(f),(g),(h)) */
/* #define hipsolverXpotri_bufferSize(a,b,c,d,e,f) rocsolver_zpotri_bufferSize((a),(b),(c),(rocblas_double_complex*)(d),(e),(f)) */
#define hipsolverXsytrf(a,b,c,d,e,f,g,h,i)      rocsolver_zsytrf((a),(b),(c),(rocblas_double_complex*)(d),(e),(f),(rocblas_double_complex*)(g),(h),(i))
#define hipsolverXsytrf_bufferSize(a,b,c,d,e)   rocsolver_zsytrf_bufferSize((a),(b),(rocblas_double_complex*)(c),(d),(e))
#define hipsolverXgetrf(a,b,c,d,e,f,g,h)        rocsolver_zgetrf((a),(b),(c),(rocblas_double_complex*)(d),(e),(rocblas_double_complex*)(f),(g),(h))
#define hipsolverXgetrf_bufferSize(a,b,c,d,e,f) rocsolver_zgetrf_bufferSize((a),(b),(c),(rocblas_double_complex*)(d),(e),(f))
#define hipsolverXgetrs(a,b,c,d,e,f,g,h,i,j)    rocsolver_zgetrs((a),(b),(c),(d),(rocblas_double_complex*)(e),(f),(g),(rocblas_double_complex*)(h),(i),(j))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define hipsolverXpotrf(a,b,c,d,e,f,g,h)        rocsolver_spotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverXpotrf_bufferSize(a,b,c,d,e,f) rocsolver_spotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverXpotrs(a,b,c,d,e,f,g,h,i)      rocsolver_spotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
/* #define hipsolverXpotri(a,b,c,d,e,f,g,h)        rocsolver_spotri((a),(b),(c),(d),(e),(f),(g),(h)) */
/* #define hipsolverXpotri_bufferSize(a,b,c,d,e,f) rocsolver_spotri_bufferSize((a),(b),(c),(d),(e),(f)) */
#define hipsolverXsytrf(a,b,c,d,e,f,g,h,i)      rocsolver_ssytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverXsytrf_bufferSize(a,b,c,d,e)   rocsolver_ssytrf_bufferSize((a),(b),(c),(d),(e))
#define hipsolverXgetrf(a,b,c,d,e,f,g,h)        rocsolver_sgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverXgetrf_bufferSize(a,b,c,d,e,f) rocsolver_sgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverXgetrs(a,b,c,d,e,f,g,h,i,j)    rocsolver_sgetrs((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
#else /* real double */
#define hipsolverXpotrf(a,b,c,d,e,f,g,h)        rocsolver_dpotrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverXpotrf_bufferSize(a,b,c,d,e,f) rocsolver_dpotrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverXpotrs(a,b,c,d,e,f,g,h,i)      rocsolver_dpotrs((a),(b),(c),(d),(e),(f),(g),(h),(i))
/* #define hipsolverXpotri(a,b,c,d,e,f,g,h)        rocsolver_dpotri((a),(b),(c),(d),(e),(f),(g),(h)) */
/* #define hipsolverXpotri_bufferSize(a,b,c,d,e,f) rocsolver_dpotri_bufferSize((a),(b),(c),(d),(e),(f)) */
#define hipsolverXsytrf(a,b,c,d,e,f,g,h,i)      rocsolver_dsytrf((a),(b),(c),(d),(e),(f),(g),(h),(i))
#define hipsolverXsytrf_bufferSize(a,b,c,d,e)   rocsolver_dsytrf_bufferSize((a),(b),(c),(d),(e))
#define hipsolverXgetrf(a,b,c,d,e,f,g,h)        rocsolver_dgetrf((a),(b),(c),(d),(e),(f),(g),(h))
#define hipsolverXgetrf_bufferSize(a,b,c,d,e,f) rocsolver_dgetrf_bufferSize((a),(b),(c),(d),(e),(f))
#define hipsolverXgetrs(a,b,c,d,e,f,g,h,j)      rocsolver_dgetrs((a),(b),(c),(d),(e),(f),(g),(h),(j)) /* Do not use i */
#endif
#endif

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool   user_alloc;
  PetscScalar *unplacedarray; /* if one called MatHIPDensePlaceArray(), this is where it stashed the original */
  PetscBool   unplaced_user_alloc;
  /* factorization support */
  int         *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_work; /* device workspace */
  int         fact_lwork;
  int         *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseHIP;

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPSetPreallocation(Mat A, PetscScalar *d_data)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;
  PetscBool        iship;
  hipError_t      cerr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSEHIP,&iship);CHKERRQ(ierr);
  if (!iship) PetscFunctionReturn(0);
  /* it may happen CPU preallocation has not been performed */
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (cA->lda <= 0) cA->lda = A->rmap->n;
  if (!dA->user_alloc) { cerr = hipFree(dA->d_v);CHKERRHIP(cerr); }
  if (!d_data) { /* petsc-allocated storage */
    ierr = PetscIntMultError(cA->lda,A->cmap->n,NULL);CHKERRQ(ierr);
    cerr = hipMalloc((void**)&dA->d_v,cA->lda*A->cmap->n*sizeof(PetscScalar));CHKERRHIP(cerr);
    dA->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    dA->d_v        = d_data;
    dA->user_alloc = PETSC_TRUE;
    A->offloadmask = PETSC_OFFLOAD_GPU;
  }
  A->preallocated = PETSC_TRUE;
  A->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPCopyFromGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;
  hipError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",A->offloadmask == PETSC_OFFLOAD_GPU ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!cA->v) { /* MatCreateSeqDenseHIP may not allocate CPU memory. Allocate if needed */
      ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
    }
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

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPCopyToGPU(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscBool        copy;
  PetscErrorCode   ierr;
  hipError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSEHIP);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      ierr = MatSeqDenseHIPSetPreallocation(A,NULL);CHKERRQ(ierr);
    }
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

PETSC_EXTERN PetscErrorCode MatCopy_SeqDenseHIP(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense*)B->data;
  PetscErrorCode    ierr;
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt          lda1=a->lda,lda2=b->lda, m=A->rmap->n,n=A->cmap->n, j;
  hipError_t       cerr;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (m != B->rmap->n || n != B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  ierr = MatDenseHIPGetArrayRead(A,&va);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayWrite(B,&vb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      cerr = hipMemcpy(vb+j*lda2,va+j*lda1,m*sizeof(PetscScalar),hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
    }
  } else {
    cerr = hipMemcpy(vb,va,m*(n*sizeof(PetscScalar)),hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
  }
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(B,&vb);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&va);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatDenseHIPPlaceArray_SeqDenseHIP(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (aa->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (aa->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseHIPResetArray() must be called first");
  if (aa->v) { ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr); }
  dA->unplacedarray = dA->d_v;
  dA->unplaced_user_alloc = dA->user_alloc;
  dA->d_v = (PetscScalar*)a;
  dA->user_alloc = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPResetArray_SeqDenseHIP(Mat A)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->v) { ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr); }
  dA->d_v = dA->unplacedarray;
  dA->user_alloc = dA->unplaced_user_alloc;
  dA->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPReplaceArray_SeqDenseHIP(Mat A, const PetscScalar *a)
{
  Mat_SeqDense     *aa = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  hipError_t       cerr;

  PetscFunctionBegin;
  if (aa->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (aa->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseHIPResetArray() must be called first");
  if (!dA->user_alloc) { cerr = hipFree(dA->d_v);CHKERRHIP(cerr); }
  dA->d_v = (PetscScalar*)a;
  dA->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArrayWrite_SeqDenseHIP(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!dA->d_v) {
    ierr = MatSeqDenseHIPSetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  *a = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArrayWrite_SeqDenseHIP(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArrayRead_SeqDenseHIP(Mat A, const PetscScalar **a)
{
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArrayRead_SeqDenseHIP(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPGetArray_SeqDenseHIP(Mat A, PetscScalar **a)
{
  Mat_SeqDenseHIP  *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr);
  *a   = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseHIPRestoreArray_SeqDenseHIP(Mat A, PetscScalar **a)
{
  PetscFunctionBegin;
  *a = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseHIPInvertFactors_Private(Mat A)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP    *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  PetscErrorCode     ierr;
  hipError_t         ccer;
  /* rocblas_status     cerr; */
  rocblas_handle     handle;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for HIP because of missing rocsolver functionality");

  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolvergetri not implemented");
  else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (!dA->d_fact_ipiv) { /* spd */
      size_t il;

      ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
      /* cerr = hipsolverXpotri_bufferSize(handle,rocsparse_fill_mode_lower,n,da,lda,&il);CHKERRHIPSOLVER(cerr); */
      rocblas_start_device_memory_size_query(handle);
      /* cerr = hipsolverXpotri(handle, rocsparse_fill_mode_lower, n, da, lda, nullptr, nullptr, nullptr); CHKERRHIPBLAS(cerr); */
      rocblas_stop_device_memory_size_query(handle, &il);


      if (il > dA->fact_lwork) {
        dA->fact_lwork = il;

        ccer = hipFree(dA->d_fact_work);CHKERRHIP(ccer);
        ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
      }
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      /* cerr = hipsolverXpotri(handle,rocsparse_fill_mode_lower,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRHIPSOLVER(cerr); */
      ccer = WaitForHIP();CHKERRHIP(ccer);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = MatDenseHIPRestoreArray(A,&da);CHKERRQ(ierr);
      /* TODO (write hip kernel) */
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDnsytri not implemented");
  }
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: leading minor of order %d is zero",info);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to rocSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  A->ops->solve          = NULL;
  A->ops->solvetranspose = NULL;
  A->ops->matsolve       = NULL;
  A->factortype          = MAT_FACTOR_NONE;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode MatMatSolve_SeqDenseHIP(Mat A,Mat B,Mat X)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense       *x = (Mat_SeqDense*)X->data;
  Mat_SeqDenseHIP    *dA = (Mat_SeqDenseHIP*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *dx;
  rocblas_handle     handle;
  PetscBool          iship;
  int                nrhs,n,lda,ldx;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipError_t        ccer;
  rocblas_status     cerr;
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
  ierr = PetscHIPSOLVERGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    cerr = hipsolverXgetrs(handle,T ? rocblas_operation_transpose : rocblas_operation_none,n,nrhs,da,lda,dA->d_fact_ipiv,dx,ldx,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program hit hipErrorNotReady (error 34) due to "device not ready" on HIP API call to hipEventQuery. */
      cerr = hipsolverDnXpotrs(handle,rocsparse_fill_mode_lower,n,nrhs,da,lda,dx,ldx,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolverDnsytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ccer = WaitForHIP();CHKERRHIP(ccer);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(X,&dx);CHKERRQ(ierr);
  if (!iship) {
    ierr = MatConvert(X,MATSEQDENSE,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
  if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
  else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to rocSolver %d",-info);
#endif
  ierr = PetscLogGpuFlops(nrhs*(2.0*n*n - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode MatSolve_SeqDenseHIP_Private(Mat A,Vec xx,Vec yy,PetscBool trans)
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP    *dA = (Mat_SeqDenseHIP*)A->spptr;
  const PetscScalar  *da;
  PetscScalar        *y;
  rocblas_handle     handle;
  int                one = 1,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  hipError_t        ccer;
  rocblas_status    cerr;
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
  /* cublas/cusolver uses const while rocblas/rocsolver uses non-const.  To keep the interfaces the same, just do a cast */
  PetscScalar *dda =(PetscScalar *)&da;
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscHIPSOLVERGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    ierr = PetscInfo2(A,"LU solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    /* WORKS cerr = rocsolver_dgetrs(handle,trans ? rocblas_operation_transpose : rocblas_operation_none,
                            n,one,dda,lda,dA->d_fact_ipiv,
                            y,*dA->d_fact_info);CHKERRHIPSOLVER(cerr); */
    cerr = hipsolverXgetrs(handle,trans ? rocblas_operation_transpose : rocblas_operation_none,n,one,dda,lda,dA->d_fact_ipiv,y,*dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    ierr = PetscInfo2(A,"Cholesky solve %d x %d on backend\n",n,n);CHKERRQ(ierr);
    if (!dA->d_fact_ipiv) { /* spd */
      /* ========= Program has problem */
      cerr = hipsolverXpotrs(handle,rocsparse_fill_mode_lower,n,one,da,lda,y,n,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipsolversytrs not implemented");
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown factor type %d",A->factortype);
  ccer = WaitForHIP();CHKERRHIP(ccer);
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
  Mat_SeqDenseHIP    *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  int                m,n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  rocblas_status     cerr;
  rocblas_handle     handle;
  hipError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERGetHandle(&handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"LU factor %d x %d on backend\n",m,n);CHKERRQ(ierr);
  if (!dA->d_fact_ipiv) {
    ccer = hipMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRHIP(ccer);
  }
  if (!dA->fact_lwork) {
    cerr = hipsolverXgetrf_bufferSize(handle,m,n,da,lda,&dA->fact_lwork);CHKERRHIPSOLVER(cerr);
    ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
  }
  if (!dA->d_fact_info) {
    ccer = hipMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRHIP(ccer);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = hipsolverXgetrf(handle,m,n,da,lda,dA->d_fact_work,dA->d_fact_ipiv,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
  ccer = WaitForHIP();CHKERRHIP(ccer);
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
  Mat_SeqDenseHIP    *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscScalar        *da;
  int                n,lda;
#if defined(PETSC_USE_DEBUG)
  int                info;
#endif
  rocblas_status     cerr;
  rocblas_handle     handle;
  hipError_t        ccer;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPSOLVERGetHandle(&handle);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(A->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"Cholesky factor %d x %d on backend\n",n,n);CHKERRQ(ierr);
  if (A->spd) {
    ierr = MatDenseHIPGetArray(A,&da);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(a->lda,&lda);CHKERRQ(ierr);
    if (!dA->fact_lwork) {
      cerr = hipsolverXpotrf_bufferSize(handle,rocsparse_fill_mode_lower,n,da,lda,&dA->fact_lwork);CHKERRHIPSOLVER(cerr);
      ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = hipMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRHIP(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = hipsolverXpotrf(handle,rocsparse_fill_mode_lower,n,da,lda,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    ccer = WaitForHIP();CHKERRHIP(ccer);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatDenseHIPRestoreArray(A,&da);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    ccer = hipMemcpy(&info, dA->d_fact_info, sizeof(int), hipMemcpyDeviceToHost);CHKERRHIP(ccer);
    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to hipSolver %d",-info);
#endif
    A->factortype = MAT_FACTOR_CHOLESKY;
    ierr = PetscLogGpuFlops(1.0*n*n*n/3.0);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"hipsolversytrs unavailable. Use MAT_FACTOR_LU");
#if 0
    /* at the time of writing this interface , hipsolverDn does not implement *sytrs and *hetr* routines
       The code below should work, and it can be activated when *sytrs routines will be available */o
    if (!dA->d_fact_ipiv) {
      ccer = hipMalloc((void**)&dA->d_fact_ipiv,n*sizeof(*dA->d_fact_ipiv));CHKERRHIP(ccer);
    }
    if (!dA->fact_lwork) {
      cerr = hipsolverXsytrf_bufferSize(handle,n,da,lda,&dA->fact_lwork);CHKERRHIPSOLVER(cerr);
      ccer = hipMalloc((void**)&dA->d_fact_work,dA->fact_lwork*sizeof(*dA->d_fact_work));CHKERRHIP(ccer);
    }
    if (!dA->d_fact_info) {
      ccer = hipMalloc((void**)&dA->d_fact_info,sizeof(*dA->d_fact_info));CHKERRHIP(ccer);
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = hipsolverXsytrf(handle,rocsparse_fill_mode_lower,n,da,lda,dA->d_fact_ipiv,dA->d_fact_work,dA->fact_lwork,dA->d_fact_info);CHKERRHIPSOLVER(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif

  A->ops->solve          = MatSolve_SeqDenseHIP;
  A->ops->solvetranspose = MatSolveTranspose_SeqDenseHIP;
  A->ops->matsolve       = MatMatSolve_SeqDenseHIP;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERHIP,&A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* GEMM kernel: C = op(A)*op(B), tA, tB flag transposition */
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(Mat A,Mat B,Mat C,PetscBool tA,PetscBool tB)
{
  const PetscScalar *da,*db;
  PetscScalar       *dc;
  PetscScalar       one=1.0,zero=0.0;
  int               m,n,k;
  PetscInt          alda,blda,clda;
  PetscErrorCode    ierr;
  rocblas_handle    hipblasv2handle;
  PetscBool         Aiship,Biship;
  rocblas_status    berr;
  hipError_t       cerr;

  PetscFunctionBegin;
  /* we may end up with SEQDENSE as one of the arguments */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSEHIP,&Aiship);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSEHIP,&Biship);CHKERRQ(ierr);
  if (!Aiship) {
    ierr = MatConvert(A,MATSEQDENSEHIP,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  if (!Biship) {
    ierr = MatConvert(B,MATSEQDENSEHIP,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
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
  ierr = MatDenseGetLDA(A,&alda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = hipblasXgemm(hipblasv2handle,
                     tA ? rocblas_operation_transpose : rocblas_operation_none,
                     tB ? rocblas_operation_transpose : rocblas_operation_none,
                     m,n,k,&one,da,alda,db,blda,&zero,dc,clda);CHKERRHIPBLAS(berr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(B,&db);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayWrite(C,&dc);CHKERRQ(ierr);
  if (!Aiship) {
    ierr = MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  if (!Biship) {
    ierr = MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
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

PetscErrorCode MatProductSetFromOptions_SeqDenseHIP(Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatProductSetFromOptions_SeqDense(C);CHKERRQ(ierr);
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
  rocblas_handle     hipblasv2handle;
  rocblas_status    berr;
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
  ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = hipblasXgemv(hipblasv2handle,trans ? rocblas_operation_transpose : rocblas_operation_none,
                     m,n,&one,da,lda,xarray,1,(yy ? &one : &zero),zarray,1);CHKERRHIPBLAS(berr);
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

static PetscErrorCode MatDenseGetArrayRead_SeqDenseHIP(Mat A,const PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArrayWrite_SeqDenseHIP(Mat A,PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mat->v) { /* MatCreateSeqDenseHIP may not allocate CPU memory. Allocate if needed */
    ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDenseHIP(Mat A,PetscScalar **array)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
  *array = mat->v;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDenseHIP(Mat Y,PetscScalar alpha)
{
  Mat_SeqDense   *y = (Mat_SeqDense*)Y->data;
  PetscScalar    *dy;
  int            j,N,m,lday,one = 1;
  rocblas_handle hipblasv2handle;
  rocblas_status berr;
  PetscErrorCode ierr;
  hipError_t     cerr;

  PetscFunctionBegin;
  ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
  ierr = MatDenseHIPGetArray(Y,&dy);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(Y->rmap->n*Y->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(Y->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscMPIIntCast(y->lda,&lday);CHKERRQ(ierr);
  ierr = PetscInfo2(Y,"Performing Scale %d x %d on backend\n",Y->rmap->n,Y->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (lday>m) {
    for (j=0; j<Y->cmap->n; j++) {
      berr = hipblasXscal(hipblasv2handle,m,&alpha,dy+lday*j,one);CHKERRHIPBLAS(berr);
    }
  } else {
    berr = hipblasXscal(hipblasv2handle,N,&alpha,dy,one);CHKERRHIPBLAS(berr);
  }
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(N);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(Y,&dy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDenseHIP(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data;
  Mat_SeqDense      *y = (Mat_SeqDense*)Y->data;
  const PetscScalar *dx;
  PetscScalar       *dy;
  int               j,N,m,ldax,lday,one = 1;
  rocblas_handle    hipblasv2handle;
  rocblas_status    berr;
  PetscErrorCode    ierr;
  hipError_t        cerr;

  PetscFunctionBegin;
  if (!X->rmap->n || !X->cmap->n) PetscFunctionReturn(0);
  ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
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
      berr = hipblasXaxpy(hipblasv2handle,m,&alpha,dx+j*ldax,one,dy+j*lday,one);CHKERRHIPBLAS(berr);
    }
  } else {
    berr = hipblasXaxpy(hipblasv2handle,N,&alpha,dx,one,dy,one);CHKERRHIPBLAS(berr);
  }
  cerr = WaitForHIP();CHKERRHIP(cerr);
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
    if (dA->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"MatDenseHIPResetArray() must be called first");
    if (!dA->user_alloc) { cerr = hipFree(dA->d_v);CHKERRHIP(cerr); }
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
    PetscInt          ldb;

    ierr = MatDenseHIPGetArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseHIPGetArrayWrite(*B,&db);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(*B,&ldb);CHKERRQ(ierr);
    if (a->lda > A->rmap->n || ldb > A->rmap->n) {
      PetscInt j,m = A->rmap->n;

      for (j=0; j<A->cmap->n; j++) { /* it can be done better */
        cerr = hipMemcpy(db+j*ldb,da+j*a->lda,m*sizeof(PetscScalar),hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
      }
    } else {
      cerr = hipMemcpy(db,da,(sizeof(PetscScalar)*A->cmap->n)*A->rmap->n,hipMemcpyDeviceToDevice);CHKERRHIP(cerr);
    }
    ierr = MatDenseHIPRestoreArrayRead(A,&da);CHKERRQ(ierr);
    ierr = MatDenseHIPRestoreArrayWrite(*B,&db);CHKERRQ(ierr);
    (*B)->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

#include <petsc/private/vecimpl.h>

static PetscErrorCode MatGetColumnVector_SeqDenseHIP(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;
  PetscScalar      *x;
  PetscBool        viship;
  hipError_t      cerr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&viship,VECSEQHIP,VECMPIHIP,VECHIP,"");CHKERRQ(ierr);
  if (viship && !v->boundtocpu) { /* update device data */
    ierr = VecHIPGetArrayWrite(v,&x);CHKERRQ(ierr);
    if (A->offloadmask & PETSC_OFFLOAD_GPU) {
      cerr = hipMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),hipMemcpyHostToHost);CHKERRHIP(cerr);
    } else {
      cerr = hipMemcpy(x,a->v + col*a->lda,A->rmap->n*sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
    }
    ierr = VecHIPRestoreArrayWrite(v,&x);CHKERRQ(ierr);
  } else { /* update host data */
    ierr = VecGetArrayWrite(v,&x);CHKERRQ(ierr);
    if (A->offloadmask & PETSC_OFFLOAD_CPU) {
      ierr = PetscArraycpy(x,a->v+col*a->lda,A->rmap->n);CHKERRQ(ierr);
    } else {
      cerr = hipMemcpy(x,dA->d_v + col*a->lda,A->rmap->n*sizeof(PetscScalar),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    }
    ierr = VecRestoreArrayWrite(v,&x);CHKERRQ(ierr);
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
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERHIP,&(*fact)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVec_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseHIPGetArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    ierr = VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecHIPPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecHIPResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseHIPGetArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    ierr = VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecHIPPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  ierr = VecLockReadPush(a->cvec);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecLockReadPop(a->cvec);CHKERRQ(ierr);
  ierr = VecHIPResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetColumnVecWrite_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = MatDenseHIPGetArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  if (!a->cvec) { /* we pass the data of A, to prevent allocating needless GPU memory the first time VecHIPPlaceArray is called */
    ierr = VecCreateSeqHIPWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,a->ptrinuse,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = VecHIPPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDenseHIP(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = VecHIPResetArray(a->cvec);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseHIP(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_SeqDense     *a = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && cend-cbegin != a->cmat->cmap->N) {
    ierr = MatDestroy(&a->cmat);CHKERRQ(ierr);
  }
  ierr = MatSeqDenseHIPCopyToGPU(A);CHKERRQ(ierr);
  if (!a->cmat) {
    ierr = MatCreateDenseHIP(PetscObjectComm((PetscObject)A),A->rmap->n,PETSC_DECIDE,A->rmap->N,cend-cbegin,dA->d_v + (size_t)cbegin * (size_t)a->lda,&a->cmat);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat);CHKERRQ(ierr);
  } else {
    ierr = MatDenseHIPPlaceArray(a->cmat,dA->d_v + (size_t)cbegin * (size_t)a->lda);CHKERRQ(ierr);
  }
  ierr = MatDenseSetLDA(a->cmat,a->lda);CHKERRQ(ierr);
  if (a->v) { ierr = MatDensePlaceArray(a->cmat,a->v + (size_t)cbegin * (size_t)a->lda);CHKERRQ(ierr); }
  a->cmat->offloadmask = A->offloadmask;
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseHIP(Mat A,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  if (!a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column matrix");
  if (*v != a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatDenseHIPResetArray(a->cmat);CHKERRQ(ierr);
  ierr = MatDenseResetArray(a->cmat);CHKERRQ(ierr);
  *v = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatDenseSetLDA_SeqDenseHIP(Mat A,PetscInt lda)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseHIP *dA = (Mat_SeqDenseHIP*)A->spptr;
  PetscBool        data;

  PetscFunctionBegin;
  data = (PetscBool)((A->rmap->n > 0 && A->cmap->n > 0) ? (dA->d_v ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE);
  if (!dA->user_alloc && data && cA->lda!=lda) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"LDA cannot be changed after allocation of internal storage");
  if (lda < A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %D must be at least matrix dimension %D",lda,A->rmap->n);
  cA->lda = lda;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqDenseHIP(Mat A,PetscBool flg)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = flg;
  if (!flg) {
    PetscBool iship;

    ierr = PetscObjectTypeCompare((PetscObject)a->cvec,VECSEQHIP,&iship);CHKERRQ(ierr);
    if (!iship) {
      ierr = VecDestroy(&a->cvec);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompare((PetscObject)a->cmat,MATSEQDENSEHIP);CHKERRQ(ierr);
    if (!iship) {
      ierr = MatDestroy(&a->cmat);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArrayRead_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArrayWrite_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDenseHIP);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDenseHIP);CHKERRQ(ierr);

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
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDenseHIP;
    A->ops->getcolumnvector         = MatGetColumnVector_SeqDenseHIP;
    A->ops->scale                   = MatScale_SeqDenseHIP;
    A->ops->copy                    = MatCopy_SeqDenseHIP;
  } else {
    /* make sure we have an up-to-date copy on the CPU */
    ierr = MatSeqDenseHIPCopyFromGPU(A);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArray_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayRead_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetArrayWrite_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDense);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDense);CHKERRQ(ierr);

    A->ops->duplicate               = MatDuplicate_SeqDense;
    A->ops->mult                    = MatMult_SeqDense;
    A->ops->multadd                 = MatMultAdd_SeqDense;
    A->ops->multtranspose           = MatMultTranspose_SeqDense;
    A->ops->multtransposeadd        = MatMultTransposeAdd_SeqDense;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDense;
    A->ops->matmultnumeric          = MatMatMultNumeric_SeqDense_SeqDense;
    A->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqDense_SeqDense;
    A->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqDense_SeqDense;
    A->ops->axpy                    = MatAXPY_SeqDense;
    A->ops->choleskyfactor          = MatCholeskyFactor_SeqDense;
    A->ops->lufactor                = MatLUFactor_SeqDense;
    A->ops->productsetfromoptions   = MatProductSetFromOptions_SeqDense;
    A->ops->getcolumnvector         = MatGetColumnVector_SeqDense;
    A->ops->scale                   = MatScale_SeqDense;
    A->ops->copy                    = MatCopy_SeqDense;
  }
  if (a->cmat) {
    ierr = MatBindToCPU(a->cmat,flg);CHKERRQ(ierr);
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
  ierr = MatBindToCPU_SeqDenseHIP(B,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatReset_SeqDenseHIP(B);CHKERRQ(ierr);
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensehip_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPPlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPReplaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensehip_C",NULL);CHKERRQ(ierr);

  B->ops->bindtocpu = NULL;
  B->ops->destroy = MatDestroy_SeqDense;
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
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdensehip_seqdense_C",            MatConvert_SeqDenseHIP_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArray_C",                        MatDenseHIPGetArray_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArrayRead_C",                    MatDenseHIPGetArrayRead_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPGetArrayWrite_C",                   MatDenseHIPGetArrayWrite_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArray_C",                    MatDenseHIPRestoreArray_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArrayRead_C",                MatDenseHIPRestoreArrayRead_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPRestoreArrayWrite_C",               MatDenseHIPRestoreArrayWrite_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPPlaceArray_C",                      MatDenseHIPPlaceArray_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPResetArray_C",                      MatDenseHIPResetArray_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseHIPReplaceArray_C",                    MatDenseHIPReplaceArray_SeqDenseHIP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdensehip_C",MatProductSetFromOptions_SeqAIJ_SeqDense);CHKERRQ(ierr);

  ierr     = PetscNewLog(B,&dB);CHKERRQ(ierr);
  B->spptr = dB;

  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

  ierr = MatBindToCPU_SeqDenseHIP(B,PETSC_FALSE);CHKERRQ(ierr);
  B->ops->bindtocpu = MatBindToCPU_SeqDenseHIP;
  B->ops->destroy  = MatDestroy_SeqDenseHIP;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqDenseHIP - Creates a sequential matrix in dense format using HIP.

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

.seealso: MatCreate(), MatCreateSeqDense()
@*/
PetscErrorCode  MatCreateSeqDenseHIP(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"Invalid communicator size %d",size);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSEHIP);CHKERRQ(ierr);
  ierr = MatSeqDenseHIPSetPreallocation(*A,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSEHIP - MATSEQDENSEHIP = "seqdensehip" - A matrix type to be used for sequential dense matrices on GPUs.

   Options Database Keys:
. -mat_type seqdensehip - sets the matrix type to "seqdensehip" during a call to MatSetFromOptions()

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseHIP(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqDense(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqDense_SeqDenseHIP(B,MATSEQDENSEHIP,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
