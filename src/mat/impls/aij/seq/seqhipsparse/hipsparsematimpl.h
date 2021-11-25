#if !defined(__HIPSPARSEMATIMPL)
#define __HIPSPARSEMATIMPL

/*
 * Note on hip versus roc:
 * For the basic hip language itself, we used the hip label in keeping with the
 * recommended porting guide and to enable testing with a cuda back end.
 * However, with the rocblas/rocsparse, the hip layers do not do a good job of
 * maintaining the compatibility so we abandon their hip layers and go straigt
 * to the roc layers.  This is recommended by conversations with AMD developers
 * themselves.  This means that we should enable vec_type=hip and
 * HIP_PLATFORM=cuda and use cusparse.  Permuations like this will be deferred
 * down the road.
 *
 * We do continue using the hip label at the PETSc level; i.e., internally we
 * use the hip moniker even if the underlying call is out to roc
 *
 */
#include <petscpkg_version.h>
#include <petsc/private/hipvecimpl.h>

#include <rocsparse.h>
/* csrsv2Info_t is defined in hipsparse and not rocsparse so we'll have some
 * work to re-implement it */
/* #include <hipsparse.h> */

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/system/system_error.h>

#define CHKERRHIPSPARSE(stat) do {if (PetscUnlikely(stat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"rocsparse error %d",(int)stat);} while (0)

#define PetscStackCallThrust(body) do {                                     \
    try {                                                                   \
      body;                                                                 \
    } catch(thrust::system_error& e) {                                      \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Thrust %s",e.what());\
    }                                                                       \
  } while (0)

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    const hipComplex PETSC_HIPSPARSE_ONE        = {1.0f, 0.0f};
    const hipComplex PETSC_HIPSPARSE_ZERO       = {0.0f, 0.0f};
  #elif defined(PETSC_USE_REAL_DOUBLE)
    const hipDoubleComplex PETSC_HIPSPARSE_ONE  = {1.0, 0.0};
    const hipDoubleComplex PETSC_HIPSPARSE_ZERO = {0.0, 0.0};
  #endif
#else
  const PetscScalar PETSC_HIPSPARSE_ONE        = 1.0;
  const PetscScalar PETSC_HIPSPARSE_ZERO       = 0.0;
#endif

/* TODO
 * Namespace:  prsparse = PETSc's rocsparse interface
 * PETSc cusparse interface uses cusparse for namespacing.  This is perhaps OK
 * because cusparse uses camelCase and PETSc's wrapper uses snake_case so one
 * can distinguish the two by looking at them.  rocsparse uses camel case so
 * need a better way of namespacing our own architecture
 */
#define prsparse_create_analysis_info  rocsparse_create_mat_info((rocsparse_mat_info*)info)
#define prsparse_destroy_analysis_info rocsparse_destroy_mat_info((rocsparse_mat_info)info)
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define prsparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) rocsparse_ccsrsv_buffer_size(a,b,c,d,e,(rocsparse_float_complex*)(f),g,h,i,j)
     /* TODO:  The cusparse version of analysis is blocking so need to check if I have to worry about that */
    #define prsparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     rocsparse_ccsrsv_analysis(a,b,c,d,e,(const rocsparse_float_complex*)(f),g,h,i,j,k)
    #define prsparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  rocsparse_ccsrsv_solve(a,b,c,d,(const rocsparse_float_complex*)(e),f,(const rocsparse_float_complex*)(g),h,i,j,(const rocsparse_float_complex*)(k),(rocsparse_float_complex*)(l),m,n)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define prsparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) rocsparse_zcsrsv_buffer_size(a,b,c,d,e,(rocsparse_double_complex*)(f),g,h,i,j)
    #define prsparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     rocsparse_zcsrsv_analysis(a,b,c,d,e,(const rocsparse_double_complex*)(f),g,h,i,j,k)
    #define prsparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  rocsparse_zcsrsv_solve(a,b,c,d,(const rocsparse_double_complex*)(e),f,(const rocsparse_double_complex*)(g),h,i,j,(const rocsparse_double_complex*)(k),(rocsparse_double_complex*)(l),m,n)
  #endif
#else /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define prsparse_get_svbuffsize rocsparse_scsrsv_buffer_size
    #define prsparse_analysis       rocsparse_scsrsv_analysis
    #define prsparse_solve          rocsparse_scsrsv_solve
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define prsparse_get_svbuffsize rocsparse_scsrsv_buffer_size
    #define prsparse_analysis       rocsparse_dcsrsv_analysis
    #define prsparse_solve          rocsparse_dcsrsv_solve
  #endif
#endif
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define prsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       rocsparse_ccsrmv((a),(b),(c),(d),(e),(rocsparse_float_complex*)(f),(g),(rocsparse_float_complex*)(h),(i),(j),(rocsparse_float_complex*)(k),(rocsparse_float_complex*)(l),(rocsparse_float_complex*)(m))
    #define prsparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) rocsparse_ccsrmm((a),(b),(c),(d),(e),(f),(rocspare_float_complex*)(g),(h),(rocspare_float_complex*)(i),(j),(k),(rocspare_float_complex*)(l),(m),(rocspare_float_complex*)(n),(rocspare_float_complex*)(o),(p))
    #define prsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          hipsparseCcsr2csc((a),(b),(c),(d),(rocspare_float_complex*)(e),(f),(g),(rocspare_float_complex*)(h),(i),(j),(k),(l))
    #define prsparse_hyb_spmv(a,b,c,d,e,f,g,h)                 hipsparseChybmv((a),(b),(rocspare_float_complex*)(c),(d),(e),(rocspare_float_complex*)(f),(rocspare_float_complex*)(g),(rocspare_float_complex*)(h))
    #define prsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              hipsparseCcsr2hyb((a),(b),(c),(d),(rocspare_float_complex*)(e),(f),(g),(h),(i),(j))
    #define prsparse_hyb2csr(a,b,c,d,e,f)                      rocsparseChyb2csr((a),(b),(c),(rocspare_float_complex*)(d),(e),(f))
    #define prsparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) hipsparseCcsrgemm(a,b,c,d,e,f,g,h,(rocspare_float_complex*)i,j,k,l,m,(rocspare_float_complex*)n,o,p,q,(rocspare_float_complex*)r,s,t)
    #define prsparse_csr_spgeam(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s)   hipsparseCcsrgeam(a,b,c,(rocspare_float_complex*)d,e,f,(rocspare_float_complex*)g,h,i,(rocspare_float_complex*)j,k,l,(rocspare_float_complex*)m,n,o,p,(rocspare_float_complex*)q,r,s)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define prsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       hipsparseZcsrmv((a),(b),(c),(d),(e),(rocsparse_double_complex*)(f),(g),(rocsparse_double_complex*)(h),(i),(j),(rocsparse_double_complex*)(k),(rocsparse_double_complex*)(l),(rocsparse_double_complex*)(m))
    #define prsparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) hipsparseZcsrmm((a),(b),(c),(d),(e),(f),(rocsparse_double_complex*)(g),(h),(rocsparse_double_complex*)(i),(j),(k),(rocsparse_double_complex*)(l),(m),(rocsparse_double_complex*)(n),(rocsparse_double_complex*)(o),(p))
    #define prsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          hipsparseZcsr2csc((a),(b),(c),(d),(rocsparse_double_complex*)(e),(f),(g),(rocsparse_double_complex*)(h),(i),(j),(k),(l))
    #define prsparse_hyb_spmv(a,b,c,d,e,f,g,h)                 hipsparseZhybmv((a),(b),(rocsparse_double_complex*)(c),(d),(e),(rocsparse_double_complex*)(f),(rocsparse_double_complex*)(g),(rocsparse_double_complex*)(h))
    #define prsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              hipsparseZcsr2hyb((a),(b),(c),(d),(rocsparse_double_complex*)(e),(f),(g),(h),(i),(j))
    #define prsparse_hyb2csr(a,b,c,d,e,f)                      rocsparseZhyb2csr((a),(b),(c),(rocsparse_double_complex*)(d),(e),(f))
    #define prsparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) hipsparseZcsrgemm(a,b,c,d,e,f,g,h,(rocsparse_double_complex*)i,j,k,l,m,(rocsparse_double_complex*)n,o,p,q,(rocsparse_double_complex*)r,s,t)
    #define prsparse_csr_spgeam(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s)   hipsparseZcsrgeam(a,b,c,(rocsparse_double_complex*)d,e,f,(rocsparse_double_complex*)g,h,i,(rocsparse_double_complex*)j,k,l,(rocsparse_double_complex*)m,n,o,p,(rocsparse_double_complex*)q,r,s)
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define prsparse_csr_spmv rocsparse_scsrmv
    #define prsparse_csr_spmm rocsparse_scsrmm
    #define prsparse_csr2csc  rocsparse_scsr2csc
    #define prsparse_hyb_spmv rocsparse_shybmv
    #define prsparse_csr2hyb  rocsparse_scsr2hyb
    #define prsparse_hyb2csr  rocsparse_shyb2csr
    #define prsparse_csr_spgemm rocsparse_scsrgemm
    #define prsparse_csr_spgeam rocsparse_scsrgeam
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define prsparse_csr_spmv rocsparse_dcsrmv
    #define prsparse_csr_spmm rocsparse_dcsrmm
    #define prsparse_csr2csc  rocsparse_dcsr2csc
    #define prsparse_hyb_spmv rocsparse_dhybmv
    #define prsparse_csr2hyb  rocsparse_dcsr2hyb
    #define prsparse_hyb2csr  rocsparse_dhyb2csr
    #define prsparse_csr_spgemm rocsparse_dcsrgemm
    #define prsparse_csr_spgeam rocsparse_dcsrgeam
  #endif
#endif

#define THRUSTINTARRAY32 thrust::device_vector<int>
#define THRUSTINTARRAY thrust::device_vector<PetscInt>
#define THRUSTARRAY thrust::device_vector<PetscScalar>

/* A CSR matrix structure */
struct CsrMatrix {
  PetscInt         num_rows;
  PetscInt         num_cols;
  PetscInt         num_entries;
  THRUSTINTARRAY32 *row_offsets;
  THRUSTINTARRAY32 *column_indices;
  THRUSTARRAY      *values;
};

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJHIPSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  rocsparse_mat_descr          descr;
  rocsparse_operation          solveOp;
  CsrMatrix                   *csrMat;
  rocsparse_mat_info          solveInfo;
  rocsparse_solve_policy      solvePolicy;     /* whether level information is generated and used */
  int                         solveBufferSize;
  void                        *solveBuffer;
  size_t                      csr2cscBufferSize; /* to transpose the triangular factor (only used for CUDA >= 11.0) */
  void                        *csr2cscBuffer;
  PetscScalar                 *AA_h; /* managed host buffer for moving values to the GPU */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and any indices used in a reordering */
struct Mat_SeqAIJHIPSPARSETriFactors {
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtrTranspose; /* pointer for lower triangular (factored matrix) on GPU for the transpose (useful for BiCG) */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtrTranspose; /* pointer for upper triangular (factored matrix) on GPU for the transpose (useful for BiCG)*/
  THRUSTINTARRAY                    *rpermIndices;  /* indices used for any reordering */
  THRUSTINTARRAY                    *cpermIndices;  /* indices used for any reordering */
  THRUSTARRAY                       *workVector;
  rocsparse_handle                  handle;   /* a handle to the rocsparse library */
  PetscInt                          nnz;      /* number of nonzeros ... need this for accurate logging between ICC and ILU */
  PetscScalar                       *a_band_d; /* GPU data for banded CSR LU factorization matrix diag(L)=1 */
  int                               *i_band_d; /* this could be optimized away */
  cudaDeviceProp                    dev_prop;
  PetscBool                         init_dev_prop;
};

struct Mat_rocsparseSpMV {
  PetscBool             initialized;    /* Don't rely on spmvBuffer != NULL to test if the struct is initialized, */
  size_t                spmvBufferSize; /* since I'm not sure if smvBuffer can be NULL even after rocsparseSpMV_bufferSize() */
  void                  *spmvBuffer;
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJHIPSPARSEMultStruct {
  void               *mat;  /* opaque pointer to a matrix. This could be either a hipsparseHybMat_t or a CsrMatrix */
  rocsparse_mat_descr descr; /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY     *cprowIndices;   /* compressed row indices used in the parallel SpMV */
  PetscScalar        *alpha_one; /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar        *beta_zero; /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar        *beta_one; /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Tranpose */
struct Mat_SeqAIJHIPSPARSE {
  Mat_SeqAIJHIPSPARSEMultStruct *mat;            /* pointer to the matrix on the GPU */
  Mat_SeqAIJHIPSPARSEMultStruct *matTranspose;   /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                  *workVector;     /* pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  THRUSTINTARRAY32             *rowoffsets_gpu; /* rowoffsets on GPU in non-compressed-row format. It is used to convert CSR to CSC */
  PetscInt                     nrows;           /* number of rows of the matrix seen by GPU */
  MatHIPSPARSEStorageFormat    format;          /* the storage format for the matrix on the device */
  hipStream_t                  stream;          /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  rocsparse_handle             handle;          /* a handle to the rocsparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState             nonzerostate;    /* track nonzero state to possibly recreate the GPU matrix */
  THRUSTINTARRAY               *csr2csc_i;
  PetscSplitCSRDataStructure   deviceMat;       /* Matrix on device for, eg, assembly */
  THRUSTINTARRAY               *cooPerm;        /* permutation array that sorts the input coo entris by row and col */
  THRUSTINTARRAY               *cooPerm_a;      /* ordered array that indicate i-th nonzero (after sorting) is the j-th unique nonzero */
};

PETSC_INTERN PetscErrorCode MatHIPSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetStream(Mat, const hipStream_t stream);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetHandle(Mat, const rocsparse_handle handle);
PETSC_INTERN PetscErrorCode MatHIPSPARSEClearHandle(Mat);
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat,const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEMergeMats(Mat,Mat,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Reset(Mat_SeqAIJHIPSPARSETriFactors_p*);

PETSC_STATIC_INLINE bool isHipMem(const void *data)
{
  hipError_t                  cerr;
  struct hipPointerAttribute_t attr;
  enum hipMemoryType          mtype;
  cerr = hipPointerGetAttributes(&attr,data);
  /* This gives warning:  hipGetLastError(); /* Reset the last error */ 
  mtype = attr.memoryType;
  if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) return true;
  else return false;
}

#endif
