#if !defined(__HIPSPARSEMATIMPL)
#define __HIPSPARSEMATIMPL

#include <../src/vec/vec/impls/seq/seqhip/hipvecimpl.h>

#include <hipsparse.h>

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#define CHKERRHIPSPARSE(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name  = hipsparseGetErrorName(stat); \
      const char *descr = hipsparseGetErrorString(stat); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipSPARSE error %d (%s) : %s",(int)stat,name,descr); \
   } \
} while(0)

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)  
#define hipsparse_solve(a,b,c,d,e,f,g,h,i,j,k)        hipsparseCcsrsv_solve((a),(b),(c),(hipComplex*)(d),(e),(hipComplex*)(f),(g),(h),(i),(hipComplex*)(j),(hipComplex*)(k))
#define hipsparse_analysis(a,b,c,d,e,f,g,h,i)         hipsparseCcsrsv_analysis((a),(b),(c),(d),(e),(hipComplex*)(f),(g),(h),(i))
#define hipsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m) hipsparseCcsrmv((a),(b),(c),(d),(e),(hipComplex*)(f),(g),(hipComplex*)(h),(i),(j),(hipComplex*)(k),(hipComplex*)(l),(hipComplex*)(m))
#define hipsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)    hipsparseCcsr2csc((a),(b),(c),(d),(hipComplex*)(e),(f),(g),(hipComplex*)(h),(i),(j),(k),(l))
#define hipsparse_hyb_spmv(a,b,c,d,e,f,g,h)           hipsparseChybmv((a),(b),(hipComplex*)(c),(d),(e),(hipComplex*)(f),(hipComplex*)(g),(hipComplex*)(h))
#define hipsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)        hipsparseCcsr2hyb((a),(b),(c),(d),(hipComplex*)(e),(f),(g),(h),(i),(j))
#define hipsparse_hyb2csr(a,b,c,d,e,f)                hipsparseChyb2csr((a),(b),(c),(hipComplex*)(d),(e),(f))
const hipFloatComplex PETSC_HIPSPARSE_ONE  = {1.0f, 0.0f};
const hipFloatComplex PETSC_HIPSPARSE_ZERO = {0.0f, 0.0f};
#elif defined(PETSC_USE_REAL_DOUBLE)
#define hipsparse_solve(a,b,c,d,e,f,g,h,i,j,k)        hipsparseZcsrsv_solve((a),(b),(c),(hipDoubleComplex*)(d),(e),(hipDoubleComplex*)(f),(g),(h),(i),(hipDoubleComplex*)(j),(hipDoubleComplex*)(k))
#define hipsparse_analysis(a,b,c,d,e,f,g,h,i)         hipsparseZcsrsv_analysis((a),(b),(c),(d),(e),(hipDoubleComplex*)(f),(g),(h),(i))
#define hipsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m) hipsparseZcsrmv((a),(b),(c),(d),(e),(hipDoubleComplex*)(f),(g),(hipDoubleComplex*)(h),(i),(j),(hipDoubleComplex*)(k),(hipDoubleComplex*)(l),(hipDoubleComplex*)(m))
#define hipsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)    hipsparseZcsr2csc((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(g),(hipDoubleComplex*)(h),(i),(j),(k),(l))
#define hipsparse_hyb_spmv(a,b,c,d,e,f,g,h)           hipsparseZhybmv((a),(b),(hipDoubleComplex*)(c),(d),(e),(hipDoubleComplex*)(f),(hipDoubleComplex*)(g),(hipDoubleComplex*)(h))
#define hipsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)        hipsparseZcsr2hyb((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(g),(h),(i),(j))
#define hipsparse_hyb2csr(a,b,c,d,e,f)                hipsparseZhyb2csr((a),(b),(c),(hipDoubleComplex*)(d),(e),(f))
const hipDoubleComplex PETSC_HIPSPARSE_ONE  = {1.0, 0.0};
const hipDoubleComplex PETSC_HIPSPARSE_ZERO = {0.0, 0.0};
#endif
#else
const PetscScalar PETSC_HIPSPARSE_ONE  = 1.0;
const PetscScalar PETSC_HIPSPARSE_ZERO = 0.0;
#if defined(PETSC_USE_REAL_SINGLE)  
#define hipsparse_solve    hipsparseScsrsv_solve
#define hipsparse_analysis hipsparseScsrsv_analysis
#define hipsparse_csr_spmv hipsparseScsrmv
#define hipsparse_csr2csc  hipsparseScsr2csc
#define hipsparse_hyb_spmv hipsparseShybmv
#define hipsparse_csr2hyb  hipsparseScsr2hyb
#define hipsparse_hyb2csr  hipsparseShyb2csr
#elif defined(PETSC_USE_REAL_DOUBLE)
#define hipsparse_solve    hipsparseDcsrsv_solve
#define hipsparse_analysis hipsparseDcsrsv_analysis
#define hipsparse_csr_spmv hipsparseDcsrmv
#define hipsparse_csr2csc  hipsparseDcsr2csc
#define hipsparse_hyb_spmv hipsparseDhybmv
#define hipsparse_csr2hyb  hipsparseDcsr2hyb
#define hipsparse_hyb2csr  hipsparseDhyb2csr
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

//#define HIPSPMATRIXCSR32 hip::csr_matrix<int,PetscScalar,hip::device_memory>

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJHIPSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  hipsparseMatDescr_t          descr;
  hipsparseSolveAnalysisInfo_t solveInfo;
  hipsparseOperation_t         solveOp;
  CsrMatrix                   *csrMat; 
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJHIPSPARSEMultStruct {
  void               *mat;  /* opaque pointer to a matrix. This could be either a hipsparseHybMat_t or a CsrMatrix */
  hipsparseMatDescr_t descr; /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY     *cprowIndices;   /* compressed row indices used in the parallel SpMV */
  PetscScalar        *alpha; /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar        *beta_zero; /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar        *beta_one; /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and
 any indices used in a reordering */
struct Mat_SeqAIJHIPSPARSETriFactors {
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtrTranspose; /* pointer for lower triangular (factored matrix) on GPU for the transpose (useful for BiCG) */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtrTranspose; /* pointer for upper triangular (factored matrix) on GPU for the transpose (useful for BiCG)*/
  THRUSTINTARRAY                    *rpermIndices;  /* indices used for any reordering */
  THRUSTINTARRAY                    *cpermIndices;  /* indices used for any reordering */
  THRUSTARRAY                       *workVector;
  hipsparseHandle_t                  handle;   /* a handle to the hipsparse library */
  PetscInt                          nnz;      /* number of nonzeros ... need this for accurate logging between ICC and ILU */
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Tranpose */
struct Mat_SeqAIJHIPSPARSE {
  Mat_SeqAIJHIPSPARSEMultStruct *mat; /* pointer to the matrix on the GPU */
  Mat_SeqAIJHIPSPARSEMultStruct *matTranspose; /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                  *workVector; /*pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  PetscInt                     nonzerorow; /* number of nonzero rows ... used in the flop calculations */
  MatHIPSPARSEStorageFormat     format;   /* the storage format for the matrix on the device */
  hipStream_t                 stream;   /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  hipsparseHandle_t             handle;   /* a handle to the hipsparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState             nonzerostate;
};

PETSC_INTERN PetscErrorCode MatHIPSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetStream(Mat, const hipStream_t stream);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetHandle(Mat, const hipsparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatHIPSPARSEClearHandle(Mat);
#endif
