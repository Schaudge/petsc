#if !defined(__HIPVECIMPL)
#define __HIPVECIMPL

#include <petscvec.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscScalar  *GPUarray;           /* this always holds the GPU data */
  PetscScalar  *GPUarray_allocated; /* if the array was allocated by PETSc this is its pointer */
  hipStream_t stream;              /* A stream for doing asynchronous data transfers */
} Vec_HIP;

PETSC_INTERN PetscErrorCode VecHIPGetArrays_Private(Vec,const PetscScalar**,const PetscScalar**,PetscOffloadMask*);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqHIP(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqHIP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqHIP(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqHIP(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode VecSet_SeqHIP(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqHIP(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqHIP(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqHIP(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqHIP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqHIP(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqHIP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqHIP(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqHIP(Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode VecCopy_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqHIP(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqHIP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqHIP(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqHIP(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecConjugate_SeqHIP(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqHIP(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecHIPCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecHIPAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqHIP_Private(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIHIP_Private(Vec,PetscBool,PetscInt,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_HIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIHIP(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqHIP(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqHIP(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecGetLocalVectorRead_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVectorRead_SeqHIP(Vec,Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArray_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqHIP(Vec,PetscScalar**,PetscMemType*);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqHIP(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecCopy_SeqHIP_Private(Vec,Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHIP_Private(Vec);
PETSC_INTERN PetscErrorCode VecMax_SeqHIP(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMin_SeqHIP(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqHIP(Vec);
PETSC_INTERN PetscErrorCode VecSum_SeqHIP(Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecShift_SeqHIP(Vec,PetscScalar);

/* complex single */
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define hipblasXaxpy(a,b,c,d,e,f,g) rocblas_caxpy((a),(b),(rocblas_float_complex*)(c),(rocblas_float_complex*)(d),(e),(rocblas_float_complex*)(f),(g))
#define hipblasXscal(a,b,c,d,e)     rocblas_cscal((a),(b),(rocblas_float_complex*)(c),(rocblas_float_complex*)(d),(e))
#define hipblasXdotu(a,b,c,d,e,f,g) rocblas_cdotu((a),(b),(rocblas_float_complex*)(c),(d),(rocblas_float_complex*)(e),(f),(rocblas_float_complex*)(g))
#define hipblasXdot(a,b,c,d,e,f,g)  rocblas_cdotc((a),(b),(rocblas_float_complex*)(c),(d),(rocblas_float_complex*)(e),(f),(rocblas_float_complex*)(g))
#define hipblasXswap(a,b,c,d,e,f)   rocblas_cswap((a),(b),(rocblas_float_complex*)(c),(d),(rocblas_float_complex*)(e),(f))
#define hipblasXnrm2(a,b,c,d,e)     rocblas_scnrm2((a),(b),(rocblas_float_complex*)(c),(d),(e))
#define hipblasIXamax(a,b,c,d,e)    rocblas_icamax((a),(b),(rocblas_float_complex*)(c),(d),(e))
#define hipblasXasum(a,b,c,d,e)     rocblas_scasum((a),(b),(rocblas_float_complex*)(c),(d),(e))
#define hipblasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) rocblas_cgemv((a),(b),(c),(d),(rocblas_float_complex*)(e),(rocblas_float_complex*)(f),(g),(rocblas_float_complex*)(h),(i),(rocblas_float_complex*)(j),(rocblas_float_complex*)(k),(l))
#define hipblasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) rocblas_cgemm((a),(b),(c),(d),(e),(f),(rocblas_float_complex*)(g),(rocblas_float_complex*)(h),(i),(rocblas_float_complex*)(j),(k),(rocblas_float_complex*)(l),(rocblas_float_complex*)(m),(n))
#define hipblasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   rocblas_cgeam((a),(b),(c),(d),(e),(rocblas_float_complex*)(f),(rocblas_float_complex*)(g),(h),(rocblas_float_complex*)(i),(rocblas_float_complex*)(j),(k),(rocblas_float_complex*)(l),(m))
#else /* complex double */
#define hipblasXaxpy(a,b,c,d,e,f,g) rocblas_Zaxpy((a),(b),(rocblas_double_complex*)(c),(rocblas_double_complex*)(d),(e),(rocblas_double_complex*)(f),(g))
#define hipblasXscal(a,b,c,d,e)     rocblas_zscal((a),(b),(rocblas_double_complex*)(c),(rocblas_double_complex*)(d),(e))
#define hipblasXdotu(a,b,c,d,e,f,g) rocblas_zdotu((a),(b),(rocblas_double_complex*)(c),(d),(rocblas_double_complex*)(e),(f),(rocblas_double_complex*)(g))
#define hipblasXdot(a,b,c,d,e,f,g)  rocblas_zdotc((a),(b),(rocblas_double_complex*)(c),(d),(rocblas_double_complex*)(e),(f),(rocblas_double_complex*)(g))
#define hipblasXswap(a,b,c,d,e,f)   rocblas_zswap((a),(b),(rocblas_double_complex*)(c),(d),(rocblas_double_complex*)(e),(f))
#define hipblasXnrm2(a,b,c,d,e)     rocblas_dznrm2((a),(b),(rocblas_double_complex*)(c),(d),(e))
#define hipblasIXamax(a,b,c,d,e)    rocblas_izamax((a),(b),(rocblas_double_complex*)(c),(d),(e))
#define hipblasXasum(a,b,c,d,e)     rocblas_dzasum((a),(b),(rocblas_double_complex*)(c),(d),(e))
#define hipblasXgemv(a,b,c,d,e,f,g,h,i,j,k,l) rocblas_zgemv((a),(b),(c),(d),(rocblas_double_complex*)(e),(rocblas_double_complex*)(f),(g),(rocblas_double_complex*)(h),(i),(rocblas_double_complex*)(j),(rocblas_double_complex*)(k),(l))
#define hipblasXgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n) rocblas_zgemm((a),(b),(c),(d),(e),(f),(rocblas_double_complex*)(g),(rocblas_double_complex*)(h),(i),(rocblas_double_complex*)(j),(k),(rocblas_double_complex*)(l),(rocblas_double_complex*)(m),(n))
#define hipblasXgeam(a,b,c,d,e,f,g,h,i,j,k,l,m)   rocblas_zgeam((a),(b),(c),(d),(e),(rocblas_double_complex*)(f),(rocblas_double_complex*)(g),(h),(rocblas_double_complex*)(i),(rocblas_double_complex*)(j),(k),(rocblas_double_complex*)(l),(m))
#endif
#else /* real single */
#if defined(PETSC_USE_REAL_SINGLE)
#define hipblasXaxpy  rocblas_saxpy
#define hipblasXscal  rocblas_sscal
#define hipblasXdotu  rocblas_sdot
#define hipblasXdot   rocblas_sdot
#define hipblasXswap  rocblas_sswap
#define hipblasXnrm2  rocblas_snrm2
#define hipblasIXamax rocblas_isamax
#define hipblasXasum  rocblas_sasum
#define hipblasXgemv  rocblas_sgemv
#define hipblasXgemm  rocblas_sgemm
#define hipblasXgeam  rocblas_sgeam
#else /* real double */
#define hipblasXaxpy  rocblas_daxpy
#define hipblasXscal  rocblas_dscal
#define hipblasXdotu  rocblas_ddot
#define hipblasXdot   rocblas_ddot
#define hipblasXswap  rocblas_dswap
#define hipblasXnrm2  rocblas_dnrm2
#define hipblasIXamax rocblas_idamax
#define hipblasXasum  rocblas_dasum
#define hipblasXgemv  rocblas_dgemv
#define hipblasXgemm  rocblas_dgemm
#define hipblasXgeam  rocblas_dgeam
#endif
#endif

#endif
