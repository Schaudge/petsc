#if !defined(PETSCHIPBLAS_H)
#define PETSCHIPBLAS_H

/* #include <hip/hip_runtime.h> */
/* #include <hipblas.h> */
/* #include <cusolverDn.h> */
#include <petscsys.h>

#define WaitForGPU() PetscHIPSynchronize ? hipDeviceSynchronize() : hipSuccess;

#define CHKERRHIP(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = hipGetErrorName(cerr); \
      const char *descr = hipGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hip error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while(0)

#define CHKERRHIPBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscHIPBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipBLAS error %d (%s)",(int)stat,name); \
   } \
} while(0)

PETSC_INTERN PetscErrorCode PetscHIPBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscHIPSOLVERDnInitializeHandle(void);

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */
PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERDnGetHandle(hipsolverDnHandle_t*);
#endif
