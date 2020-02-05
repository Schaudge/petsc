#if !defined(__MPIHIPSPARSEMATIMPL)
#define __MPIHIPSPARSEMATIMPL

#include <hipsparse.h>
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatHIPSPARSEStorageFormat diagGPUMatFormat;
  MatHIPSPARSEStorageFormat offdiagGPUMatFormat;
  hipStream_t             stream;
  hipsparseHandle_t         handle;
} Mat_MPIAIJHIPSPARSE;

PETSC_INTERN PetscErrorCode MatHIPSPARSESetStream(Mat, const hipStream_t stream);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetHandle(Mat, const hipsparseHandle_t handle);
PETSC_INTERN PetscErrorCode MatHIPSPARSEClearHandle(Mat);

#endif
