#if !defined(VECHYBRIDIMPL_H_)
#define VECHYBRIDIMPL_H_

#include <petschybrid.h>
#include <petsc/private/vecimpl.h>

#include "libaxb.h"

// TODO: Properly define this
#define WaitForGPU() 0


typedef struct {
  struct axbVec_s *vec;
} Vec_Hybrid;

struct axbHandle_s     *axb_handle;
struct axbMemBackend_s *axb_mem_backend;
struct axbOpBackend_s  *axb_op_backend;

PETSC_INTERN PetscErrorCode VecDotNorm2_SeqHybrid(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqHybrid(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqHybrid(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqHybrid(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecSet_SeqHybrid(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqHybrid(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqHybrid(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqHybrid(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqHybrid(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqHybrid(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqHybrid(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqHybrid(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqHybrid(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqHybrid(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqHybrid(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqHybrid(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecNorm_SeqHybrid(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecHybridCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecHybridAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIHybrid(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqHybrid(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecCopy_SeqHybrid_Private(Vec,Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHybrid(Vec);






#endif
