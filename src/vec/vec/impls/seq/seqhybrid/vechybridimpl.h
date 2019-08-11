#if !defined(VECHYBRIDIMPL_H_)
#define VECHYBRIDIMPL_H_

#include <petschybrid.h>
#include <petsc/private/vecimpl.h>

#include "libaxb.h"

// TODO: Properly define this
#define WaitForGPU() 0


typedef struct {
  axbVec_t vec;
} Vec_Hybrid;



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
PETSC_EXTERN PetscErrorCode VecCopy_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqHybrid(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqHybrid(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqHybrid(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqHybrid(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecConjugate_SeqHybrid(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqHybrid(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecHybridCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecHybridAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqHybrid_Private(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_MPIHybrid(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIHybrid_Private(Vec,PetscBool,PetscInt,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_Hybrid(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHybrid(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIHybrid(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqHybrid(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqHybrid(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqHybrid(Vec,Vec);
PETSC_INTERN PetscErrorCode VecCopy_SeqHybrid_Private(Vec xin,Vec yin);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqHybrid_Private(Vec xin,PetscRandom r);
PETSC_INTERN PetscErrorCode VecDestroy_SeqHybrid_Private(Vec v);
PETSC_INTERN PetscErrorCode VecResetArray_SeqHybrid_Private(Vec vin);

PETSC_INTERN PetscErrorCode VecHybridCopyToGPU_Public(Vec);
PETSC_INTERN PetscErrorCode VecHybridAllocateCheck_Public(Vec);





#endif
