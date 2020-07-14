#if !defined(VECAXBIMPL_H_)
#define VECAXBIMPL_H_

#include <petscaxb.h>
#include <petsc/private/vecimpl.h>

#include "libaxb.h"

// TODO: Properly define this
#define WaitForGPU() 0


typedef struct {
  struct axbVec_s *vec;
} Vec_AXB;

struct axbHandle_s     *axb_handle;
struct axbMemBackend_s *axb_mem_backend;
struct axbOpBackend_s  *axb_op_backend;

PETSC_INTERN PetscErrorCode VecDotNorm2_SeqAXB(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqAXB(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqAXB(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqAXB(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecSet_SeqAXB(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqAXB(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqAXB(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqAXB(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqAXB(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqAXB(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqAXB(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqAXB(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqAXB(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqAXB(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqAXB(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqAXB(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqAXB(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqAXB(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqAXB(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecNorm_SeqAXB(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecAXBCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecAXBAllocateCheck(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqAXB(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqAXB(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_MPIAXB(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqAXB(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqAXB(Vec,Vec);
PETSC_INTERN PetscErrorCode VecCopy_SeqAXB_Private(Vec,Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqAXB(Vec);
PETSC_INTERN PetscErrorCode VecResetArray_SeqAXB(Vec);






#endif
