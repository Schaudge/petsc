#include <petscmacros.h>

#if !defined(PetscTypeSuffix)
#  error "PetscTypeSuffix must be defined"
#endif

#ifndef PetscManagedType
#define PetscManagedType PetscConcat(PetscManaged,PetscTypeSuffix)
#endif

#ifndef PetscType
#define PetscType PetscConcat(Petsc,PetscTypeSuffix)
#endif

#define _n_PetscManagedType PetscConcat(_n_,PetscManagedType)

// real implementations found in src/sys/objects/device/interface/managedtype.cxx
#define PetscManagedTypeCreate               PetscConcat(PetscManagedType,Create)
#define PetscManagedTypeCreateDefault        PetscConcat(PetscManagedTypeCreate,Default)
#define PetscManageHostType                  PetscConcat(PetscManageHost,PetscTypeSuffix)
#define PetscCopyHostType                    PetscConcat(PetscCopyHost,PetscTypeSuffix)
#define PetscManageDeviceType                PetscConcat(PetscManageDevice,PetscTypeSuffix)
#define PetscCopyDeviceType                  PetscConcat(PetscCopyDevice,PetscTypeSuffix)
#define PetscManagedHostTypeDestroy          PetscConcat(PetscConcat(PetscManagedHost,PetscTypeSuffix),Destroy)
#define PetscManagedDeviceTypeDestroy        PetscConcat(PetscConcat(PetscManagedDevice,PetscTypeSuffix),Destroy)
#define PetscManagedTypeDestroy              PetscConcat(PetscManagedType,Destroy)
#define PetscManagedTypeGetValues            PetscConcat(PetscManagedType,GetValues)
#define PetscManagedTypeSetValues            PetscConcat(PetscManagedType,SetValues)
#define PetscManagedTypeGetPointerAndMemType PetscConcat(PetscManagedType,GetPointerAndMemType)
#define PetscManagedTypeEnsureOffload        PetscConcat(PetscManagedType,EnsureOffload)
#define PetscManagedTypeCopy                 PetscConcat(PetscManagedType,Copy)
#define PetscManagedTypeApplyOperator        PetscConcat(PetscManagedType,ApplyOperator)
#define PetscManagedTypeGetSubRange          PetscConcat(PetscManagedType,GetSubRange)
#define PetscManagedTypeRestoreSubRange      PetscConcat(PetscManagedType,RestoreSubRange)
#define PetscManagedTypeEqual                PetscConcat(PetscManagedType,Equal)
#define PetscManagedTypeGetSize              PetscConcat(PetscManagedType,GetSize)
#define PetscManagedTypeKnownAndEqual        PetscConcat(PetscManagedType,KnownAndEqual)
#define PetscManagedTypeGetValuesAvailable   PetscConcat(PetscManagedType,GetValuesAvailable)
