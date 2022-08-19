#include <petscdevicetypes.h>
#include <petscsys.h>
#include <petscmanagedtypedefs.inl>

typedef struct _n_PetscManagedType *PetscManagedType;
struct _n_PetscManagedType
{
  PetscType        *host;
#if PetscDefined(HAVE_CXX)
  PetscType        *device;
  PetscDeviceType   dtype;
  PetscOffloadMask  mask;
  PetscCopyMode     d_cmode;
#endif
  PetscCopyMode     h_cmode;
  PetscInt          n;
  PetscObjectId     id;
  PetscManagedType  parent;
#if PetscDefined(USE_DEBUG)
  PetscInt          lock; // >1  = locked, 0 = unlocked
#endif
  PetscBool         pure; // is offload to be believed
};

PETSC_EXTERN PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext,PetscType*,PetscType*,PetscInt,PetscCopyMode,PetscCopyMode,PetscOffloadMask,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeGetArray(PetscDeviceContext,PetscManagedType,PetscMemType,PetscMemoryAccessMode,PetscBool,PetscType**);
PETSC_EXTERN PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext,PetscManagedType,PetscMemType,const PetscType*,PetscInt);
PETSC_EXTERN PetscErrorCode PetscManagedTypeGetPointerAndMemType(PetscDeviceContext,PetscManagedType,PetscMemoryAccessMode,PetscType**,PetscMemType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeEnsureOffload(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscBool);
PETSC_EXTERN PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext,PetscManagedType,PetscManagedType);
PETSC_EXTERN PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext,PetscManagedType,PetscOperatorType,PetscMemType,const PetscType*,PetscManagedType);
PETSC_EXTERN PetscErrorCode PetscManagedTypeGetSubRange(PetscDeviceContext,PetscManagedType,PetscInt,PetscInt,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeRestoreSubRange(PetscDeviceContext,PetscManagedType,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedTypeEqual(PetscManagedType,PetscType,PetscBool*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscManagedHostTypeDestroy(PetscDeviceContext,PetscManagedType*);
PETSC_EXTERN PetscErrorCode PetscManagedDeviceTypeDestroy(PetscDeviceContext,PetscManagedType*);

static inline PetscErrorCode PetscManageHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_USE_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_CPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscCopyHostType(PetscDeviceContext dctx, PetscType *host_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,host_ptr,PETSC_NULLPTR,n,PETSC_COPY_VALUES,PETSC_OWN_POINTER,PETSC_OFFLOAD_CPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManageDeviceType(PetscDeviceContext dctx, PetscType *device_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,device_ptr,n,PETSC_OWN_POINTER,PETSC_USE_POINTER,PETSC_OFFLOAD_GPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscCopyDeviceType(PetscDeviceContext dctx, PetscType *device_ptr, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,device_ptr,n,PETSC_OWN_POINTER,PETSC_COPY_VALUES,PETSC_OFFLOAD_GPU,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeCreateDefault(PetscDeviceContext dctx, PetscInt n, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeCreate(dctx,PETSC_NULLPTR,PETSC_NULLPTR,n,PETSC_OWN_POINTER,PETSC_OWN_POINTER,PETSC_OFFLOAD_UNALLOCATED,scal));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeGetSize(PetscManagedType scal, PetscInt *n)
{
  PetscFunctionBegin;
  PetscAssert(n,PETSC_COMM_SELF,PETSC_ERR_POINTER,"Null pointer, argument 2");
  *n = scal->n;
  PetscFunctionReturn(0);
}

static inline PetscBool PetscManagedTypeKnownAndEqual(PetscManagedType scal, PetscType v)
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF,PetscManagedTypeEqual(scal,v,&known,&equal));
  known = (PetscBool)(known && equal);
  PetscFunctionReturn(known);
}

static inline PetscErrorCode PetscManagedTypeGetArrayAvailable(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr, PetscBool *avail)
{
  PetscFunctionBegin;
  // todo
  *ptr   = PETSC_NULLPTR;
  *avail = PetscMemTypeHost(mtype) ? scal->pure : PETSC_FALSE;
  if (*avail) PetscCall(PetscManagedTypeGetArray(dctx,scal,mtype,mode,PETSC_FALSE,ptr));
  PetscFunctionReturn(0);
}

#include <petscmanagedtypeundefs.inl>
