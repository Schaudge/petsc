#ifdef PETSC_DEBUG_MANAGED_TYPE_IMPL_PRIVATE__
#  include <petsc/private/deviceimpl.h>
#  include <petsc/private/cpputil.hpp>
#  include "objpool.hpp"
#  define PetscTypeSuffix   Scalar
#  define PetscTypeSuffix_L scalar
#endif

#include <petscmanagedtypedefs.inl>

#if !defined(PetscTypeSuffix_L)
#  error "Must define PetscTypeSuffix_L"
#endif

#ifndef PetscConcat3
#define PetscConcat3_(a,b,c) a ## b ## c
#define PetscConcat3(a,b,c) PetscConcat3_(a,b,c)
#endif

#ifndef PetscManagedTypeCallMethod
#define PetscManagedTypeCallMethod(func__,...) do {                                            \
    PetscCheck(                                                                                \
      (func__),PETSC_COMM_SELF,PETSC_ERR_SUP,                                                  \
      "No support operation for operation %s (" PetscStringize(func__) ")",PETSC_FUNCTION_NAME \
    );                                                                                         \
    PetscCall((*(func__))(__VA_ARGS__));                                                       \
} while (0)
#endif

// utility
#define PetscValidTypePointer                     PetscConcat3(PetscValid,PetscTypeSuffix,Pointer)
#define PetscManagedTypeSetPurity_Private         PetscConcat(PetscManagedType,SetPurity_Private)
#define PetscManagedTypeGetOffloadMask_Private    PetscConcat(PetscManagedType,GetOffloadMask_Private)
#define PetscManagedTypeSetOffloadMask_Private    PetscConcat(PetscManagedType,SetOffloadMask_Private)
#define PetscManagedTypeSetLock_Private           PetscConcat(PetscManagedType,SetLock_Private)
#define PetscManagedTypeGetLock_Private           PetscConcat(PetscManagedType,GetLock_Private)
#define PetscManagedTypeCheckLock_Private         PetscConcat(PetscManagedType,CheckLock_Private)
#define PetscManagedTypeCopyValues_Private        PetscConcat(PetscManagedType,CopyValues_Private)
#define PetscManagedTypeCheckCopyMode_Private     PetscConcat(PetscManagedType,CheckCopyMode_Private)
#define PetscManagedTypeCheckSuperOwnsSub_Private PetscConcat(PetscManagedType,CheckSuperOwnsSub_Private)
#define PetscManagedTypeDestroySpecific_Private   PetscConcat(PetscManagedType,DestroySpecific_Private)
#define destroymanagedtype                        PetscConcat(destroymanaged,PetscTypeSuffix_L)
#define getmanagedvaluestype                      PetscConcat(getmanagedvalues,PetscTypeSuffix_L)
#define applyoperatortype                         PetscConcat(applyoperator,PetscTypeSuffix_L)

/* -------------------------------------------------------------------------------- */
/*                                 utility functions                                */
/* -------------------------------------------------------------------------------- */

static PetscErrorCode PetscManagedTypeSetPurity_Private(PetscManagedType scal, PetscBool pure)
{
  PetscFunctionBegin;
  scal->pure = pure;
  // if impure, then we need to mark the parent as impure as well. cannot set the parent as
  // pure if we are though, since the parent may have other (impure) children
  if (!pure && scal->parent) PetscCall(PetscManagedTypeSetPurity_Private(scal->parent,pure));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeGetOffloadMask_Private(PetscManagedType scal, PetscOffloadMask *mask)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CXX)
  *mask = scal->mask;
#else
  *mask = scal->host ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_UNALLOCATED;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeSetOffloadMask_Private(PetscManagedType scal, PetscOffloadMask mask)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CXX)
  if (scal->mask != mask) {
    scal->mask = mask;
    // should not update the parent if our mask did not change!
    if (scal->parent) PetscCall(PetscManagedTypeSetOffloadMask_Private(scal->parent,mask));
  }
#else
  (void)scal;
  (void)mask;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeSetLock_Private(PetscManagedType scal, PetscBool lock)
{
  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  scal->lock = (PetscInt)lock;
#else
  (void)scal;
  (void)lock;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeGetLock_Private(PetscManagedType scal, PetscBool *lock)
{
  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  *lock = scal->lock ? PETSC_TRUE : PETSC_FALSE;
#else
  (void)scal;
  *lock = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeCheckLock_Private(PetscManagedType scal, PetscBool v)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const char *const strings[] = {"unlocked","locked"};
    PetscBool         locked;

    PetscCall(PetscManagedTypeGetLock_Private(scal,&locked));
    PetscCheck(locked == v,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Managed type object is %s expected it to be %s",strings[locked],strings[v]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeCopyValues_Private(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask src_mask, const PetscType *src_ptr)
{
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(PetscManagedTypeGetSize(scal,&n));
  if (PetscLikely(n)) {
    PetscDeviceCopyMode  mode;
    PetscOffloadMask     mask;
    PetscType           *ptr;

    PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
    PetscCall(PetscManagedTypeGetValues(dctx,scal,PetscOffloadMaskToMemType(mask),PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&ptr));
    PetscCall(PetscOffloadMaskToDeviceCopyMode(mask,src_mask,&mode));
    PetscCall(PetscDeviceArrayCopy(dctx,ptr,src_ptr,n,mode));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeCheckCopyMode_Private(PetscCopyMode mode, PetscType **ptr, const char name[])
{
  PetscFunctionBegin;
  PetscAssert(mode != PETSC_COPY_VALUES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Managed types should not have a %s copy mode of %s",PetscCopyModes[mode],name);
  if (mode == PETSC_USE_POINTER) *ptr = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeCheckSuperOwnsSub_Private(const PetscType* super_begin, PetscInt supern, const PetscType *sub_begin, PetscInt subn)
{
  const char base_mess[] = "Sub-range does nto appear to belong to input amanged type;";

  PetscFunctionBegin;
  // do pointer checks here since pointer addition on nullptr is UB
  if (sub_begin) {
    PetscAssert(super_begin,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"%s sub-range begin is %p while super-begin is NULL",base_mess,sub_begin);
    {
      const PetscType *super_end = super_begin+supern,*sub_end = sub_begin+subn;

      PetscAssert((super_begin <= sub_begin) && (sub_begin < super_end) && (sub_end <= super_end),PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"%s sub-range [%p, %p) not in super-range [%p, %p)",base_mess,sub_begin,sub_end,super_begin,super_end);
    }
  } else {
    PetscAssert(!super_begin,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"%s sub-range begin is NULL while super-begin is not (%p)",base_mess,super_begin);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscManagedTypeDestroySpecific_Private(PetscDeviceContext dctx, PetscManagedType *scal, PetscOffloadMask mask)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(scal,2);
  if (*scal) PetscCall(PetscManagedTypeEnsureOffload(dctx,*scal,mask,PETSC_TRUE));
  PetscCall(PetscManagedTypeDestroy(dctx,scal));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
/*                        implementation specific functions                         */
/* -------------------------------------------------------------------------------- */

PetscErrorCode PetscManagedTypeCreate(PetscDeviceContext dctx, PetscType *host_ptr, PetscType *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (host_ptr && n) PetscValidTypePointer(host_ptr,2);
  PetscAssert(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot request negative amount of managed memory %" PetscInt_FMT,n);
  PetscValidPointer(scal,8);
  if (host_ptr && device_ptr) {
    PetscAssert(mask != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Set both host and device pointer but offloadmask was %s",PetscOffloadMasks(mask));
    // this is the only instance in which we believe whatever the user has fed us
  } else if (host_ptr) {
    // clearly no device_ptr, so we own it
    mask         = PETSC_OFFLOAD_CPU;
    device_cmode = PETSC_OWN_POINTER;
  } else if (device_ptr) {
    // clearly no host_ptr, so we own it
    mask       = PETSC_OFFLOAD_GPU;
    host_cmode = PETSC_OWN_POINTER;
  } else {
    // user gave us nothing, we own everything
    mask       = PETSC_OFFLOAD_UNALLOCATED;
    host_cmode = device_cmode = PETSC_OWN_POINTER;
  }

  // finally get our pointer
  PetscCall(PetscManagedTypeAllocate(scal));

  // populate known quantities
  PetscCall(PetscObjectNewId_Internal(&(*scal)->id));
  (*scal)->n = n;
  PetscCall(PetscManagedTypeSetOffloadMask_Private(*scal,mask));
#if PetscDefined(HAVE_CXX)
  PetscCall(PetscDeviceContextGetDeviceType(dctx,&(*scal)->dtype));
  (*scal)->d_cmode = device_cmode;
#endif
  (*scal)->h_cmode = host_cmode;
  // unallocated is technically a "pure" state
  PetscCall(PetscManagedTypeSetPurity_Private(*scal,PetscOffloadDevice(mask) ? PETSC_FALSE : PETSC_TRUE));

  if (host_cmode == PETSC_COPY_VALUES) {
    PetscCall(PetscManagedTypeCopyValues_Private(dctx,*scal,PETSC_OFFLOAD_CPU,host_ptr));
    (*scal)->h_cmode = PETSC_OWN_POINTER;
  } else {
    (*scal)->host = host_ptr;
  }

#if PetscDefined(HAVE_CXX)
  if (device_cmode == PETSC_COPY_VALUES) {
    PetscCall(PetscManagedTypeCopyValues_Private(dctx,*scal,PETSC_OFFLOAD_GPU,device_ptr));
    (*scal)->d_cmode = PETSC_OWN_POINTER;
  } else {
    (*scal)->device = device_ptr;
  }
#else
  (void)device_ptr;
  (void)device_cmode;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscValidPointer(scal,2);
  if (!*scal) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,*scal,2);
  PetscCall(PetscManagedTypeCheckLock_Private(*scal,PETSC_FALSE));
  // if we don't own it, nullify it since the memory pools may try to return it below
  PetscCall(PetscManagedTypeCheckCopyMode_Private((*scal)->h_cmode,&(*scal)->host,"host"));
#if PetscDefined(HAVE_CXX)
  PetscCall(PetscManagedTypeCheckCopyMode_Private((*scal)->d_cmode,&(*scal)->device,"device"));
  PetscManagedTypeCallMethod(dctx->ops->destroymanagedtype,dctx,*scal);
#endif
  // if the host pointer still exists at this point it is because it didn't belong to its
  // respective memory pool. If copy mode is PETSC_OWN_POINTER its because we have
  // co-opted the users pointer, so we should free it now.
  if ((*scal)->host && ((*scal)->h_cmode == PETSC_OWN_POINTER)) PetscCall(PetscFree((*scal)->host));
  // cannot handle device pointers though
#if PetscDefined(HAVE_CXX)
  PetscAssert(!(*scal)->device || ((*scal)->d_cmode != PETSC_OWN_POINTER),PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext (id %" PetscInt64_FMT ", device type %s) failed to free the owned device pointer",dctx->id,PetscDeviceTypes[dctx->device->type]);
#endif
  PetscCall(PetscManagedTypeDeallocate(*scal));
  *scal = PETSC_NULLPTR;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscType **ptr)
{
  PetscOffloadMask mask;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,scal,2);
  PetscValidPointer(ptr,6);
  PetscCall(PetscManagedTypeCheckLock_Private(scal,PETSC_FALSE));
  PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
  PetscCheck(!(PetscOffloadUnallocated(mask) && PetscMemoryAccessRead(mode)),PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Trying to read (using %s) from a managed type (id %" PetscInt64_FMT ") that has not been written to (has offload mask %s)",PetscMemoryAccessModes(mode),scal->id,PetscOffloadMasks(mask));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx,scal->id,mode,PETSC_NULLPTR));
  if (PetscDefined(HAVE_CXX)) {
    PetscManagedTypeCallMethod(dctx->ops->getmanagedvaluestype,dctx,scal,mtype,mode,ptr);
    // if user intends to write to device in any capacity then we are impure
    if (PetscMemTypeDevice(mtype) && PetscMemoryAccessWrite(mode)) {
      PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_FALSE));
    }
    // get the updated mask
    PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
  } else {
    PetscAssert(PetscMemTypeHost(mtype),PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot request memory of type %s (device memory), reconfigure with device support",PetscMemTypes(mtype));
    if (!scal->host) {
      PetscInt n;

      PetscAssert(PetscOffloadUnallocated(mask),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Managed type (id %" PetscInt64_FMT ") has offload mask %s, but no valid pointer %p",scal->id,PetscOffloadMasks(mask),scal->host);
      PetscCall(PetscManagedTypeGetSize(scal,&n));
      PetscCall(PetscMalloc1(n,&scal->host));
      scal->h_cmode = PETSC_OWN_POINTER;
      PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_TRUE));
      mask = PETSC_OFFLOAD_CPU;
    }
    *ptr = scal->host;
  }
  // also sets the parents mask if needed
  PetscCall(PetscManagedTypeSetOffloadMask_Private(scal,mask));
  // REVIEW ME:
  // if we are pure, there is no need to synchronize (I think)
  if (sync && !scal->pure) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    if (PetscMemTypeHost(mtype)) PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_TRUE));
  }
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ManagedType (id %" PetscInt64_FMT ") Returned null pointer for mtype %s",scal->id,PetscMemTypes(mtype));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeApplyOperator(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, PetscManagedType ret)
{
  PetscBool                   src_avail  = PETSC_FALSE;
  const PetscBool             in_place   = (PetscBool)(!ret || ret == scal);
  const PetscMemoryAccessMode src_access = in_place ? PETSC_MEMORY_ACCESS_READ_WRITE : PETSC_MEMORY_ACCESS_READ;
  const PetscMemoryAccessMode ret_access = PETSC_MEMORY_ACCESS_WRITE;
  PetscType                   *ptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,scal,2);
  PetscCall(PetscManagedTypeCheckLock_Private(scal,PETSC_FALSE));
  if (ret) {
    PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,ret,6);
    PetscCall(PetscManagedTypeCheckLock_Private(ret,PETSC_FALSE));
  }
  if (PetscMemTypeHost(mtype)) {
    PetscValidTypePointer(rhs,5);
    // if rhs is host, check if we can short circuit and evade a copy
    PetscCall(PetscManagedTypeGetValuesAvailable(dctx,scal,mtype,src_access,&ptr,&src_avail));
  }
  if (src_avail) {
    const PetscType  rhsv = *rhs;
    PetscType       *retptr;
    PetscInt         n;

    if (in_place) {
      retptr = ptr;
    } else {
      PetscCall(PetscManagedTypeGetValues(dctx,ret,PETSC_MEMTYPE_HOST,ret_access,PETSC_TRUE,&retptr));
    }

    PetscCall(PetscManagedTypeGetSize(scal,&n));
    for (PetscInt i = 0; i < n; ++i) {
      switch (otype) {
      case PETSC_OPERATOR_PLUS:     retptr[i] = ptr[i]+rhsv; break;
      case PETSC_OPERATOR_MINUS:    retptr[i] = ptr[i]-rhsv; break;
      case PETSC_OPERATOR_MULTIPLY: retptr[i] = ptr[i]*rhsv; break;
      case PETSC_OPERATOR_DIVIDE:   retptr[i] = ptr[i]/rhsv; break;
      case PETSC_OPERATOR_EQUAL:    retptr[i] = rhsv;        break;
      }
    }
  } else {
    PetscCheck(PetscDefined(HAVE_CXX),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Should not get here if PetscDefined(HAVE_CXX) is false");
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx,scal->id,src_access,PETSC_NULLPTR));
    if (ret && !in_place) {
      PetscCall(PetscDeviceContextMarkIntentFromID(dctx,ret->id,ret_access,PETSC_NULLPTR));
    }
    PetscManagedTypeCallMethod(dctx->ops->applyoperatortype,dctx,scal,otype,mtype,rhs,ret);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
/*                        functions implementable via others                        */
/* -------------------------------------------------------------------------------- */

PetscErrorCode PetscManagedHostTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeDestroySpecific_Private(dctx,scal,PETSC_OFFLOAD_CPU));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedDeviceTypeDestroy(PetscDeviceContext dctx, PetscManagedType *scal)
{
  PetscFunctionBegin;
  PetscCall(PetscManagedTypeDestroySpecific_Private(dctx,scal,PETSC_OFFLOAD_GPU));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeSetValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, const PetscType *ptr, PetscInt n)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,scal,2);
  if (PetscMemTypeHost(mtype) && n) PetscValidPointer(ptr,4);
  PetscCall(PetscManagedTypeCheckLock_Private(scal,PETSC_FALSE));
  if (PetscDefined(USE_DEBUG)) {
    PetscInt scaln;

    PetscCall(PetscManagedTypeGetSize(scal,&scaln));
    PetscCheck(n <= scaln,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write %" PetscInt_FMT " values to " PetscStringize(PetscManagedType) " but it only holds %" PetscInt_FMT " entries",n,scaln);
  }
  if (n) {
    const PetscMemoryAccessMode  mode = PETSC_MEMORY_ACCESS_WRITE;
    PetscOffloadMask             mask;
    PetscMemType                 scalmtype = mtype;
    PetscType                   *scalptr;

    PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
    // if scal is unallocated or in both locations we defer to the given mtype, otherwise we
    // copy to wherever scal is currently up to date
    if (mask == PETSC_OFFLOAD_UNALLOCATED || mask == PETSC_OFFLOAD_BOTH) {
      PetscCall(PetscManagedTypeGetValues(dctx,scal,mtype,mode,PETSC_FALSE,&scalptr));
    } else {
      PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,scal,mode,&scalptr,&scalmtype));
    }

    if (PetscMemTypeHost(mtype) && PetscMemTypeHost(scalmtype) && scal->pure) {
      for (PetscInt i = 0; i < n; ++i) scalptr[i] = ptr[i];
    } else {
      // expensive!
      PetscCall(PetscDeviceArrayCopy(dctx,scalptr,ptr,n,PetscMemTypeToDeviceCopyMode(scalmtype,mtype)));
    }
    if (PetscMemTypeHost(mtype) && PetscMemTypeHost(scalmtype)) {
      // both on host? we are pure again
      PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_TRUE));
    } else if (PetscMemTypeDevice(mtype)) {
      // unless we are on device, in which case we are very much not pure
      PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_FALSE));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeGetPointerAndMemType(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType **ptr, PetscMemType *mtype)
{
  PetscMemType     retmtype;
  PetscOffloadMask mask;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,scal,2);
  PetscCall(PetscManagedTypeCheckLock_Private(scal,PETSC_FALSE));
  PetscValidPointer(ptr,4);
  if (mtype) PetscValidPointer(mtype,5);
  PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
  switch (mask) {
    // if both prefer CPU, since we may be able to set purity
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_CPU:
    retmtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_OFFLOAD_GPU:
    retmtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx,&dtype));
    retmtype = dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for offloadmask %d",mask);
  }
  PetscCall(PetscManagedTypeGetValues(dctx,scal,retmtype,mode,PETSC_FALSE,ptr));
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_PLIB,PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values",PetscMemTypes(retmtype));
  if (mtype) *mtype = retmtype;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeCopy(PetscDeviceContext dctx, PetscManagedType dest, PetscManagedType src)
{
  PetscInt sn,dn;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,dest,2);
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,src,3);
  PetscCall(PetscManagedTypeGetSize(dest,&dn));
  PetscCall(PetscManagedTypeGetSize(src,&sn));
  PetscAssert(dn >= sn,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT,dn,sn);
  if ((dest != src) && sn) {
    PetscMemType  dest_mtype = PETSC_MEMTYPE_DEVICE,src_mtype = PETSC_MEMTYPE_DEVICE;
    PetscType    *dest_ptr,*src_ptr;

    PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,dest,PETSC_MEMORY_ACCESS_WRITE,&dest_ptr,&dest_mtype));
    PetscCall(PetscManagedTypeGetPointerAndMemType(dctx,src,PETSC_MEMORY_ACCESS_READ,&src_ptr,&src_mtype));
    PetscCall(PetscDeviceArrayCopy(dctx,dest_ptr,src_ptr,sn,PetscMemTypeToDeviceCopyMode(dest_mtype,src_mtype)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeEnsureOffload(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask omask, PetscBool sync)
{
#ifdef __cplusplus
  static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH),"");
#endif
  PetscOffloadMask mask;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,scal,2);
  PetscAssert(omask != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot request %s as offload mask",PetscOffloadMasks(omask));
  PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
  if (mask != omask) {
    // if we are unallocated, the first access should "write" since we cannot "read"
    // unallocated memory, the second will write
    PetscMemoryAccessMode  mode     = PETSC_MEMORY_ACCESS_READ;
    // small optimization, we don't want the individal get_values calls to sync in case we do
    // it twice!
    const PetscBool        sub_sync = PETSC_FALSE;
    PetscType             *ptr;

    switch (mask) {
    case PETSC_OFFLOAD_BOTH: // we are already offloaded, so do nothing
      break;
    case PETSC_OFFLOAD_CPU:
      if (PetscOffloadDevice(omask)) {
        OFFLOAD_TO_GPU: // currently on CPU but want it on GPU
        PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_DEVICE,mode,sub_sync,&ptr));
      }
      break;
    case PETSC_OFFLOAD_GPU:
      if (PetscOffloadHost(omask)) {
        OFFLOAD_TO_CPU: // currently on GPU but also want it on CPU
        PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_HOST,mode,sub_sync,&ptr));
      }
      break;
    case PETSC_OFFLOAD_UNALLOCATED:
      mode = PETSC_MEMORY_ACCESS_WRITE;
      switch (omask) {
        // any of the "simple" requests we can just delegate to above (but instead of reading
        // we "write")
      case PETSC_OFFLOAD_CPU: goto OFFLOAD_TO_CPU;
      case PETSC_OFFLOAD_GPU: goto OFFLOAD_TO_GPU;
      case PETSC_OFFLOAD_BOTH:
        // The only complex configuration is when user wants both but we have neither. We
        // create some dummy host values first, then pipe them to device. The fact that we
        // "write" from host *first* is important! This allows us to maintain a "pure" state
        // since we "read" on the device-side
        PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_HOST,mode,sub_sync,&ptr));
        PetscCall(PetscManagedTypeGetValues(dctx,scal,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,sub_sync,&ptr));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported offloadmask %s",PetscOffloadMasks(omask));
      }
      break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported offloadmask %s",PetscOffloadMasks(mask));
    }

    PetscCall(PetscManagedTypeGetOffloadMask_Private(scal,&mask));
    PetscAssert(mask == omask || mask == PETSC_OFFLOAD_BOTH,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Managed type offload mask %s != requested mask %s",PetscOffloadMasks(mask),PetscOffloadMasks(omask));
  } else {
    // even if we don't alter the offloadmask we still count this context as "reading"
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx,scal->id,PETSC_MEMORY_ACCESS_READ,PETSC_NULLPTR));
  }
  // we're going to assume that the user has properly serialized the device contexts here.
  // it is possible for the user to have changed the offload mask on one stream, then call
  // this routine with another and hence pass the mask check above. But this event is
  // unlikely enough that it doesn't warrant throwing away the purity optimization
  if (sync) {
    if (!scal->pure) PetscCall(PetscDeviceContextSynchronize(dctx));
    if (PetscOffloadHost(omask)) PetscCall(PetscManagedTypeSetPurity_Private(scal,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

// REVIEW ME: can only have a single outstanding sub-range, this seems a bit restrictive? but
// then how to handle the child changing values?
PetscErrorCode PetscManagedTypeGetSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscInt begin, PetscInt len, PetscManagedType *out)
{
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,in,2);
  PetscValidPointer(out,5);
  PetscCall(PetscManagedTypeGetSize(in,&n));
  PetscAssert(len >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Cannot extract a subrange of negative size %" PetscInt_FMT,len);
  PetscAssert(begin >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Begin %" PetscInt_FMT "must be >= 0",begin);
  PetscAssert(begin+len <= n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Trying to extract a subrange of [%" PetscInt_FMT ",%" PetscInt_FMT ") from managed type of size %" PetscInt_FMT,begin,begin+len,n);
  if (len == n) {
    // curious case of trying to extract a subrange that is exactly the size of the current
    // object, in which case we simply return ourselves and don't need to lock
    *out = in;
  } else {
    const PetscBool   purity       = in->pure;
    PetscOffloadMask  mask         = PETSC_OFFLOAD_BOTH;
    PetscType        *host_begin   = PETSC_NULLPTR;
    PetscType        *device_begin = PETSC_NULLPTR;
    PetscDeviceType   dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx,&dtype));
    if (dtype == PETSC_DEVICE_HOST) mask = PETSC_OFFLOAD_CPU;
    PetscCall(PetscManagedTypeEnsureOffload(dctx,in,mask,PETSC_FALSE));
    host_begin = in->host+begin;
#if PetscDefined(HAVE_CXX)
    device_begin = in->device;
#endif
    if (device_begin) device_begin += begin;
    PetscCheck(in->pure == purity,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscManagedTypeGetSubRange() should not be changing purity, went from %s to %s",PetscBools[purity],PetscBools[in->pure]);
    PetscCall(PetscManagedTypeGetOffloadMask_Private(in,&mask));
    PetscCall(PetscManagedTypeCreate(dctx,host_begin,device_begin,len,PETSC_USE_POINTER,PETSC_USE_POINTER,mask,out));
    // copy state over to the subrange
    PetscCall(PetscManagedTypeSetPurity_Private(*out,in->pure));
    // set the parent *after* we set purity! otherwise setpurity loops forever
    (*out)->parent = in;
    // unlock the subrange
    PetscCall(PetscManagedTypeSetLock_Private(*out,PETSC_FALSE));
    // but lock ourselves
    PetscCall(PetscManagedTypeSetLock_Private(in,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeRestoreSubRange(PetscDeviceContext dctx, PetscManagedType in, PetscManagedType *out)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,in,2);
  PetscValidPointer(out,3);
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,*out,3);
  if (in == *out) {
    // case where the subrange len was the same as the original (and hence we returned
    // ourselves)
    *out = PETSC_NULLPTR;
  } else {
    PetscInt         inn,outn;
    PetscOffloadMask mask;

    // assert that we are locked
    PetscCall(PetscManagedTypeCheckLock_Private(in,PETSC_TRUE));
    // the restored obj can't also have an outstanding subrange
    PetscCall(PetscManagedTypeCheckLock_Private(*out,PETSC_FALSE));
    PetscCall(PetscManagedTypeGetSize(in,&inn));
    PetscCall(PetscManagedTypeGetSize(*out,&outn));
    PetscAssert(outn < inn,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sub-range does not appear to belong to input managed type, sub length %" PetscInt_FMT " > input length %" PetscInt_FMT,outn,inn);
    PetscCall(PetscManagedTypeCheckSuperOwnsSub_Private(in->host,inn,(*out)->host,outn));
#if PetscDefined(HAVE_CXX)
    PetscCall(PetscManagedTypeCheckSuperOwnsSub_Private(in->device,inn,(*out)->device,outn));
#endif
    PetscCall(PetscManagedTypeGetOffloadMask_Private(*out,&mask));
    PetscCall(PetscManagedTypeSetOffloadMask_Private(in,mask));
    (*out)->parent = PETSC_NULLPTR;
#if PetscDefined(USE_DEBUG)
    in->lock = (*out)->lock;
#endif
    PetscCall(PetscManagedTypeDestroy(dctx,out));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscManagedTypeEqual(PetscManagedType scal, PetscType val, PetscBool *known, PetscBool *equal)
{
  PetscFunctionBegin;
  PetscValidManagedType(scal,1);
  PetscValidBoolPointer(known,3);
  PetscValidBoolPointer(equal,4);
  *equal = PETSC_FALSE;
  *known = scal->pure;
  // REVIEW ME:
  // it if is unknown we can technically just forgo the equal check since it is worthless
  // anyways
  if (*known && (scal->host)) {
    PetscInt n;

    PetscCall(PetscManagedTypeGetSize(scal,&n));
    for (PetscInt i = 0; i < n; ++i) if (scal->host[i] != val) PetscFunctionReturn(0);
    // if we get here it is equal
    *equal = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef PetscValidTypePointer
#undef PetscManagedTypeSetPurity_Private
#undef PetscManagedTypeGetOffloadMask_Private
#undef PetscManagedTypeSetOffloadMask_Private
#undef PetscManagedTypeSetLock_Private
#undef PetscManagedTypeGetLock_Private
#undef PetscManagedTypeCheckLock_Private
#undef PetscManagedTypeCopyValues_Private
#undef PetscManagedTypeCheckCopyMode_Private
#undef PetscManagedTypeCheckSuperOwnsSub_Private
#undef PetscManagedTypeDestroySpecific_Private
#undef destroymanagedtype
#undef getmanagedvaluestype
#undef applyoperatortype
#undef PetscTypeSuffix_L

#include <petscmanagedtypeundefs.inl>
