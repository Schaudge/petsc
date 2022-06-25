#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

#include <array>
#include <vector>
#include <algorithm> // std::find
#include <unordered_map>
#include <type_traits> // std::is_same
#define PETSC_USE_DEBUG_AND_INFO (PetscDefined(USE_DEBUG) && PetscDefined(USE_INFO))
#if PETSC_USE_DEBUG_AND_INFO
#include <string>
#endif

const char *const PetscStreamTypes[] = {"global_blocking", "default_blocking", "global_nonblocking", "max", "PetscStreamType", "PETSC_STREAM_", nullptr};

const char *const PetscDeviceContextJoinModes[] = {"destroy", "sync", "no_sync", "PetscDeviceContextJoinMode", "PETSC_DEVICE_CONTEXT_JOIN_", nullptr};

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_AUTO) == 4, "");

/* Define the allocator */
struct PetscDeviceContextAllocator : Petsc::AllocatorBase<PetscDeviceContext> {
  static PetscInt PetscDeviceContextID;

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext *dctx)) {
    PetscFunctionBegin;
    PetscCall(PetscNew(dctx));
    (*dctx)->id         = PetscDeviceContextID++;
    (*dctx)->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying", dctx->numChildren);
    PetscTryTypeMethod(dctx, destroy);
    PetscCall(PetscDeviceDestroy(&dctx->device));
    PetscCall(PetscFree(dctx->childIDs));
    PetscCall(PetscFree(dctx));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    /* don't deallocate the child array, rather just zero it out */
    PetscCall(PetscArrayzero(dctx->childIDs, dctx->maxNumChildren));
    dctx->setup       = PETSC_FALSE;
    dctx->numChildren = 0;
    dctx->streamType  = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};
/* an ID = 0 is invalid */
PetscInt PetscDeviceContextAllocator::PetscDeviceContextID = 1;

static Petsc::ObjectPool<PetscDeviceContext, PetscDeviceContextAllocator> contextPool;

/*@C
  PetscDeviceContextCreate - Creates a `PetscDeviceContext`

  Not Collective, Asynchronous

  Output Paramemter:
. dctx - The `PetscDeviceContext`

  Notes:
  Unlike almost every other PETSc class it is advised that most users use
  `PetscDeviceContextDuplicate()` rather than this routine to create new contexts. Contexts of
  different types are incompatible with one another; using `PetscDeviceContextDuplicate()`
  ensures compatible types.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextDuplicate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetStreamType()`, `PetscDeviceContextSetUp()`,
`PetscDeviceContextSetFromOptions()`, `PetscDeviceContextDestroy()`
@*/
PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx) {
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  PetscCall(PetscDeviceInitializePackage());
  PetscCall(contextPool.get(*dctx));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDestroy - Frees a `PetscDeviceContext`

  Not Collective, Asynchronous

  Input Parameters:
. dctx - The `PetscDeviceContext`

  Notes:
  No implicit synchronization occurs due to this routine, all resources are released completely
  asynchronously w.r.t. the host. If one needs to guarantee access to the data produced on this
  contexts stream one should perform the appropriate synchronization before calling this routine.

  Developer Notes:
  The context is never actually "destroyed", only returned to an ever growing pool of
  contexts. There are currently no safeguards on the size of the pool, this should perhaps be
  implemented.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetUp()`, `PetscDeviceContextSynchronize()`
@*/
PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx) {
  PetscFunctionBegin;
  if (!*dctx) PetscFunctionReturn(0);
  // std::move of the expression of the trivially-copyable type 'PetscDeviceContext' (aka
  // '_n_PetscDeviceContext *') has no effect; remove std::move() [performance-move-const-arg]
  // can't remove std::move, since reclaim only takes r-value reference
  PetscCall(contextPool.reclaim(std::move(*dctx))); // NOLINT (performance-move-const-arg)
  *dctx = nullptr;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetStreamType - Set the implementation type of the underlying stream for a
  `PetscDeviceContext`

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The `PetscDeviceContext`
- type - The `PetscStreamType`

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available types
  and their interactions. If the PetscDeviceContext was previously set up and stream type was
  changed, you must call PetscDeviceContextSetUp() again after this routine.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscStreamType`, `PetscDeviceContextGetStreamType()`, `PetscDeviceContextCreate()`,
`PetscDeviceContextSetUp()`, `PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext dctx, PetscStreamType type) {
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change the stream
  // type
  PetscValidDeviceContext(dctx, 1);
  PetscValidStreamType(type, 2);
  // only need to do complex swapping if the object has already been setup
  if (dctx->setup && (dctx->streamType != type)) {
    PetscUseTypeMethod(dctx, changestreamtype, type);
    dctx->setup = PETSC_FALSE;
  }
  dctx->streamType = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetStreamType - Get the implementation type of the underlying stream for a
  `PetscDeviceContext`

  Not Collective, Synchronous

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Output Parameter:
. type - The `PetscStreamType`

  Notes:
  See `PetscStreamType` in include/petscdevicetypes.h for more information on the available types
  and their interactions

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextCreate()`,
`PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext dctx, PetscStreamType *type) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidIntPointer(type, 2);
  *type = dctx->streamType;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetDevice - Set the underlying device for the `PetscDeviceContext`

  Not Collective, Possibly Synchronous

  Input Parameters:
+ dctx   - The `PetscDeviceContext`
- device - The `PetscDevice`

  Notes:
  This routine is effectively `PetscDeviceContext`'s "set-type" (so every `PetscDeviceContext` must
  also have an attached `PetscDevice`). Unlike the usual set-type semantics, it is not stricly
  necessary to set a contexts device to enable usage, any created device contexts will always
  come equipped with the "default" device.

  This routine is a no-op if `dctx` is already attached to `device`.

  This routine may initialize the backend device and incur synchronization.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceCreate()`, `PetscDeviceConfigure()`, `PetscDeviceContextGetDevice()`,
`PetscDeviceContextGetDeviceType()`
@*/
PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext dctx, PetscDevice device) {
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change its device
  // type
  PetscValidDeviceContext(dctx, 1);
  PetscValidDevice(device, 2);
  if (dctx->device) {
    /* can't do a strict pointer equality check since PetscDevice's are reused */
    if (dctx->device->ops->createcontext == device->ops->createcontext) PetscFunctionReturn(0);
  }
  PetscCall(PetscDeviceDestroy(&dctx->device));
  PetscTryTypeMethod(dctx, destroy);
  PetscCall(PetscMemzero(dctx->ops, sizeof(*dctx->ops)));
  PetscCall((*device->ops->createcontext)(dctx));
  PetscCall(PetscDeviceReference_Internal(device));
  dctx->device = device;
  dctx->setup  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetDevice - Get the underlying `PetscDevice` for a `PetscDeviceContext`

  Not Collective, Synchronous

  Input Parameter:
. dctx - the `PetscDeviceContext`

  Output Parameter:
. device - The `PetscDevice`

  Notes:
  This is a borrowed reference, the user should not destroy the device.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextSetDevice()`, `PetscDevice`, `PetscDeviceContextGetDeviceType()`
@*/
PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext dctx, PetscDevice *device) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(device, 2);
  PetscAssert(dctx->device, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt_FMT " has no attached PetscDevice to get", dctx->id);
  *device = dctx->device;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetUp - Prepares a `PetscDeviceContext` for use

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Developer Notes:
  This routine is usually the stage where a `PetscDeviceContext` acquires device-side data
  structures such as streams, events, and (possibly) handles.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextDestroy()`, `PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (dctx->setup) PetscFunctionReturn(0);
  if (!dctx->device) {
    const auto dtype = PETSC_DEVICE_DEFAULT();

    PetscCall(PetscInfo(nullptr, "PetscDeviceContext %" PetscInt_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n", dctx->id, PetscDeviceTypes[dtype]));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, dtype));
  }
  PetscUseTypeMethod(dctx, setup);
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a `PetscDeviceContext` object

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The `PetscDeviceContext` to duplicate

  Output Parameter:
. dctxdup - The duplicated `PetscDeviceContext`

  Notes:
  This is a shorthand method for creating a `PetscDeviceContext` with the exact same settings as
  another. Note however that the duplicated `PetscDeviceContext` does not "share" any of the
  underlying data with the original, (including its current stream-state) they are completely
  separate objects.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetStreamType()`
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup) {
  PetscStreamType stype = PETSC_STREAM_DEFAULT_BLOCKING;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dctxdup, 2);
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextCreate(dctxdup));
  PetscCall(PetscDeviceContextSetStreamType(*dctxdup, stype));
  if (const auto device = dctx->device) PetscCall(PetscDeviceContextSetDevice(*dctxdup, device));
  PetscCall(PetscDeviceContextSetUp(*dctxdup));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextQueryIdle - Returns whether or not a `PetscDeviceContext` is idle

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. idle - `PETSC_TRUE` if `dctx` has NO work, `PETSC_FALSE` if it has work

  Notes:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if `dctx` is idle but has dependents who do have work, this routine still
  returns `PETSC_TRUE`.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextWaitForContext()`, `PetscDeviceContextFork()`
@*/
PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext dctx, PetscBool *idle) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidBoolPointer(idle, 2);
  PetscUseTypeMethod(dctx, query, idle);
  PetscCall(PetscInfo(nullptr, "PetscDeviceContext id %" PetscInt_FMT " %s idle\n", dctx->id, *idle ? "was" : "was not"));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextWaitForContext - Make one context wait for another context to finish

  Not Collective, Asynchronous

  Input Parameters:
+ dctxa - The `PetscDeviceContext` object that is waiting
- dctxb - The `PetscDeviceContext` object that is being waited on

  Notes:
  Serializes two `PetscDeviceContext`s. This routine uses only the state of `dctxb` at the moment
  this routine was called, so any future work queued will not affect `dctxa`. It is safe to pass
  the same context to both arguments.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextQueryIdle()`, `PetscDeviceContextJoin()`
@*/
PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxa));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxb));
  PetscCheckCompatibleDeviceContexts(dctxa, 1, dctxb, 2);
  if (dctxa == dctxb) PetscFunctionReturn(0);
  PetscUseTypeMethod(dctxa, waitforcontext, dctxb);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextFork - Create a set of dependent child contexts from a parent context

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The parent `PetscDeviceContext`
- n    - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Notes:
  This routine creates `n` edges of a DAG from a source node which are causally dependent on the
  source node, meaning that work queued on child contexts will not start until the parent
  context finishes its work. This accounts for work queued on the parent up until calling this
  function, any subsequent work enqueued on the parent has no effect on the children.

  Any children created with this routine have their lifetimes bounded by the parent. That is,
  the parent context expects to free all of it's children (and ONLY its children) before itself
  is freed.

  DAG representation:
.vb
  time ->

  -> dctx \----> dctx ------>
           \---> dsub[0] --->
            \--> ... ------->
             \-> dsub[n-1] ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextJoin()`, `PetscDeviceContextSynchronize()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub) {
#if PETSC_USE_DEBUG_AND_INFO
  const auto         nBefore = n;
  static std::string idList;
#endif
  PetscDeviceContext *dsubTmp = nullptr;
  PetscInt            i       = 0;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dsub, 3);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts requested %" PetscInt_FMT " < 0", n);
#if PETSC_USE_DEBUG_AND_INFO
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  PetscCallCXX(idList.reserve(4 * n));
#endif
  /* update child totals */
  dctx->numChildren += n;
  /* now to find out if we have room */
  if (dctx->numChildren > dctx->maxNumChildren) {
    const auto numChildren    = dctx->numChildren;
    auto      &maxNumChildren = dctx->maxNumChildren;
    /* no room, either from having too many kids or not having any */
    if (auto &childIDs = dctx->childIDs) {
      /* have existing children, must reallocate them */
      PetscCall(PetscRealloc(numChildren * sizeof(*childIDs), &childIDs));
      /* clear the extra memory since realloc doesn't do it for us */
      PetscCall(PetscArrayzero(std::next(childIDs, maxNumChildren), numChildren - maxNumChildren));
    } else {
      /* have no children */
      PetscCall(PetscCalloc1(numChildren, &childIDs));
    }
    /* update total number of children */
    maxNumChildren = numChildren;
  }
  PetscCall(PetscMalloc1(n, &dsubTmp));
  while (n) {
    auto &childID = dctx->childIDs[i];
    /* empty child slot */
    if (!childID) {
      /* create the child context in the image of its parent */
      PetscCall(PetscDeviceContextDuplicate(dctx, dsubTmp + i));
      PetscCall(PetscDeviceContextWaitForContext(dsubTmp[i], dctx));
      /* register the child with its parent */
      childID = dsubTmp[i]->id;
#if PETSC_USE_DEBUG_AND_INFO
      PetscCallCXX(idList += std::to_string(dsubTmp[i]->id));
      if (n != 1) PetscCallCXX(idList += ", ");
#endif
      --n;
    }
    ++i;
  }
#if PETSC_USE_DEBUG_AND_INFO
  PetscCall(PetscInfo(nullptr, "Forked %" PetscInt_FMT " children from parent %" PetscInt_FMT " with IDs: %s\n", nBefore, dctx->id, idList.c_str()));
  /* resets the size but doesn't deallocate the memory */
  idList.clear();
#endif
  /* pass the children back to caller */
  *dsub = dsubTmp;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextJoin - Converge a set of child contexts

  Not Collective, Asynchronous

  Input Parameters:
+ dctx         - A `PetscDeviceContext` to converge on
. n            - The number of sub contexts to converge
. joinMode     - The type of join to perform
- dsub         - The sub contexts to converge

  Notes:
  If `PetscDeviceContextFork()` creates `n` edges from a source node which all depend on the source
  node, then this routine is the exact mirror. That is, it creates a node (represented in dctx)
  which recieves n edges (and optionally destroys them) which is dependent on the completion of
  all incoming edges.

  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY` all contexts in `dsub` will be destroyed by
  this routine. Thus all sub contexts must have been created with the `dctx` passed to this
  routine.

  if `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC` `dctx` waits for all sub contexts but the sub
  contexts do not wait for one another afterwards.

  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC` all sub contexts will additionally wait on `dctx`
  after converging. This has the effect of "synchronizing" the outgoing edges.

  DAG representations:
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY`
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/
  ->  ... -------/
  -> dsub[n-1] -/
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC`
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/--------->
  ->  ... -------/---------->
  -> dsub[n-1] -/----------->
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC`
.vb
  time ->

  -> dctx ---------/- dctx -\----> dctx ------>
  -> dsub[0] -----/          \---> dsub[0] --->
  ->  ... -------/            \--> ... ------->
  -> dsub[n-1] -/              \-> dsub[n-1] ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextSynchronize()`, `PetscDeviceContextJoinMode`
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub) {
#if PETSC_USE_DEBUG_AND_INFO
  static std::string idList;
#endif

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* validity of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub, 4);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts merged %" PetscInt_FMT " < 0", n);
#if PETSC_USE_DEBUG_AND_INFO
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  PetscCallCXX(idList.reserve(4 * n));
#endif
  /* first dctx waits on all the incoming edges */
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx, 1, (*dsub)[i], 4);
    PetscCall(PetscDeviceContextWaitForContext(dctx, (*dsub)[i]));
#if PETSC_USE_DEBUG_AND_INFO
    PetscCallCXX(idList += std::to_string((*dsub)[i]->id));
    if (i + 1 < n) PetscCallCXX(idList += ", ");
#endif
  }

  /* now we handle the aftermath */
  switch (joinMode) {
  case PETSC_DEVICE_CONTEXT_JOIN_DESTROY: {
    PetscInt j = 0;

    PetscCheck(n <= dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Trying to destroy %" PetscInt_FMT " children of a parent context that only has %" PetscInt_FMT " children, likely trying to restore to wrong parent", n, dctx->numChildren);
    /* update child count while it's still fresh in memory */
    dctx->numChildren -= n;
    for (PetscInt i = 0; i < dctx->maxNumChildren; ++i) {
      if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
        /* child is one of ours, can destroy it */
        PetscCall(PetscDeviceContextDestroy((*dsub) + j));
        /* reset the child slot */
        dctx->childIDs[i] = 0;
        if (++j == n) break;
      }
    }
    /* gone through the loop but did not find every child */
    PetscCheck(j == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT " contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checked out out in", n - j);
    PetscCall(PetscFree(*dsub));
  } break;
  case PETSC_DEVICE_CONTEXT_JOIN_SYNC:
    for (PetscInt i = 0; i < n; ++i) PetscCall(PetscDeviceContextWaitForContext((*dsub)[i], dctx));
  case PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC: break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown PetscDeviceContextJoinMode given");
  }

#if PETSC_USE_DEBUG_AND_INFO
  PetscCall(PetscInfo(nullptr, "Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt_FMT ", mode %s with IDs: %s\n", n, dctx->id, PetscDeviceContextJoinModes[joinMode], idList.c_str()));
  idList.clear();
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize - Block the host until all work queued on or associated with a
  `PetscDeviceContext` has finished

  Not Collective, Synchronous

  Input Parameters:
. dctx - The `PetscDeviceContext` to synchronize

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) PetscUseTypeMethod(dctx, synchronize);
  PetscFunctionReturn(0);
}

// since the pointers allocated via PetscDeviceAllocate() may be device pointers we cannot just
// store meta-data within the pointer itself (as we can't dereference them). So instead we need
// to keep an extra map to keep track of them
static struct MemoryMap {
  using map_type = std::unordered_map<void *, PetscMemType>;

  map_type map;
  bool     registered = false;
} memory_map;

/*@C
  PetscDeviceAllocate - Allocate device-aware memory

  Not Collective, Possibly Synchronous

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to allocate the memory
. clear - Whether or not the memory should be zeroed
. mtype - The type of memory to allocate
- n     - The amount (in bytes) to allocate

  Output Parameter:
. ptr - The pointer to store the result in

  Notes:
  The user should prefer `PetscDeviceMalloc()` over this routine as it automatically computes
  the size of the allocation based on the size of the datatype.

  Memory allocated with this function must be freed with `PetscDeviceFree()` or
  `PetscDeviceDeallocate()`.

  If the `PetscDeviceContext` backend supports asynchronous allocations (such as CUDA 11+) this
  routine is asynchronous.

  If `n` is zero, then `ptr` is set to `NULL`.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscGetMemType()`, `PetscDeviceFree()`,
`PetscDeviceDeallocate()`, `PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscDeviceAllocate(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, size_t n, void **PETSC_RESTRICT ptr) {
  PetscFunctionBegin;
  PetscValidPointer(ptr, 5);
  if (PetscLikely(n)) {
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall((*dctx->ops->memalloc)(dctx, clear, mtype, n, ptr));
    {
      const auto ret = memory_map.map.emplace(*ptr, mtype);

      // we previously allocated the pointer (with some memtype), but now emplace has failed
      // and the new mtype doesn't match. In practice this shouldn't happen, since that
      // indicates that e.g. cudaMalloc() and cudaMallocHost() have returned identical
      // pointers, but it doesn't hurt to check
      PetscAssert(ret.second || (mtype == ret.first->second), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to overwrite previously allocated pointer %p (of memtype %s) with new memtype %s", ret.first->first, PetscMemTypes(ret.first->second), PetscMemTypes(mtype));
    }
    if (PetscUnlikely(!memory_map.registered)) {
      const auto finalizer = [] {
        PetscFunctionBegin;
        memory_map.map.clear();
        memory_map.registered = false;
        PetscFunctionReturn(0);
      };

      memory_map.registered = true;
      PetscCall(PetscRegisterFinalize(std::move(finalizer)));
    }
  } else {
    *ptr = nullptr;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDeallocate - Free device-aware memory

  Not Collective, Possibly Synchronous

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to free the memory
- ptr   - The pointer to free

  Notes:
  The user should prefer `PetscDeviceFree()` over this routine as it automatically sets `ptr`
  to `PETSC_NULLPTR` on successful deallocation.

  `ptr` must have been allocated using either `PetscDeviceMalloc()`, `PetscDeviceCalloc()` or
  `PetscDeviceAllocate()`.

  If the `PetscDeviceContext` backend supports asynchronous deallocations (such as CUDA 11+)
  this routine is asynchronous.

  `ptr` may be `NULL`.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceAllocate()`
@*/
PetscErrorCode PetscDeviceDeallocate(PetscDeviceContext dctx, void *PETSC_RESTRICT ptr) {
  PetscFunctionBegin;
  if (ptr) {
    auto      &map   = memory_map.map;
    const auto found = map.find(const_cast<void *>(ptr));
    static_assert(std::is_same<MemoryMap::map_type::key_type, void *>::value, "");

    PetscAssert(found != map.end(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Pointer %p was not allocated via PetscDeviceAllocate()", ptr);
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall((*dctx->ops->memfree)(dctx, found->second, ptr));
    PetscCallCXX(map.erase(found));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceMemcpy - Copy memory in a device-aware manner

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to copy the memory
. dest - The pointer to copy to
. src  - The pointer to copy from
. n    - The amount (in bytes) to copy
- mode - The direction in which to copy

  Notes:
  The user should prefer `PetscDeviceArrayCopy()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types.

  `src` and `dest` cannot overlap

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceArrayCopy()`, `PetscDeviceMalloc()`, `PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode) {
  PetscFunctionBegin;
  PetscAssert(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
  PetscAssert(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
  if (PetscDefined(USE_DEBUG)) {
    const auto CheckMemType = [&](const void *ptr, const char name[], bool on_host) {
      PetscMemType mtype;

      PetscFunctionBegin;
      if (mode == PETSC_DEVICE_COPY_AUTO) PetscFunctionReturn(0);
      PetscCall(PetscGetMemType(ptr, &mtype));
      if ((mode == PETSC_DEVICE_COPY_HTOH) || on_host) {
        // ptr should be on the host
        PetscValidPointer(ptr, 1);
        PetscCheck(PetscMemTypeHost(mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PetscDeviceCopyMode %s implies %s is host memory but have %s instead", PetscDeviceCopyModes[mode], name, PetscMemTypes(mtype));
      } else {
        // ptr should be on the device
        PetscCheck(PetscMemTypeDevice(mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PetscDeviceCopyMode %s implies %s is device memory but have %s instead", PetscDeviceCopyModes[mode], name, PetscMemTypes(mtype));
      }
      PetscFunctionReturn(0);
    };
    const auto src_end  = std::next(static_cast<const char *>(src), n);
    const auto dest_end = std::next(static_cast<const char *>(dest), n);

    PetscCheck((dest >= src_end) || (dest_end <= src), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "dest [%p, %p) cannot overlap with src [%p, %p)", dest, dest_end, src, src_end);
    PetscCall(CheckMemType(src, "source", mode == PETSC_DEVICE_COPY_HTOD));
    PetscCall(CheckMemType(dest, "dest", mode == PETSC_DEVICE_COPY_DTOH));
  }
  if (PetscLikely(n)) {
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall((*dctx->ops->memcopy)(dctx, dest, src, n, mode));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceMemset - Memset device-aware memory

  Not Collective, Asynchronous

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to memset the memory
. mtype - The type of memory to memset
. dest  - The pointer to the memory
. v     - The value to set
- n     - The amount (in bytes) to set

  Notes:
  The user should prefer `PetscDeviceArrayZero()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types, though they should
  note that it only zeros memory.

  This routine is analogous to `memset()`. That is, this routine copies the value
  `static_cast<unsigned char>(v)` into each of the first count characters of the object pointed
  to by `dest`.

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceArrayZero()`, `PetscDeviceMalloc()`, `PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemset(PetscDeviceContext dctx, PetscMemType mtype, void *dest, PetscInt v, std::size_t n) {
  PetscFunctionBegin;
  PetscAssert(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer");
  if (PetscMemTypeHost(mtype)) PetscValidPointer(dest, 3);
  if (PetscLikely(n)) {
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall((*dctx->ops->memset)(dctx, mtype, dest, v, n));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetDeviceType - Get the `PetscDeviceType` for a `PetscDeviceContext`

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Output Parameter:
. type - The `PetscDeviceType`

  Notes:
  This routine is a convenience shorthand for `PetscDeviceContextGetDevice()` ->
  `PetscDeviceGetType()`.

  Level: beginner

.seealso: `PetscDeviceType`, `PetscDeviceContextGetDevice()`, `PetscDeviceGetType()`, `PetscDevice`
@*/
PetscErrorCode PetscDeviceContextGetDeviceType(PetscDeviceContext dctx, PetscDeviceType *type) {
  PetscDevice device = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(type, 2);
  PetscCall(PetscDeviceContextGetDevice(dctx, &device));
  PetscCall(PetscDeviceGetType(device, type));
  PetscFunctionReturn(0);
}

/* every device type has a vector of null PetscDeviceContexts -- one for each device */
static auto nullContexts          = std::array<std::vector<PetscDeviceContext>, PETSC_DEVICE_MAX>{};
static auto nullContextsFinalizer = false;

PetscErrorCode PetscDeviceContextGetNullContextForDevice_Internal(PetscDevice device, PetscDeviceContext *dctx) {
  PetscInt        devid;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscValidDevice(device, 1);
  PetscValidPointer(dctx, 2);
  if (PetscUnlikely(!nullContextsFinalizer)) {
    const auto finalizer = [] {
      PetscFunctionBegin;
      for (auto &dvec : nullContexts) {
        for (auto dctx : dvec) PetscCall(PetscDeviceContextDestroy(&dctx));
        PetscCallCXX(dvec.clear());
      }
      nullContextsFinalizer = false;
      PetscFunctionReturn(0);
    };

    nullContextsFinalizer = true;
    PetscCall(PetscRegisterFinalize(finalizer));
  }
  PetscCall(PetscDeviceGetDeviceId(device, &devid));
  PetscCall(PetscDeviceGetType(device, &dtype));
  {
    auto &ctxlist = nullContexts[dtype];

    PetscCheck(devid >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Device ID (%" PetscInt_FMT ") must be positive", devid);
    // need to resize the container if not big enough because incrementing the iterator in
    // std::next() (if we haven't initialized that ctx yet) may cause it to fall outside the
    // current size of the container.
    if (static_cast<std::size_t>(devid) >= ctxlist.size()) CHKERRCXX(ctxlist.resize(devid + 1));
    if (PetscUnlikely(!ctxlist[devid])) {
      // we have not seen this device before
      PetscCall(PetscInfo(nullptr, "Initializing null PetscDeviceContext for device %" PetscInt_FMT "\n", devid));
      PetscCall(PetscDeviceContextCreate(dctx));
      PetscCall(PetscDeviceContextSetStreamType(*dctx, PETSC_STREAM_GLOBAL_BLOCKING));
      PetscCall(PetscDeviceContextSetDevice(*dctx, device));
      PetscCall(PetscDeviceContextSetUp(*dctx));
      // would use ctxlist.cbegin() but GCC 4.8 can't handle const iterator insert!
      CHKERRCXX(ctxlist.insert(std::next(ctxlist.begin(), devid), *dctx));
    } else *dctx = ctxlist[devid];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *dctx) {
  PetscDeviceContext gctx = nullptr;
  PetscDevice        gdev = nullptr;

  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  PetscCall(PetscDeviceContextGetCurrentContext(&gctx));
  PetscCall(PetscDeviceContextGetDevice(gctx, &gdev));
  PetscCall(PetscDeviceContextGetNullContextForDevice_Internal(gdev, dctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextIsNullContext(PetscDeviceContext dctx, PetscBool *isNull) {
  PetscFunctionBegin;
  PetscValidBoolPointer(isNull, 2);
  if (dctx) {
    PetscDevice     device;
    PetscDeviceType dtype;

    PetscValidDeviceContext(dctx, 1);
    PetscCall(PetscDeviceContextGetDevice(dctx, &device));
    PetscCall(PetscDeviceGetType(device, &dtype));
    {
      PetscInt    id;
      const auto &ctxlist = nullContexts[dtype];

      PetscCall(PetscDeviceGetDeviceId(device, &id));
      if (static_cast<std::size_t>(id) < ctxlist.size()) {
        const auto end = ctxlist.cend();

        *isNull = static_cast<PetscBool>(std::find(ctxlist.cbegin(), end, dctx) != end);
      } else {
        // out of bounds? not one of ours
        *isNull = PETSC_FALSE;
      }
    }
  } else {
    *isNull = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#define PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE PETSC_DEVICE_INITIAL_DEFAULT_TYPE
// REMOVE ME (change)
#define PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE PETSC_STREAM_GLOBAL_BLOCKING

static PetscDeviceType    rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
static PetscStreamType    rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
static PetscDeviceContext globalContext  = nullptr;

/* when PetsDevice initializes PetscDeviceContext eagerly the type of device created should
 * match whatever device is eagerly intialized */
PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType type) {
  PetscFunctionBegin;
  PetscValidDeviceType(type, 1);
  rootDeviceType = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType type) {
  PetscFunctionBegin;
  PetscValidStreamType(type, 1);
  rootStreamType = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSetupGlobalContext_Private() {
  static const auto PetscDeviceContextFinalizer = [] {
    PetscFunctionBegin;
    PetscCall(PetscDeviceContextDestroy(&globalContext));
    rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
    rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
  if (PetscLikely(globalContext)) PetscFunctionReturn(0);
  /* this exists purely as a valid device check. */
  PetscCall(PetscDeviceInitializePackage());
  PetscCall(PetscRegisterFinalize(PetscDeviceContextFinalizer));
  PetscCall(PetscInfo(nullptr, "Initializing global PetscDeviceContext\n"));
  /* we call the allocator directly here since the ObjectPool creates a PetscContainer which
   * eventually tries to call logging functions. However, this routine may be purposefully
   * called __before__ logging is initialized, so the logging function would PETSCABORT */
  PetscCall(contextPool.allocator().create(&globalContext));
  PetscCall(PetscDeviceContextSetStreamType(globalContext, rootStreamType));
  PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(globalContext, PETSC_DEVICE_DEFAULT()));
  PetscCall(PetscDeviceContextSetUp(globalContext));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetCurrentContext - Get the current active `PetscDeviceContext`

  Not Collective, Asynchronous

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Notes:
  The user generally should not destroy contexts retrieved with this routine unless they
  themselves have created them. There exists no protection against destroying the root
  context.

  Developer Notes:
  Unless the user has set their own, this routine creates the "root" context the first time it
  is called, registering its destructor to `PetscFinalize()`.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextSetCurrentContext()`, `PetscDeviceContextFork()`,
          `PetscDeviceContextJoin()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx) {
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  PetscCall(PetscDeviceContextSetupGlobalContext_Private());
  /* while the static analyzer can find global variables, it will throw a warning about not
   * being able to connect this back to the function arguments */
  PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidDeviceContext(globalContext, -1));
  *dctx = globalContext;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetCurrentContext - Set the current active `PetscDeviceContext`

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Notes:
  This routine can be used to set the defacto "root" `PetscDeviceContext` to a user-defined
  implementation by calling this routine immediately after `PetscInitialize()` and ensuring that
  `PetscDevice` is not greedily intialized. In this case the user is responsible for destroying
  their `PetscDeviceContext` before `PetscFinalize()` returns.

  The old context is not stored in any way by this routine; if one is overriding a context that
  they themselves do not control, one should take care to temporarily store it by calling
  `PetscDeviceContextGetCurrentContext()` before calling this routine.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextGetCurrentContext()`, `PetscDeviceContextFork()`,
          `PetscDeviceContextJoin()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx) {
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(dctx->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt_FMT " must be set up before being set as global context", dctx->id);
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCall(PetscDeviceSetDefaultDeviceType(dtype));
  globalContext = dctx;
  PetscCall(PetscInfo(nullptr, "Set global PetscDeviceContext id %" PetscInt_FMT "\n", dctx->id));
  PetscFunctionReturn(0);
}

/*
  needed because PetscInitialize() needs to also query these options to set the defaults. Since
  it does not yet have a PetscDeviceContext to call this with, the actual options queries are
  abstracted out, so you can call this without one.
*/
PetscErrorCode PetscDeviceContextQueryOptions_Internal(MPI_Comm comm, const char prefix[], std::pair<PetscDeviceType, PetscBool> &deviceType, std::pair<PetscStreamType, PetscBool> &streamType) {
  auto dtype = static_cast<PetscInt>(deviceType.first);
  auto stype = static_cast<PetscInt>(streamType.first);

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix, 2);
  PetscOptionsBegin(comm, prefix, "PetscDeviceContext Options", "Sys");
  /* set the device type first */
  PetscCall(PetscOptionsEList("-device_context_device_type", "Underlying PetscDevice", "PetscDeviceContextSetDevice", PetscDeviceTypes, PETSC_DEVICE_MAX, PetscDeviceTypes[dtype], &dtype, &deviceType.second));
  PetscCall(PetscOptionsEList("-device_context_stream_type", "PetscDeviceContext PetscStreamType", "PetscDeviceContextSetStreamType", PetscStreamTypes, PETSC_STREAM_MAX, PetscStreamTypes[stype], &stype, &streamType.second));
  PetscOptionsEnd();
  deviceType.first = PetscDeviceTypeCast(dtype);
  streamType.first = PetscStreamTypeCast(stype);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetFromOptions - Configure a `PetscDeviceContext` from the options database

  Collective on comm, Asynchronous

  Input Parameters:
+ comm   - MPI communicator on which to query the options database
. prefix - prefix to prepend to all options database queries, `NULL` if not needed
- dctx   - The `PetscDeviceContext` to configure

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Options Database:
+ -device_context_stream_type - type of stream to create inside the `PetscDeviceContext` -
   `PetscDeviceContextSetStreamType()`
- -device_context_device_type - the type of `PetscDevice` to attach by default - `PetscDeviceType`

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextSetDevice()`
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, const char prefix[], PetscDeviceContext dctx) {
  auto dtype = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE, PETSC_FALSE);
  auto stype = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE, PETSC_FALSE);

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix, 2);
  PetscValidDeviceContext(dctx, 3);
  /* set the device type first */
  if (auto device = dctx->device) PetscCall(PetscDeviceGetType(device, &dtype.first));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype.first));
  PetscCall(PetscDeviceContextQueryOptions_Internal(comm, prefix, dtype, stype));
  if (dtype.second) PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, dtype.first));
  if (stype.second) PetscCall(PetscDeviceContextSetStreamType(dctx, stype.first));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscFunctionReturn(0);
}
