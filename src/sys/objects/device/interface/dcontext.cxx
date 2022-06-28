#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

#include <tuple>
#include <array>
#include <utility>
#include <vector>
#include <algorithm> // std::find
#include <unordered_set>
#include <unordered_map>
#include <type_traits> // std::is_same

// debugging aides
#include <sstream>
#include <string>
#if PetscDefined(USE_DEBUG) && PetscDefined(USE_INFO)
#define PETSC_USE_DEBUG_AND_INFO 1
#endif

const char *const PetscStreamTypes[] = {"global_blocking", "default_blocking", "global_nonblocking", "max", "PetscStreamType", "PETSC_STREAM_", nullptr};

const char *const PetscDeviceContextJoinModes[] = {"destroy", "sync", "no_sync", "PetscDeviceContextJoinMode", "PETSC_DEVICE_CONTEXT_JOIN_", nullptr};

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_AUTO) == 4, "");

struct PetscDeviceContextCxxData {
  struct ParentSet {
    using key_type  = PetscDeviceContext;
    using hash_type = std::hash<key_type>;

    struct cmp_type {
      bool operator()(const key_type &l, const key_type &r) const noexcept { return l->id == r->id; }
    };

    using data_type = std::unordered_set<key_type, hash_type, cmp_type>;
  };

  using parent_type = ParentSet::data_type;

  parent_type parents;

  PETSC_NODISCARD PetscErrorCode clear() noexcept;
};

PetscErrorCode PetscDeviceContextCxxData::clear() noexcept {
  PetscFunctionBegin;
  this->parents.clear();
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DECL(auto CxxDataCast(PetscDeviceContext dctx))
PETSC_DECLTYPE_AUTO_RETURNS(static_cast<PetscDeviceContextCxxData *>(dctx->cxxdata))

/* Define the allocator */
struct PetscDeviceContextAllocator : Petsc::AllocatorBase<PetscDeviceContext> {
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext *dctx)) {
    PetscFunctionBegin;
    PetscCall(PetscNew(dctx));
    PetscCallCXX((*dctx)->cxxdata = new PetscDeviceContextCxxData);
    PetscCall(reset(*dctx));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying", dctx->numChildren);
    PetscTryTypeMethod(dctx, destroy);
    PetscCall(PetscDeviceDestroy(&dctx->device));
    PetscCall(PetscFree(dctx->childIDs));
    delete CxxDataCast(dctx);
    PetscCall(PetscFree(dctx));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    if (const auto destroy = dctx->ops->destroy) PetscCall((*destroy)(dctx));
    PetscCall(PetscDeviceDestroy(&dctx->device));
    // don't deallocate the child array, rather just zero it out
    PetscCall(PetscArrayzero(dctx->childIDs, dctx->maxNumChildren));
    dctx->numChildren           = 0;
    dctx->setup                 = PETSC_FALSE;
    dctx->contained             = PETSC_FALSE;
    dctx->streamType            = PETSC_STREAM_DEFAULT_BLOCKING;
    dctx->options.allow_orphans = PetscDefined(USE_DEBUG) ? PETSC_FALSE : PETSC_TRUE;
    PetscCall(CxxDataCast(dctx)->clear());
    PetscCall(PetscObjectNewId(&dctx->id));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

static Petsc::ObjectPool<PetscDeviceContext, PetscDeviceContextAllocator> contextPool;

/*@C
  PetscDeviceContextCreate - Creates a `PetscDeviceContext`

  Not Collective

  Output Paramemter:
. dctx - The `PetscDeviceContext`

  Notes:
  Unlike almost every other PETSc class it is advised that most users use
  `PetscDeviceContextDuplicate()` rather than this routine to create new contexts. Contexts of
  different types are incompatible with one another; using `PetscDeviceContextDuplicate()`
  ensures compatible types.

  Sequential Consistency Notes:
  Sequentially Consistent on the returned `dctx`

  DAG representation:
.vb
  time ->

  |= CALL =| - dctx ->
.ve

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

static PetscErrorCode PetscDeviceContextCheckNotOrphaned(PetscDeviceContext);
/*@C
  PetscDeviceContextDestroy - Frees a `PetscDeviceContext`

  Not Collective

  Input Parameters:
. dctx - The `PetscDeviceContext`

  Notes:
  No implicit synchronization occurs due to this routine, all resources are released completely
  asynchronously w.r.t. the host. If one needs to guarantee access to the data produced on this
  contexts stream one should perform the appropriate synchronization before calling this routine.

  Sequential Consistency Notes:
  Sequentially Consistent on the returned `dctx`.

  DAG represetation:
.vb
  time ->

  -> dctx - |= CALL =|
.ve

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
  PetscCall(PetscDeviceContextCheckNotOrphaned(*dctx));
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

  Not Collective

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

  Not Collective

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

  Not Collective

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

  Not Collective

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
  PetscAssert(dctx->device, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " has no attached PetscDevice to get", dctx->id);
  *device = dctx->device;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetUp - Prepares a `PetscDeviceContext` for use

  Not Collective

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

    PetscCall(PetscInfo(nullptr, "PetscDeviceContext %" PetscInt64_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n", dctx->id, PetscDeviceTypes[dtype]));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, dtype));
  }
  PetscUseTypeMethod(dctx, setup);
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a `PetscDeviceContext` object

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext` to duplicate

  Output Parameter:
. dctxdup - The duplicated `PetscDeviceContext`

  Notes:
  This is a shorthand method for creating a `PetscDeviceContext` with the exact same settings as
  another. Note however that the duplicated `PetscDeviceContext` does not "share" any of the
  underlying data with the original, (including its current stream-state) they are completely
  separate objects.

  Sequential Consistency Notes:
  Sequentially Consistent on the input `dctx`. There is no implied ordering between `dctx` or
  `dctxdup`.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ---->
                       - dctxdup ->
.ve

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetStreamType()`
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup) {
  auto stype = PETSC_STREAM_DEFAULT_BLOCKING;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dctxdup, 2);
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextCreate(dctxdup));
  PetscCall(PetscDeviceContextSetStreamType(*dctxdup, stype));
  if (const auto device = dctx->device) PetscCall(PetscDeviceContextSetDevice(*dctxdup, device));
  (*dctxdup)->options = dctx->options;
  PetscCall(PetscDeviceContextSetUp(*dctxdup));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextQueryIdle - Returns whether or not a `PetscDeviceContext` is idle

  Not Collective

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. idle - `PETSC_TRUE` if `dctx` has NO work, `PETSC_FALSE` if it has work

  Notes:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if `dctx` is idle but has dependents who do have work, this routine still
  returns `PETSC_TRUE`.

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`.

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

  Not Collective

  Input Parameters:
+ dctxa - The `PetscDeviceContext` object that is waiting
- dctxb - The `PetscDeviceContext` object that is being waited on

  Notes:
  Serializes two `PetscDeviceContext`s. This routine uses only the state of `dctxb` at the moment
  this routine was called, so any future work queued will not affect `dctxa`. It is safe to pass
  the same context to both arguments (in which case this routine does nothing).

  Sequential Consistency Notes:
  Partially Sequentially Consistent on `dctxa`.

  DAG representation:
.vb
  time ->

  -> dctxa ---/- |= CALL =| - dctxa ->
             /
  -> dctxb -/------------------------>
.ve

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
  PetscCallCXX(CxxDataCast(dctxa)->parents.emplace(dctxb));
  dctxb->contained = PETSC_TRUE; // dctxb is now contained by dctxa
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextFork - Create a set of dependent child contexts from a parent context

  Not Collective

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

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\----> dctx ------>
                         \---> dsub[0] --->
                          \--> ... ------->
                           \-> dsub[n-1] ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextJoin()`, `PetscDeviceContextSynchronize()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub) {
  // debugging only
  std::string idList;
  auto        ninput = n;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dsub, 3);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts requested %" PetscInt_FMT " < 0", n);
  *dsub = nullptr;
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCallCXX(idList.reserve(4 * n));
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
  PetscCall(PetscMalloc1(n, dsub));
  for (PetscInt i = 0; ninput && i < dctx->numChildren; ++i) {
    auto &childID = dctx->childIDs[i];
    /* empty child slot */
    if (!childID) {
      /* create the child context in the image of its parent */
      PetscCall(PetscDeviceContextDuplicate(dctx, (*dsub) + i));
      PetscCall(PetscDeviceContextWaitForContext((*dsub)[i], dctx));
      /* register the child with its parent */
      childID = (*dsub)[i]->id;
      if (PetscDefined(USE_DEBUG_AND_INFO)) {
        PetscCallCXX(idList += std::to_string(childID));
        if (ninput != 1) PetscCallCXX(idList += ", ");
      }
      --ninput;
    }
  }
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCall(PetscInfo(nullptr, "Forked %" PetscInt_FMT " children from parent %" PetscInt64_FMT " with IDs: %s\n", ninput, dctx->id, idList.c_str()));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextJoin - Converge a set of child contexts

  Not Collective

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

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx` if `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY`. All
  contexts in `dsub` will be destroyed by this routine. Thus all sub contexts must have been
  created with the `dctx` passed to this routine.

  Sequentially Consistent on `dctx` if `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC`. All sub
  contexts will additionally wait on `dctx` after converging. This has the effect of
  "synchronizing" the outgoing edges.

  Partially Sequentially Consistent on `dctx` if `joinMode` is
  `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC`. `dctx` waits for all sub contexts but the sub contexts
  do not wait for one another or `dctx` afterwards.

  DAG representations:
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY`
.vb
  time ->

  -> dctx ---------/- |= CALL =| - dctx ->
  -> dsub[0] -----/
  ->  ... -------/
  -> dsub[n-1] -/
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC`
.vb
  time ->

  -> dctx ---------/- |= CALL =| -\----> dctx ------>
  -> dsub[0] -----/                \---> dsub[0] --->
  ->  ... -------/                  \--> ... ------->
  -> dsub[n-1] -/                    \-> dsub[n-1] ->
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC`
.vb
  time ->

  -> dctx ----------/- |= CALL =| - dctx ->
  -> dsub[0] ------/----------------------->
  ->  ... --------/------------------------>
  -> dsub[n-1] --/------------------------->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextSynchronize()`, `PetscDeviceContextJoinMode`
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub) {
  // debugging only
  std::string idList;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* validity of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub, 4);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts merged %" PetscInt_FMT " < 0", n);
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCallCXX(idList.reserve(4 * n));
  /* first dctx waits on all the incoming edges */
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx, 1, (*dsub)[i], 4);
    PetscCall(PetscDeviceContextWaitForContext(dctx, (*dsub)[i]));
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      PetscCallCXX(idList += std::to_string((*dsub)[i]->id));
      if (i + 1 < n) PetscCallCXX(idList += ", ");
    }
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

  if (PetscDefined(USE_DEBUG_AND_INFO)) {
    PetscCall(PetscInfo(nullptr, "Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt64_FMT ", mode %s with IDs: %s\n", n, dctx->id, PetscDeviceContextJoinModes[joinMode], idList.c_str()));
    idList.clear();
  }
  PetscFunctionReturn(0);
}

template <bool use_debug>
struct PetscStackFrame;

template <>
struct PetscStackFrame</* use_debug = */ true> {
  std::string file;
  std::string function;
  int         line;

  PetscStackFrame(const char *file_, const char *func_, int line_) : file(split_on_petsc_path_(file_)), function(func_), line(line_) { }

  friend std::ostream &operator<<(std::ostream &, const PetscStackFrame &);

private:
  static std::string split_on_petsc_path_(std::string &&in) {
    auto pos = in.find("petsc/src");

    if (pos == std::string::npos) pos = in.find("petsc/include");
    if (pos == std::string::npos) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find substring in %s", in.c_str());
    return in.substr(pos);
  }
};

std::ostream &operator<<(std::ostream &os, const PetscStackFrame<true> &frame) {
  os << '(' << frame.function << "() at " << frame.file << ':' << frame.line << ')';
  return os;
}

template <>
struct PetscStackFrame</* use_debug = */ false> {
  template <typename... T>
  constexpr PetscStackFrame(T &&...) noexcept { }

  friend std::ostream &operator<<(std::ostream &, const PetscStackFrame &) noexcept;
};

std::ostream &operator<<(std::ostream &os, const PetscStackFrame<false> &) noexcept {
  return os;
}

struct MarkedObject {
  using frame_type = PetscStackFrame<PetscDefined(USE_DEBUG)>;

  PetscDeviceContext    ctx;
  PetscObjectId         id;
  PetscMemoryAccessMode mode;
  frame_type            frame;

  MarkedObject(PetscDeviceContext ctx, PetscObjectId id, PetscMemoryAccessMode mode, frame_type frame) noexcept : ctx(ctx), id(id), mode(mode), frame(std::move(frame)) { }

  MarkedObject(PetscDeviceContext ctx, PetscObjectId id, PetscMemoryAccessMode mode, const char *file, const char *function, int line) noexcept : MarkedObject(ctx, id, mode, frame_type{file, function, line}) { }
};

// a helper to enumerate the types
struct MarkedObjectMap : Petsc::RegisterFinalizeable<MarkedObjectMap> {
  using key_type    = PetscObjectId;
  using value_type  = MarkedObject;
  using mapped_type = std::vector<value_type>;
  using map_type    = std::unordered_map<key_type, mapped_type>;

  map_type map;

  struct FindContext {
    const PetscObjectId id;

    constexpr FindContext(PetscObjectId id_) noexcept : id(id_) { }
    constexpr FindContext(PetscDeviceContext obj) noexcept : FindContext(obj->id) { }

    bool operator()(mapped_type::const_reference pair) const noexcept { return pair.id == id; }
  };

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;
};

PetscErrorCode MarkedObjectMap::finalize_() noexcept {
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    std::ostringstream oss;
    auto               wrote_to_oss = false;
    const auto         end          = this->map.cend();
    PetscMPIInt        rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    for (auto it = this->map.cbegin(); it != end; ++it) {
      // need a temporary since we want to prepend "object xxx has orphaned dependencies" if
      // any of the dependencies have orphans. but we also need to check that in the loop, so
      // use a temporary to accumulate and then build the rest from it.
      std::ostringstream oss_tmp;
      auto               wrote_to_oss_tmp = false;
      auto             &&dependencies     = it->second;

      for (auto &&dep : dependencies) {
        auto &&ctx = dep.ctx;

        if (!ctx->options.allow_orphans) {
          oss_tmp << "  [" << rank << "] dctx " << ctx << " (id " << dep.id << ", intent " << PetscMemoryAccessModes(dep.mode) << ' ' << dep.frame << ")\n";
          wrote_to_oss_tmp = true;
        }
      }
      // check if we wrote to it
      if (wrote_to_oss_tmp) {
        oss << '[' << rank << "] object " << it->first << " has orphaned dependencies:\n" << oss_tmp.str();
        wrote_to_oss = true;
      }
    }
    if (wrote_to_oss) {
      PetscCall((*PetscErrorPrintf)("%s\n", oss.str().c_str()));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Orphaned dependencies found, see above");
    }
  }
  PetscCallCXX(this->map.clear());
  PetscFunctionReturn(0);
}

static MarkedObjectMap marked_object_map;

PetscErrorCode PetscDeviceContextMarkIntentFromID(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode) {
#if PetscDefined(USE_DEBUG)
  const auto index    = petscstack.currentsize > 2 ? petscstack.currentsize - 2 : 0;
  const auto file     = petscstack.file[index];
  const auto function = petscstack.function[index];
  const auto line     = petscstack.line[index];
#else
  constexpr auto file     = "unknown_file";
  constexpr auto function = "unknown_function";
  constexpr auto line     = 1;
#endif
  PetscObjectId dctx_id;
  auto         &deps = marked_object_map.map[id];

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  // dereference after valid ctx check
  dctx_id = dctx->id;
  PetscCall(marked_object_map.register_finalize());
  if (!deps.empty()) {
    const auto end   = deps.end();
    const auto it    = std::find_if(deps.begin(), end, MarkedObjectMap::FindContext{dctx_id});
    const auto found = it != end;
    const auto next  = std::next(it);

    if (found) {
      // we have been found
#if PetscDefined(USE_DEBUG)
      const auto frame_file     = it->frame.file.c_str();
      const auto frame_function = it->frame.function.c_str();
      const auto frame_line     = it->frame.line;
#else
      constexpr auto frame_file     = file;
      constexpr auto frame_function = function;
      constexpr auto frame_line     = line;
#endif

      PetscCall(PetscInfo(nullptr, "dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " found old self %s (%s() at %s:%d) -> %s (%s() at %s:%d)\n", dctx_id, id, PetscMemoryAccessModes(it->mode), frame_function, frame_file, frame_line, PetscMemoryAccessModes(mode), function, file, line));
      it->mode  = mode;
      it->frame = MarkedObject::frame_type{file, function, line};
      // last element, we already are the leaf
      if (next == end) PetscFunctionReturn(0);
    }
    {
      const auto ointent            = deps.back().mode;
      const auto we_read_they_write = PetscMemoryAccessRead(mode) && PetscMemoryAccessWrite(ointent);
      const auto we_write           = PetscMemoryAccessWrite(mode);

      if (we_read_they_write || we_write) {
        const auto &octx = deps.back().ctx;

        PetscCall(PetscInfo(nullptr, "dctx %" PetscInt64_FMT " - %" PetscInt64_FMT " other context %" PetscInt64_FMT " intent %s conflicts with %s -> serializing\n", dctx_id, id, octx->id, PetscMemoryAccessModes(ointent), PetscMemoryAccessModes(mode)));
        PetscCall(PetscDeviceContextWaitForContext(dctx, octx));
      }
    }
    if (found) {
      PetscCall(PetscInfo(nullptr, "dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " rotating to leaf position with intent %s\n", dctx_id, id, PetscMemoryAccessModes(mode)));
      PetscCallCXX(std::rotate(it, next, end));
      PetscFunctionReturn(0);
    }
  }
  // become the new leaf by appending ourselves
  PetscCall(PetscInfo(nullptr, "dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " adding new leaf with intent %s\n", dctx_id, id, PetscMemoryAccessModes(mode)));
  PetscCallCXX(deps.emplace_back(dctx, dctx_id, mode, file, function, line));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSyncClearMap(PetscDeviceContext dctx) {
  const auto dctx_id    = dctx->id;
  auto      &object_map = marked_object_map.map;
  auto      &parentset  = CxxDataCast(dctx)->parents;
  // must clear later, otherwise the recursive sync clear map call is unbounded in case of a
  // loop! so we make a copy
  const auto parents    = std::vector<Petsc::util::decay_t<decltype(parentset)>::key_type>(parentset.cbegin(), parentset.cend());

  PetscFunctionBegin;
  for (auto mapit = object_map.begin(); mapit != object_map.end() /* do not hoist! */;) {
    auto      &deps  = mapit->second;
    const auto begin = deps.begin();
    const auto end   = deps.end();
    auto       found = std::find_if(begin, end, MarkedObjectMap::FindContext{dctx_id});

    if (found != end) {
      using const_reference_type = MarkedObjectMap::mapped_type::const_reference;

      // increment the iterator to include ourselves in the loop
      ++found;
      // we are a dep for this object, now remove ourself and any the previous dependencies if
      // they are our direct parents (since by synchronizing ourselves we have synchronized
      // their previous state)
      PetscCallCXX(deps.erase(std::remove_if(begin, found, [&](const_reference_type pair) { return parentset.count(pair.ctx) || pair.id == dctx_id; }), found));
    }
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      std::ostringstream oss;
      const auto         cend = deps.cend();

      oss << "ctx " << dctx_id << " remaining deps for obj " << mapit->first << ": {";
      for (auto it = deps.cbegin(); it != cend; ++it) {
        oss << "[ctx " << it->id << ", " << PetscMemoryAccessModes(it->mode) << ' ' << it->frame << ']';
        if (std::next(it) != cend) oss << ", ";
      }
      oss << '}';
      PetscCall(PetscInfo(nullptr, "%s\n", oss.str().c_str()));
    }
    // continue to next object, but erase this one if it has no more dependencies
    mapit = deps.empty() ? object_map.erase(mapit) : std::next(mapit);
  }
  // aftermath, clear our set of parents (to avoid infinite recursion) and mark ourselves as no
  // longer contained (while the empty graph technically *is* always contained, it is not what
  // we mean by it)
  parentset.clear();
  dctx->contained = PETSC_FALSE;
  for (auto &&parent : parents) PetscCall(PetscDeviceContextSyncClearMap(parent));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextCheckNotOrphaned(PetscDeviceContext dctx) {
  std::ostringstream oss;
  const auto         allow        = dctx->options.allow_orphans;
  const auto         contained    = dctx->contained;
  auto               wrote_to_oss = allow;
  auto              &object_map   = marked_object_map.map;

  PetscFunctionBegin;
  for (auto mapit = object_map.begin(); mapit != object_map.end() /* do not hoist! */;) {
    auto      &deps = mapit->second;
    const auto end  = deps.end();
    const auto it   = std::find_if(deps.begin(), end, MarkedObjectMap::FindContext{dctx});

    if (it != end) {
      // we were a dependency, now check if we either allow dangling dependencies or if our
      // graph is contained by another
      if (!allow && !contained) {
        wrote_to_oss = PETSC_TRUE;
        oss << "- PetscObject (id " << mapit->first << "), intent " << PetscMemoryAccessModes(it->mode) << ' ' << it->frame;
        if (deps.size() == 1) oss << " (orphaned)"; // we are the only dependency
        oss << '\n';
      }
      // remove ourselves
      PetscCallCXX(deps.erase(it));
    }
    // continue to next object, but erase this one if it has no more dependencies
    mapit = deps.empty() ? object_map.erase(mapit) : std::next(mapit);
  }
  PetscCheck(allow || contained || !wrote_to_oss, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Destroying PetscDeviceContext (id %" PetscInt64_FMT ") would leave the following dangling (possibly orphaned) dependants:\n%s\nMust synchronize before destroying it, or allow it to be destroyed with orphans",
             dctx->id, oss.str().c_str());
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize - Wait until all work queued on a `PetscDeviceContext` has
  finished

  Not Collective

  Input Parameters:
. dctx - The `PetscDeviceContext` to synchronize

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`. The host will not return from this routine until `dctx` is
  idle.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
.ve

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) {
    PetscUseTypeMethod(dctx, synchronize);
    PetscCall(PetscDeviceContextSyncClearMap(dctx));
  }
  PetscFunctionReturn(0);
}

// since the pointers allocated via PetscDeviceAllocate() may be device pointers we cannot just
// store meta-data within the pointer itself (as we can't dereference them). So instead we need
// to keep an extra map to keep track of them
struct MemoryMap : Petsc::RegisterFinalizeable<MemoryMap> {
  using map_type = std::unordered_map<void *, PetscMemType>;

  map_type map;

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept {
    PetscFunctionBegin;
    // PetscCallCXX technically not needed as map.clear() is noexcept, but no harm no foul?
    PetscCallCXX(this->map.clear());
    PetscFunctionReturn(0);
  }
};

static MemoryMap memory_map;

/*@C
  PetscDeviceAllocate - Allocate device-aware memory

  Not Collective

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

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`. Note `ptr` is always valid (the user may freely inspect
  its value).

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\- dctx -->
                         \- ptr ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscGetMemType()`, `PetscDeviceFree()`,
`PetscDeviceDeallocate()`, `PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscDeviceAllocate(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, size_t n, void **PETSC_RESTRICT ptr) {
  PetscFunctionBegin;
  PetscValidPointer(ptr, 5);
  if (PetscLikely(n)) {
    PetscCall(memory_map.register_finalize());
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscUseTypeMethod(dctx, memalloc, clear, mtype, n, ptr);
    {
      const auto ret = memory_map.map.emplace(*ptr, mtype);

      // we previously allocated the pointer (with some memtype), but now emplace has failed
      // and the new mtype doesn't match. In practice this shouldn't happen, since that
      // indicates that e.g. cudaMalloc() and cudaMallocHost() have returned identical
      // pointers, but it doesn't hurt to check
      PetscAssert(ret.second || (mtype == ret.first->second), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to overwrite previously allocated pointer %p (of memtype %s) with new memtype %s", ret.first->first, PetscMemTypes(ret.first->second), PetscMemTypes(mtype));
    }
  } else {
    *ptr = nullptr;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDeallocate - Free device-aware memory

  Not Collective

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

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`. `ptr` need not be freed on the same `PetscDeviceContext`
  that it was allocated with, but the user should take care to ensure the pointer is returned
  on a logically ordered `PetscDeviceContext`.

  DAG representation:
.vb
  time ->

  -> dctx -/- |= CALL =| - dctx ->
  -> ptr -/
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceAllocate()`
@*/
PetscErrorCode PetscDeviceDeallocate(PetscDeviceContext dctx, void *PETSC_RESTRICT ptr) {
  PetscFunctionBegin;
  if (ptr) {
    auto      &map   = memory_map.map;
    const auto found = map.find(const_cast<MemoryMap::map_type::key_type>(ptr));

    PetscAssert(found != map.end(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Pointer %p was not allocated via PetscDeviceAllocate()", ptr);
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscUseTypeMethod(dctx, memfree, found->second, ptr);
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

  Sequential Consistency Notes:
  Sequentially Consistent on `dctx`.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
  -> dest --------------------->
  -> src ---------------------->
.ve

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
    PetscUseTypeMethod(dctx, memcopy, dest, src, n, mode);
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

  Sequential Consistency Notes:
  Sequentially Consistent on

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
    PetscUseTypeMethod(dctx, memset, mtype, dest, v, n);
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

/*@C
  PetscDeviceContextSetOption - Set an option controlling a `PetscDeviceContext`'s behavior

  Input Parameter:
+ dctx  - The `PetscDeviceContext`
. opt   - The option
- value - The value to set it to

  Level: intermediate
@*/
PetscErrorCode PetscDeviceContextSetOption(PetscDeviceContext dctx, PetscDeviceContextOption opt, PetscBool value) {
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  switch (opt) {
  case PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS: dctx->options.allow_orphans = value; break;
  }
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
      for (auto &&dvec : nullContexts) {
        for (auto &&dctx : dvec) PetscCall(PetscDeviceContextDestroy(&dctx));
        PetscCallCXX(dvec.clear());
      }
      nullContextsFinalizer = false;
      PetscFunctionReturn(0);
    };

    nullContextsFinalizer = true;
    PetscCall(PetscRegisterFinalize(std::move(finalizer)));
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

static auto               rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
static auto               rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
static PetscDeviceContext globalContext  = nullptr;

/* when PetsDevice initializes PetscDeviceContext eagerly the type of device created should
 * match whatever device is eagerly intialized */
PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType type) {
  PetscFunctionBegin;
  PetscValidDeviceType(type, 1);
  rootDeviceType = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType type)
{
  PetscFunctionBegin;
  PetscValidStreamType(type,1);
  rootStreamType = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSetupGlobalContext_Private() {
  PetscFunctionBegin;
  if (PetscUnlikely(!globalContext)) {
    const auto finalizer = [] {
      PetscFunctionBegin;
      PetscCall(PetscDeviceContextDestroy(&globalContext));
      rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE;
      rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE;
      PetscFunctionReturn(0);
    };

    /* this exists purely as a valid device check. */
    PetscCall(PetscDeviceInitializePackage());
    PetscCall(PetscRegisterFinalize(std::move(finalizer)));
    PetscCall(PetscInfo(nullptr, "Initializing global PetscDeviceContext\n"));
    /* we call the allocator directly here since the ObjectPool creates a PetscContainer which
     * eventually tries to call logging functions. However, this routine may be purposefully
     * called __before__ logging is initialized, so the logging function would PETSCABORT */
    PetscCall(contextPool.allocator().create(&globalContext));
    PetscCall(PetscDeviceContextSetStreamType(globalContext, rootStreamType));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(globalContext, PETSC_DEVICE_DEFAULT()));
    PetscCall(PetscDeviceContextSetUp(globalContext));
  }
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
  PetscAssert(dctx->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " must be set up before being set as global context", dctx->id);
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCall(PetscDeviceSetDefaultDeviceType(dtype));
  globalContext = dctx;
  PetscCall(PetscInfo(nullptr, "Set global PetscDeviceContext id %" PetscInt64_FMT "\n", dctx->id));
  PetscFunctionReturn(0);
}

/*
  needed because PetscInitialize() needs to also query these options to set the defaults. Since
  it does not yet have a PetscDeviceContext to call this with, the actual options queries are
  abstracted out, so you can call this without one.
*/
PetscErrorCode PetscDeviceContextQueryOptions_Internal(MPI_Comm comm, const char prefix[], std::pair<PetscDeviceType, PetscBool> &deviceType, std::pair<PetscStreamType, PetscBool> &streamType, std::pair<PetscBool, PetscBool> &allowOrphans) {
  auto dtype = static_cast<PetscInt>(deviceType.first);
  auto stype = static_cast<PetscInt>(streamType.first);

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix, 2);
  PetscOptionsBegin(comm, prefix, "PetscDeviceContext Options", "Sys");
  /* set the device type first */
  PetscCall(PetscOptionsEList("-device_context_device_type", "Underlying PetscDevice", "PetscDeviceContextSetDevice", PetscDeviceTypes, PETSC_DEVICE_MAX, PetscDeviceTypes[dtype], &dtype, &deviceType.second));
  PetscCall(PetscOptionsEList("-device_context_stream_type", "PetscDeviceContext PetscStreamType", "PetscDeviceContextSetStreamType", PetscStreamTypes, PETSC_STREAM_MAX, PetscStreamTypes[stype], &stype, &streamType.second));
  PetscCall(PetscOptionsBool("-device_context_allow_orphans", "Whether a PetscDeviceContext may be destroyed with dangling dependencies", "PetscDeviceContextSetOption", allowOrphans.first, &allowOrphans.first, &allowOrphans.second));
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
  auto dtype   = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE_TYPE, PETSC_FALSE);
  auto stype   = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE, PETSC_FALSE);
  auto orphans = std::make_pair(PETSC_TRUE, PETSC_FALSE);

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix, 2);
  PetscValidDeviceContext(dctx, 3);
  orphans.first = dctx->options.allow_orphans;
  /* set the device type first */
  if (auto device = dctx->device) PetscCall(PetscDeviceGetType(device, &dtype.first));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype.first));
  PetscCall(PetscDeviceContextQueryOptions_Internal(comm, prefix, dtype, stype, orphans));
  if (dtype.second) PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, dtype.first));
  if (stype.second) PetscCall(PetscDeviceContextSetStreamType(dctx, stype.first));
  if (orphans.second) PetscCall(PetscDeviceContextSetOption(dctx, PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS, orphans.first));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscFunctionReturn(0);
}
