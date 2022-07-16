#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petsc/private/viewerimpl.h>
#include <petsc/private/cpputil.hpp>

#include <petsc/private/cpp/flat_map.hpp>
#include "objpool.hpp"

#include <utility>
#include <array>
#include <vector>
#include <algorithm> // std::find
#include <unordered_map>

// debugging aides
#include <sstream>
#include <string>
#if PetscDefined(USE_DEBUG) && PetscDefined(USE_INFO)
#define PETSC_USE_DEBUG_AND_INFO 1
#define PetscDebugInfo(...)      PetscInfo(nullptr, __VA_ARGS__)
#else
#define PetscDebugInfo(...) 0
#endif

const char *const PetscStreamTypes[] = {"global_blocking", "default_blocking", "global_nonblocking", "max", "PetscStreamType", "PETSC_STREAM_", nullptr};

const char *const PetscDeviceContextJoinModes[] = {"destroy", "sync", "no_sync", "PetscDeviceContextJoinMode", "PETSC_DEVICE_CONTEXT_JOIN_", nullptr};

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_AUTO) == 4, "");

PetscLogEvent DCONTEXT_Create, DCONTEXT_Destroy, DCONTEXT_ChangeStream, DCONTEXT_SetUp, DCONTEXT_SetDevice;
PetscLogEvent DCONTEXT_Duplicate, DCONTEXT_QueryIdle, DCONTEXT_WaitForCtx, DCONTEXT_Fork, DCONTEXT_Join;
PetscLogEvent DCONTEXT_Mark, DCONTEXT_Sync;

template <typename T, typename C = std::equal_to<T>>
class flat_set {
public:
  using key_type        = T;
  using value_type      = key_type;
  using key_equal       = C;
  using data_type       = std::vector<key_type>;
  using size_type       = typename data_type::size_type;
  using difference_type = typename data_type::difference_type;
  using iterator        = typename data_type::iterator;
  using const_iterator  = typename data_type::const_iterator;

  PETSC_NODISCARD const key_equal &key_eq() const noexcept { return cmp_; }
  PETSC_NODISCARD key_equal       &key_eq() noexcept { return cmp_; }

  PETSC_NODISCARD const_iterator cbegin() const noexcept { return data_.cbegin(); }
  PETSC_NODISCARD const_iterator cend() const noexcept { return data_.cend(); }

  PETSC_NODISCARD iterator begin() noexcept { return data_.begin(); }
  PETSC_NODISCARD iterator end() noexcept { return data_.end(); }

  PETSC_NODISCARD const_iterator begin() const noexcept { return cbegin(); }
  PETSC_NODISCARD const_iterator end() const noexcept { return cend(); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args &&...args) {
    key_type tmp{std::forward<Args>(args)...};
    auto     it = find(tmp);

    if (it == end()) {
      data_.push_back(std::move(tmp));
      return std::make_pair(std::prev(end()), true);
    }
    return std::make_pair(it, false);
  }

  PETSC_NODISCARD iterator find(const key_type &key) noexcept { return find_(begin(), end(), key_eq(), key); }

  PETSC_NODISCARD const_iterator find(const key_type &key) const noexcept { return find_(cbegin(), cend(), key_eq(), key); }

  PETSC_NODISCARD bool contains(const key_type &key) const noexcept { return cend() != find(key); }

  iterator erase(const_iterator pos) noexcept(noexcept(data_.erase(pos))) { return data_.erase(pos); }

  iterator erase(const_iterator begin, const_iterator end) noexcept(noexcept(data_.erase(begin, end))) { return data_.erase(begin, end); }

  void clear() noexcept(noexcept(data_.clear())) { return data_.clear(); }

  PETSC_NODISCARD size_type size() const noexcept { return data_.size(); }

private:
  data_type data_;
  key_equal cmp_;

  template <typename Iterator, typename F>
  static constexpr Iterator find_(Iterator it, Iterator end, F &&cmp, const key_type &key) noexcept {
    for (; it != end; ++it)
      if (cmp(*it, key)) break;
    return it;
  }
};

struct PetscDeviceContextCxxData {
  struct parent_type {
    PetscObjectId    id;
    PetscObjectState state;

    constexpr parent_type() noexcept : parent_type(0, 0) { }

    constexpr parent_type(PetscObjectId id_, PetscObjectState state_) noexcept : id(id_), state(state_) { }

    constexpr explicit parent_type(PetscDeviceContext dctx) noexcept : parent_type(dctx->id, dctx->state) { }
  };

  using upstream_type = Petsc::util::flat_map<PetscDeviceContext, parent_type>;
  using dep_type      = flat_set<PetscObjectId>;

  upstream_type upstream;
  dep_type      deps;

  PETSC_NODISCARD PetscErrorCode clear() noexcept {
    PetscFunctionBegin;
    PetscCallCXX(this->upstream.clear());
    PetscCallCXX(this->deps.clear());
    PetscFunctionReturn(0);
  }
};

PETSC_CXX_COMPAT_DECL(auto CxxDataCast(PetscDeviceContext dctx))
PETSC_DECLTYPE_AUTO_RETURNS(static_cast<PetscDeviceContextCxxData *>(dctx->cxxdata))

/* Define the allocator */
class PetscDeviceContextAllocator : public Petsc::AllocatorBase<PetscDeviceContext> {
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset_options_(DeviceContextOptions &options)) {
    PetscFunctionBegin;
    options.allow_orphans = PetscDefined(USE_DEBUG) ? PETSC_FALSE : PETSC_TRUE;
    PetscFunctionReturn(0);
  }

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscDeviceContext *dctx)) {
    PetscFunctionBegin;
    PetscCall(PetscNew(dctx));
    PetscCallCXX((*dctx)->cxxdata = new PetscDeviceContextCxxData);
    PetscCall(reset(*dctx, false));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying", dctx->numChildren);
    PetscTryTypeMethod(dctx, destroy);
    PetscCall(PetscDeviceDestroy(&dctx->device));
    PetscCall(PetscFree(dctx->childIDs));
    PetscCall(PetscFree(dctx->name));
    delete CxxDataCast(dctx);
    PetscCall(PetscFree(dctx));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(PetscDeviceContext dctx, bool zero = true)) {
    PetscFunctionBegin;
    if (zero) {
      // reset the device if the user set it
      if (auto &userset = dctx->usersetdevice) {
        userset = PETSC_FALSE;
        if (const auto destroy = dctx->ops->destroy) PetscCall((*destroy)(dctx));
        PetscCall(PetscDeviceDestroy(&dctx->device));
      }
      dctx->state       = 0;
      dctx->numChildren = 0;
      dctx->setup       = PETSC_FALSE;
      dctx->contained   = PETSC_FALSE;
      PetscCall(CxxDataCast(dctx)->clear());
      // don't deallocate the child array, rather just zero it out
      PetscCall(PetscArrayzero(dctx->childIDs, dctx->maxNumChildren));
      PetscCall(PetscFree(dctx->name));
    }
    dctx->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscCall(reset_options_(dctx->options));
    PetscCall(PetscObjectNewId_Internal(&dctx->id));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_Create, 0, 0, 0, 0));
  PetscCall(contextPool.get(*dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_Create, 0, 0, 0, 0));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_Destroy, 0, 0, 0, 0));
         PetscCall(PetscDeviceContextCheckNotOrphaned(*dctx));
         // std::move of the expression of the trivially-copyable type 'PetscDeviceContext' (aka
  // '_n_PetscDeviceContext *') has no effect; remove std::move() [performance-move-const-arg]
  // can't remove std::move, since reclaim only takes r-value reference
  PetscCall(contextPool.reclaim(std::move(*dctx))); // NOLINT (performance-move-const-arg)
         PetscCall(PetscLogEventEnd(DCONTEXT_Destroy, 0, 0, 0, 0));
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
    PetscCall(PetscLogEventBegin(DCONTEXT_ChangeStream, 0, 0, 0, 0));
    PetscUseTypeMethod(dctx, changestreamtype, type);
    PetscCall(PetscLogEventEnd(DCONTEXT_ChangeStream, 0, 0, 0, 0));
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

/*
  Actual function to set the device.

  1. Repeatedly destroying and recreating internal data structures (like streams and events)
     for recycled PetscDeviceContexts is not free. If done often, it does add up.
  2. The vast majority of PetscDeviceContexts are created by PETSc either as children or
     default contexts. The default contexts *never* change type, and the chilren are extremely
     unlikely to (chances are if you fork once, you will fork again very soon).
  3. The only time this calculus changes is if the user themselves sets the device type. In
     this case since we do not know what the user has changed, so must always wipe the slate
     clean.

  Thus we need to keep track whether the user explicitly sets the device contexts device.
*/
static PetscErrorCode PetscDeviceContextSetDevice_Internal(PetscDeviceContext dctx, PetscDevice device, PetscBool user_set) {
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change its device
  PetscValidDeviceContext(dctx, 1);
  PetscValidDevice(device, 2);
  if (dctx->device && (dctx->device->id == device->id)) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(DCONTEXT_SetDevice, 0, 0, 0, 0));
  if (const auto destroy = dctx->ops->destroy) PetscCall((*destroy)(dctx));
  PetscCall(PetscDeviceDestroy(&dctx->device));
  PetscCall(PetscMemzero(dctx->ops, sizeof(*dctx->ops)));
  PetscCall((*device->ops->createcontext)(dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_SetDevice, 0, 0, 0, 0));
  PetscCall(PetscDeviceReference_Internal(device));
  dctx->device        = device;
  dctx->setup         = PETSC_FALSE;
  dctx->usersetdevice = user_set;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext dctx, PetscDeviceType type) {
  PetscDevice device;

  PetscFunctionBegin;
  PetscCall(PetscDeviceGetDefaultForType_Internal(type, &device));
  PetscCall(PetscDeviceContextSetDevice_Internal(dctx, device, PETSC_FALSE));
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
  PetscCall(PetscDeviceContextSetDevice_Internal(dctx, device, PETSC_TRUE));
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
    const auto default_dtype = PETSC_DEVICE_DEFAULT();

    PetscCall(PetscInfo(nullptr, "PetscDeviceContext %" PetscInt64_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n", dctx->id, PetscDeviceTypes[default_dtype]));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, default_dtype));
  }
  PetscCall(PetscLogEventBegin(DCONTEXT_SetUp, 0, 0, 0, 0));
  PetscUseTypeMethod(dctx, setup);
  PetscCall(PetscLogEventEnd(DCONTEXT_SetUp, 0, 0, 0, 0));
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextDuplicate_Private(PetscDeviceContext dctx, PetscStreamType stype, PetscDeviceContext *dctxdup) {
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DCONTEXT_Duplicate, 0, 0, 0, 0));
  PetscCall(PetscDeviceContextCreate(dctxdup));
  PetscCall(PetscDeviceContextSetStreamType(*dctxdup, stype));
  if (const auto device = dctx->device) { PetscCall(PetscDeviceContextSetDevice_Internal(*dctxdup, device, dctx->usersetdevice)); }
  (*dctxdup)->options = dctx->options;
  PetscCall(PetscDeviceContextSetUp(*dctxdup));
  PetscCall(PetscLogEventEnd(DCONTEXT_Duplicate, 0, 0, 0, 0));
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
  PetscCall(PetscDeviceContextDuplicate_Private(dctx, stype, dctxdup));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_QueryIdle, 0, 0, 0, 0));
  PetscUseTypeMethod(dctx, query, idle);
  PetscCall(PetscLogEventEnd(DCONTEXT_QueryIdle, 0, 0, 0, 0));
  PetscCall(PetscInfo(nullptr, "PetscDeviceContext id %" PetscInt64_FMT " %s idle\n", dctx->id, *idle ? "was" : "was not"));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_WaitForCtx, 0, 0, 0, 0));
  PetscUseTypeMethod(dctxa, waitforcontext, dctxb);
  PetscCallCXX(CxxDataCast(dctxa)->upstream[dctxb] = PetscDeviceContextCxxData::parent_type{dctxb});
  PetscCall(PetscLogEventEnd(DCONTEXT_WaitForCtx, 0, 0, 0, 0));
  PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " waiting on dctx %" PetscInt64_FMT "\n", dctxa->id, dctxb->id));
  ++dctxa->state;
  dctxb->contained = PETSC_TRUE; // dctxb is now contained by dctxa
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextForkWithStreamType - Create a set of dependent child contexts from a parent
  context with a prescribed `PetscStreamType`

  Not Collective

  Input Parameters:
+ dctx  - The parent `PetscDeviceContext`
. stype - The prescribed `PetscStreamType`
- n     - The number of children to create

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

  The `PetscStreamType` of `dctx` does not have to equal `stype`. In fact, it is often the case
  that they are different. This is useful in cases where a routine can locally exploit stream
  parallelism without needing to worry about what stream type the incoming `PetscDeviceContext`
  carries.

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
PetscErrorCode PetscDeviceContextForkWithStreamType(PetscDeviceContext dctx, PetscStreamType stype, PetscInt n, PetscDeviceContext **dsub) {
  // debugging only
  std::string idList;
  auto        ninput = n;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts requested %" PetscInt_FMT " < 0", n);
  PetscValidPointer(dsub, 3);
  *dsub = nullptr;
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCallCXX(idList.reserve(4 * n));
  PetscCall(PetscLogEventBegin(DCONTEXT_Fork, 0, 0, 0, 0));
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
  for (PetscInt i = 0; ninput && (i < dctx->numChildren); ++i) {
    auto &childID = dctx->childIDs[i];
    /* empty child slot */
    if (!childID) {
      /* create the child context in the image of its parent */
      PetscCall(PetscDeviceContextDuplicate_Private(dctx, stype, (*dsub) + i));
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
  PetscCall(PetscLogEventEnd(DCONTEXT_Fork, 0, 0, 0, 0));
  PetscCall(PetscDebugInfo("Forked %" PetscInt_FMT " children from parent %" PetscInt64_FMT " with IDs: %s\n", n, dctx->id, idList.c_str()));
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
  Behaves identically to `PetscDeviceContextForkWithStreamType()` except that the prescribed
  `PetscStreamType` is taken from `dctx`. That is, this routine is shorthand for\:

.vb
  PetscDeviceContextGetStreamType(dctx, &stype);
  PetscDeviceContextForkWithStreamType(dctx, stype, ...);
.ve

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextForkWithStreamType()`, `PetscDeviceContextJoin()`,
`PetscDeviceContextSynchronize()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub) {
  auto stype = PETSC_STREAM_DEFAULT_BLOCKING;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextForkWithStreamType(dctx, stype, n, dsub));
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

  Level: beginner

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
  PetscCall(PetscLogEventBegin(DCONTEXT_Join, 0, 0, 0, 0));
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
  PetscCall(PetscLogEventEnd(DCONTEXT_Join, 0, 0, 0, 0));

  PetscCall(PetscDebugInfo("Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt64_FMT ", mode %s with IDs: %s\n", n, dctx->id, PetscDeviceContextJoinModes[joinMode], idList.c_str()));
  PetscFunctionReturn(0);
}

// REVIEW ME
// REMOVE ME
#include <map>
class map_size_counter : Petsc::RegisterFinalizeable<map_size_counter> {
  friend class Petsc::RegisterFinalizeable<map_size_counter>;
  std::map<std::size_t, std::size_t, std::greater<std::size_t>> s_;
  PetscBool                                                     print_;
  const std::string                                             map_name;
  bool                                                          finalize_called_ = false;

  PetscErrorCode register_finalize_() noexcept {
    auto flg = PETSC_FALSE;

    PetscFunctionBegin;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-print_stats", &print_, &flg));
    print_ = (PetscBool)(print_ && flg);
    if (!print_) PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-print_map_stats", &print_, &flg));
    print_ = (PetscBool)(print_ && flg);
    PetscFunctionReturn(0);
  }

  PetscErrorCode finalize_() noexcept {
    PetscFunctionBegin;
    finalize_called_ = true;
    if (print_) {
      std::size_t        minval = (size_t)-1, maxval = 0, samples = 0, weighted_sum = 0;
      std::ostringstream oss;

      for (auto it = s_.cbegin(); it != s_.cend(); ++it) {
        const auto [size, count] = *it;

        minval = std::min(minval, size);
        maxval = std::max(maxval, size);
        samples += count;
        weighted_sum += count * size;
        oss << "  size " << size << " count " << count << std::endl;
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s map statistics: min size %zu max size %zu avg %g\n", map_name.c_str(), minval, maxval, (double)(weighted_sum / samples)));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s\n", oss.str().c_str()));
      print_ = PETSC_FALSE;
    }
    s_.clear();
    PetscFunctionReturn(0);
  }

public:
  map_size_counter(std::string name) : s_(), print_(PETSC_FALSE), map_name(std::move(name)) { }

  template <typename T>
  PetscErrorCode count(const T &container) noexcept {
    PetscFunctionBegin;
    if (!finalize_called_) PetscCall(this->register_finalize());
    if (print_) ++s_[container.size()];
    PetscFunctionReturn(0);
  }
};

static map_size_counter dependency_ctr("Dependency");
static map_size_counter upstream_ctr("Upstream parent");
static map_size_counter marked_ctr("Marked Object");

// This exists outside of MarkedObjectMap because it needs to be specialized...
template <bool use_debug>
struct PetscStackFrame;

template <>
struct PetscStackFrame</* use_debug = */ true> {
  std::string file;
  std::string function;
  int         line;

  explicit PetscStackFrame(const char *file_, const char *func_, int line_) : file(split_on_petsc_path_(file_)), function(func_), line(line_) { }

  bool operator==(const PetscStackFrame &other) const noexcept { return line == other.line && file == other.file && function == other.function; }

private:
  static std::string split_on_petsc_path_(std::string &&in) {
    auto pos = in.find("petsc/src");

    if (pos == std::string::npos) pos = in.find("petsc/include");
    if (pos == std::string::npos) pos = 0;
    return in.substr(pos);
  }

  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &frame) {
    os << '(' << frame.function << "() at " << frame.file << ':' << frame.line << ')';
    return os;
  }
};

template <>
struct PetscStackFrame</* use_debug = */ false> {
  template <typename... T>
  constexpr PetscStackFrame(T &&...) noexcept { }

  constexpr bool       operator==(const PetscStackFrame &) const noexcept { return true; }
  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &) noexcept { return os; }
};

// a helper to enumerate the types
struct MarkedObjectMap : Petsc::RegisterFinalizeable<MarkedObjectMap> {
  // A snapshot of the state of the PetscDeviceContext when it last accessed a particular
  // object. Rationale for each member is as follows:
  // ctx        - to be able to synchronize/serialize with later
  // dctx_id    - uniquely identify the ctx, comparing pointers is not sufficient as
  //              PetscDeviceContexts are reused
  // dctx_state - to determine if a downstream ctx has already synchronized past this state
  // frame      - the stack frame where the object was accessed for debug purposes
  struct snapshot_type {
    using frame_type = PetscStackFrame<PetscDefined(USE_DEBUG)>;

    PetscDeviceContext ctx;
    PetscObjectId      dctx_id;
    PetscObjectState   dctx_state;
    frame_type         frame;

    snapshot_type(PetscDeviceContext ctx, frame_type frame) noexcept : ctx(ctx), dctx_id(ctx->id), dctx_state(ctx->state), frame(std::move(frame)) { }

    snapshot_type(PetscDeviceContext ctx, const char *file, const char *function, int line) noexcept : snapshot_type(ctx, frame_type{file, function, line}) { }

    bool operator==(const snapshot_type &other) const noexcept { return ctx == other.ctx && dctx_state == other.dctx_state && dctx_id == other.dctx_id && frame == other.frame; }

    bool operator!=(const snapshot_type &other) const noexcept { return !(*this == other); }
  };

  struct mapped_type {
    using mode_type       = PetscMemoryAccessMode;
    using dependency_type = std::vector<snapshot_type>;

    mode_type       mode = PETSC_MEMORY_ACCESS_READ;
    dependency_type dependencies{};
  };

  using map_type = std::unordered_map<PetscObjectId, mapped_type>;

  map_type map;

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept {
    PetscFunctionBegin;
    PetscCall(PetscInfo(nullptr, "Finalizing marked object map\n"));
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
        const auto        &mapped           = it->second;
        const auto         mode             = PetscMemoryAccessModes(mapped.mode);

        for (auto &&dep : mapped.dependencies) {
          if (!dep.ctx->options.allow_orphans) {
            wrote_to_oss_tmp = true;
            oss_tmp << "  [" << rank << "] dctx " << dep.ctx << " (id " << dep.dctx_id << ", state " << dep.dctx_state << ", intent " << mode << ' ' << dep.frame << ")\n";
          }
        }
        // check if we wrote to it
        if (wrote_to_oss_tmp) {
          oss << '[' << rank << "] object " << it->first << " has orphaned dependencies:\n" << oss_tmp.str();
          wrote_to_oss = true;
        }
      }
      if (wrote_to_oss) {
        //PetscCall((*PetscErrorPrintf)("%s\n",oss.str().c_str()));
        //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Orphaned dependencies found, see above");
      }
    }
    // replace with new map, since clear() does not necessarily free memory
    PetscCallCXX(this->map = map_type{});
    PetscFunctionReturn(0);
  }
};

// A mapping between PetscObjectId (i.e. some PetscObject) to the list of PetscDeviceContexts
// which last accessed them.
static MarkedObjectMap marked_object_map;

namespace detail {

struct ignore {
  template <typename... T>
  constexpr bool operator()(T &&...) const noexcept {
    return false;
  }
};

} // namespace detail

template <typename T = detail::ignore, typename U = detail::ignore>
static PetscErrorCode PetscDeviceContextMapIterVistor(PetscDeviceContext dctx, T &&callback = T{}, U &&pred = U{}) noexcept {
  const auto dctx_id    = dctx->id;
  auto      &object_map = marked_object_map.map;

  PetscFunctionBegin;
  PetscCall(dependency_ctr.count(CxxDataCast(dctx)->deps));
  PetscCall(upstream_ctr.count(CxxDataCast(dctx)->upstream));
  for (auto &&dep : CxxDataCast(dctx)->deps) {
    const auto mapit = object_map.find(dep);

    // Need this check since the final PetscDeviceContext may run through this *after* the map
    // has been finalized (and cleared), and hence might fail to find its dependencies. This is
    // perfectly valid since the user no longer cares about dangling dependencies after PETSc
    // is finalized
    if (PetscLikely(object_map.cend() != mapit)) {
      auto      &deps = mapit->second.dependencies;
      const auto end  = deps.end();
      const auto it   = std::remove_if(deps.begin(), end, [&](const MarkedObjectMap::snapshot_type &obj) { return (obj.dctx_id == dctx_id) || pred(obj); });

      PetscCall(callback(mapit, deps.cbegin(), static_cast<decltype(deps.cend())>(it)));
      // remove ourselves
      PetscCallCXX(deps.erase(it, end));
      // continue to next object, but erase this one if it has no more dependencies
      if (deps.empty()) PetscCallCXX(object_map.erase(mapit));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSyncClearMap(PetscDeviceContext dctx) {
  const auto &upstream    = CxxDataCast(dctx)->upstream;
  using upstream_type     = Petsc::util::decay_t<decltype(upstream)>::value_type;
  using map_iterator_type = MarkedObjectMap::map_type::const_iterator;
  using dep_iterator_type = MarkedObjectMap::mapped_type::dependency_type::const_iterator;
  // the recursive sync clear map call is unbounded in case of a dependenct loop so we make a
  // copy
  const std::vector<upstream_type> upstream_copy(std::make_move_iterator(upstream.cbegin()), std::make_move_iterator(upstream.cend()));
  const auto                       is_ancestor = [&](PetscObjectId id, PetscObjectState state) {
    auto found = false;

    PetscFunctionBegin;
    for (const auto &parent : upstream_copy) {
      if (parent.second.id == id) {
        found = state <= parent.second.state;
        break;
      }
    }
    PetscFunctionReturn(found);
  };

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMapIterVistor(
    dctx,
    [&](map_iterator_type mapit, dep_iterator_type it, dep_iterator_type end) {
      PetscFunctionBegin;
      if (PetscDefined(USE_DEBUG_AND_INFO)) {
        std::ostringstream oss;
        const auto         mode = PetscMemoryAccessModes(mapit->second.mode);

        oss << "synced dctx " << dctx->id << ", remaining leaves for obj " << mapit->first << ": {";
        for (; it != end; ++it) {
          oss << "[dctx " << it->dctx_id << ", " << mode << ' ' << it->frame << ']';
          if (std::next(it) != end) oss << ", ";
        }
        oss << '}';
        PetscCall(PetscInfo(nullptr, "%s\n", oss.str().c_str()));
      }
      PetscFunctionReturn(0);
    },
    [&](const MarkedObjectMap::snapshot_type &s) { return is_ancestor(s.dctx_id, s.dctx_state); }));
  // aftermath, clear our set of parents (to avoid infinite recursion) and mark ourselves as no
  // longer contained (while the empty graph technically *is* always contained, it is not what
  // we mean by it)
  PetscCall(CxxDataCast(dctx)->clear());
  dctx->contained = PETSC_FALSE;
  for (auto &&upstrm : upstream_copy) {
    // check that this parent still points to what we originally thought it was
    PetscCheck(upstrm.second.id == upstrm.first->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Upstream dctx %" PetscInt64_FMT " no longer exists, now has id %" PetscInt64_FMT, upstrm.second.id, upstrm.first->id);
    PetscCall(PetscDeviceContextSyncClearMap(upstrm.first));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextCheckNotOrphaned(PetscDeviceContext dctx) {
  std::ostringstream oss;
  const auto         allow = dctx->options.allow_orphans, contained = dctx->contained;
  auto               wrote_to_oss = false;
  using map_iterator_type         = MarkedObjectMap::map_type::const_iterator;
  using dep_iterator_type         = MarkedObjectMap::mapped_type::dependency_type::const_iterator;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMapIterVistor(dctx, [&](map_iterator_type mapit, dep_iterator_type it, dep_iterator_type end) {
    PetscFunctionBegin;
    if (allow || contained) PetscFunctionReturn(0);
    else {
      wrote_to_oss = true;
      oss << "- PetscObject (id " << mapit->first << "), intent " << PetscMemoryAccessModes(mapit->second.mode) << ' ' << it->frame;
      if (std::distance(it, end) == 0) oss << " (orphaned)"; // we were the only dependency
      oss << '\n';
    }
    PetscFunctionReturn(0);
  }));
  PetscCheck(!wrote_to_oss, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Destroying PetscDeviceContext (id %" PetscInt64_FMT ") would leave the following dangling (possibly orphaned) dependants:\n%s\nMust synchronize before destroying it, or allow it to be destroyed with orphans",
             dctx->id, oss.str().c_str());
  PetscCall(CxxDataCast(dctx)->clear());
  PetscFunctionReturn(0);
}

template <bool use_debug>
static PetscErrorCode PetscDeviceContextMarkIntentFromID_Private(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> frame, const char *name) {
  const auto dctx_id             = dctx->id;
  auto      &marked              = marked_object_map.map[id];
  auto      &old_mode            = marked.mode;
  auto      &object_dependencies = marked.dependencies;

  PetscFunctionBegin;
  PetscCall(marked_ctr.count(marked_object_map.map));
  if ((mode == PETSC_MEMORY_ACCESS_READ) && (old_mode == mode)) {
    // If both the new and last mode are read-only op then we don't need to serialize, but we
    // do need to update our previous entry (if we left one). We could in theory search our
    // dependencies (to determine if we've been here before), but then we end up searching 2
    // things
    const auto end = object_dependencies.end();
    const auto it  = std::find_if(object_dependencies.begin(), end, [=](const MarkedObjectMap::snapshot_type &obj) { return obj.dctx_id == dctx_id; });

    PetscCall(
      PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): new mode (%s) COMPATIBLE with %s mode (%s), no need to serialize\n", dctx_id, id, name, PetscMemoryAccessModes(mode), PetscMemoryAccessModes(old_mode), object_dependencies.empty() ? "default" : "old"));
    if (it != end) {
      PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): found old self as dependency, updating\n", dctx_id, id, name));
      PetscAssert(CxxDataCast(dctx)->deps.contains(id), PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext %" PetscInt64_FMT " listed as dependency for object %" PetscInt64_FMT " (%s), but does not have the object in private dependency list!", dctx_id, id, name);
      // update our previous state and bail
      it->dctx_state = dctx->state;
      it->frame      = std::move(frame);
      PetscFunctionReturn(0);
    }
  } else {
    // Any kind of write means we clear the slate (and become the sole leaf ourselves). We
    // must serialize with all prior leaves, so the question becomes can we skip serializing
    // with some of them
    const auto &upstream    = CxxDataCast(dctx)->upstream;
    const auto  is_ancestor = [&](PetscObjectId id, PetscObjectState state) {
      auto found = false;

      PetscFunctionBegin;
      for (const auto &parent : upstream) {
        if (parent.second.id == id) {
          found = state <= parent.second.state;
          break;
        }
      }
      PetscFunctionReturn(found);
    };

    PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): new mode (%s) NOT COMPATIBLE with %s mode (%s), serializing then clearing %zu %s\n", dctx_id, id, name, PetscMemoryAccessModes(mode), object_dependencies.empty() ? "default" : "old", PetscMemoryAccessModes(old_mode),
                             object_dependencies.size(), object_dependencies.size() == 1 ? "dependency" : "dependencies"));
    for (auto &&dep : object_dependencies) {
      const auto &dep_ctx = dep.ctx;
      const auto &dep_id  = dep.dctx_id;

      PetscCheck(dep_ctx->id == dep_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dctx %" PetscInt64_FMT " no longer matches expected id %" PetscInt64_FMT, dep_ctx->id, dep_id);
      if (dep_id == dctx_id) {
        PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): found old self as dependency, skipping\n", dctx_id, id, name));
      } else {
        const auto &dep_state = dep.dctx_state;
        // check first that the previous context is not an immediate ancestor. This situation
        // arises when:
        //   write(obj, ctx_parent)
        //   ctx_child.after(ctx_parent)
        //   write(other_obj, ctx_parent)
        // *do not need to serialize with ctx_parent (and implicitly wait for the other write)
        //   read(obj, ctx_child)
        PetscCheckCompatibleDeviceContexts(dctx, -1, dep_ctx, -1);
        if (is_ancestor(dep_id, dep_state)) {
          // no need to serialize, this is an ancestor
          PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): most recent leaf %" PetscInt64_FMT " is parent with same or lesser state %" PetscInt64_FMT ", no need to serialize\n", dctx_id, id, name, dep_id, dep_state));
        } else {
          PetscCall(PetscDeviceContextWaitForContext(dctx, dep_ctx));
        }
      }
    }
    // clear out the old dependencies and update the mode, we are about to append ourselves
    object_dependencies.clear();
  }
  old_mode = mode;
  // become the new leaf by appending ourselves
  PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " - obj %" PetscInt64_FMT " (%s): %s with intent %s\n", dctx_id, id, name, object_dependencies.empty() ? "no prior dependencies, creating new leaf" : "appending to existing leaves", PetscMemoryAccessModes(mode)));
  PetscCallCXX(object_dependencies.emplace_back(dctx, std::move(frame)));
  PetscCallCXX(CxxDataCast(dctx)->deps.emplace(id));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextMarkIntentFromID - Indicate a `PetscDeviceContext`s access intent to the
  auto-dependency system

  Not Collective

  Input Parameters:
+ dctx - The `PetscDeviceContext`
. id   - The `PetscObjectId` to mark
. mode - The desired access intent
- name - The object name (for debug purposes, ignored in optimized builds)

  Notes:
  This routine formally informs the dependency system that `dctx` will access the object
  represented by `id` with `mode` and adds `dctx` to `id`'s list of dependencies (termed
  "leaves").

  If the existing set of leaves have an incompatible `PetscMemoryAccessMode` to `mode`, `dctx`
  will be serialized against them.

  Level: intermediate

.seealso: `PetscDeviceContextWaitForContext()`, `PetscDeviceContextSynchronize()`,
`PetscObjectGetId()`, `PetscMemoryAccessMode`
@*/
PetscErrorCode PetscDeviceContextMarkIntentFromID(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, const char name[]) {
#if PetscDefined(USE_DEBUG)
  const auto index    = petscstack.currentsize > 2 ? petscstack.currentsize - 2 : 0;
  const auto file     = petscstack.file[index];
  const auto function = petscstack.function[index];
  const auto line     = petscstack.line[index];
#else
  constexpr auto file     = "";
  constexpr auto function = "";
  constexpr auto line     = 0;
#endif

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (name) PetscValidCharPointer(name, 4);
  PetscCall(marked_object_map.register_finalize());
  PetscCall(PetscLogEventBegin(DCONTEXT_Mark, 0, 0, 0, 0));
  PetscCall(PetscDeviceContextMarkIntentFromID_Private(dctx, id, mode, MarkedObjectMap::snapshot_type::frame_type{file, function, line}, name ? name : "unknown object"));
  PetscCall(PetscLogEventEnd(DCONTEXT_Mark, 0, 0, 0, 0));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_Sync, 0, 0, 0, 0));
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) PetscUseTypeMethod(dctx, synchronize);
  PetscCall(PetscDeviceContextSyncClearMap(dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_Sync, 0, 0, 0, 0));
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
    PetscCall(PetscInfo(nullptr, "Finalizing memory map\n"));
    PetscCallCXX(this->map = map_type{});
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
  *ptr = nullptr;
  if (PetscUnlikely(!n)) PetscFunctionReturn(0);
  PetscCall(memory_map.register_finalize());
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (dctx->ops->memalloc) {
    PetscUseTypeMethod(dctx, memalloc, clear, mtype, n, ptr);
  } else {
    PetscCheck(mtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_SUP, "Host context can only handle allocating host memory");
    PetscCall(PetscMallocA(1, clear, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, ptr));
  }

  {
    const auto ret = memory_map.map.emplace(*ptr, mtype);

    // we previously allocated the pointer (with some memtype), but now emplace has failed
    // and the new mtype doesn't match. In practice this shouldn't happen, since that
    // indicates that e.g. cudaMalloc() and cudaMallocHost() have returned identical
    // pointers, but it doesn't hurt to check
    PetscAssert(ret.second || (mtype == ret.first->second), PETSC_COMM_SELF, PETSC_ERR_LIB, "Pointer %p appears to have been previously allocated with memtype %s, which does not match new memtype %s", ret.first->first, PetscMemTypes(ret.first->second), PetscMemTypes(mtype));
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
    const auto mtype = found->second;

    PetscAssert(found != map.end(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Pointer %p was not allocated via PetscDeviceAllocate()", ptr);
    PetscCallCXX(map.erase(found));
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    if (const auto freefn = dctx->ops->memfree) {
      PetscUseTypeMethod(dctx, memfree, mtype, ptr);
    } else {
      PetscCheck(mtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_SUP, "Default PetscDeviceDealloce() can only handle freeing host memory");
      PetscCall(PetscFree(ptr));
    }
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

  `src` and `dest` cannot overlap.

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
  if (!n) PetscFunctionReturn(0);
  PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
  PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
  if (dest == src) PetscFunctionReturn(0);
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

  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (dctx->ops->memcopy) {
    PetscUseTypeMethod(dctx, memcopy, dest, src, n, mode);
  } else {
    PetscCheck(mode == PETSC_DEVICE_COPY_HTOH, PETSC_COMM_SELF, PETSC_ERR_SUP, "default PetscDeviceMemcpy() can only copy host-to-host");
    PetscCall(PetscMemcpy(dest, src, n));
  }
  if (mode == PETSC_DEVICE_COPY_HTOD) {
    PetscCall(PetscLogCpuToGpu(n));
  } else if (mode == PETSC_DEVICE_COPY_DTOH) {
    PetscCall(PetscLogGpuToCpu(n));
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
  if (!n) PetscFunctionReturn(0);
  PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer");
  if (PetscMemTypeHost(mtype)) PetscValidPointer(dest, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (dctx->ops->memset) {
    PetscUseTypeMethod(dctx, memset, mtype, dest, v, n);
  } else {
    PetscCheck(PetscMemTypeHost(mtype), PETSC_COMM_SELF, PETSC_ERR_SUP, "default PetscDeviceMemset() can only handle setting host memory");
    std::memset(dest, static_cast<int>(v), n);
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

static PetscErrorCode PetscDeviceContextGetNullContextForDevice_Private(PetscBool user_set_device, PetscDevice device, PetscDeviceContext *dctx) {
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
      PetscCall(PetscDeviceContextSetDevice_Internal(*dctx, device, user_set_device));
      PetscCall(PetscStrallocpy((std::string("null context ") + std::to_string(devid)).c_str(), &(*dctx)->name));
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
  PetscCall(PetscDeviceContextGetNullContextForDevice_Private(gctx->usersetdevice, gdev, dctx));
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
    PetscCall(PetscStrallocpy("global root", &globalContext->name));
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
  auto dtype   = std::make_pair(PETSC_DEVICE_DEFAULT(), PETSC_FALSE);
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

PetscErrorCode PetscDeviceContextView(PetscDeviceContext dctx, PetscViewer viewer) {
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(viewer), PETSCVIEWERASCII, &iascii));
  if (iascii) {
    MPI_Comm        comm;
    PetscMPIInt     rank;
    PetscStreamType stype;
    PetscViewer     sub;

    PetscCall(PetscObjectGetComm(PetscObjectCast(viewer), &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sub));
    PetscCall(PetscViewerASCIIPrintf(sub, "[%d] PetscDeviceContext %" PetscInt64_FMT " (%s):\n", rank, dctx->id, dctx->name));
    PetscCall(PetscViewerASCIIPushTab(sub));
    PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
    PetscCall(PetscViewerASCIIPrintf(sub, "stream type: %s\n", PetscStreamTypes[stype]));
    PetscCall(PetscViewerASCIIPushTab(sub));
    PetscCall(PetscViewerASCIIPrintf(sub, "allow orphans: %s\n", PetscBools[dctx->options.allow_orphans]));
    PetscCall(PetscViewerASCIIPopTab(sub));
    PetscCall(PetscViewerASCIIPrintf(sub, "children: %" PetscInt_FMT "\n", dctx->numChildren));
    if (dctx->numChildren) {
      PetscCall(PetscViewerASCIIPushTab(sub));
      PetscCall(PetscIntView(dctx->numChildren, dctx->childIDs, sub));
      PetscCall(PetscViewerASCIIPopTab(sub));
    }
    {
      const auto &data     = CxxDataCast(dctx);
      const auto &deps     = data->deps;
      const auto &upstream = data->upstream;

      if (const auto nup = upstream.size()) {
        const auto         cend = upstream.cend();
        std::ostringstream oss;

        PetscCall(PetscViewerASCIIPrintf(sub, "upstream parents: %zu\n", nup));
        for (auto it = upstream.cbegin(); it != cend; ++it) {
          const auto id = it->second.id;

          oss << id;
          if (id != it->first->id) oss << " (invalid)";
          if (std::next(it) != cend) oss << ", ";
        }
        PetscCall(PetscViewerASCIIPushTab(sub));
        PetscCall(PetscViewerASCIIPrintf(sub, "[%s]\n", oss.str().c_str()));
        PetscCall(PetscViewerASCIIPopTab(sub));
      }
      if (const auto nobj = deps.size()) {
        const auto         cend = deps.cend();
        std::ostringstream oss;

        PetscCall(PetscViewerASCIIPrintf(sub, "marked objects: %zu\n", nobj));
        for (auto it = deps.cbegin(); it != cend; ++it) {
          oss << *it;
          if (std::next(it) != cend) oss << ", ";
        }
        PetscCall(PetscViewerASCIIPushTab(sub));
        PetscCall(PetscViewerASCIIPrintf(sub, "[%s]\n", oss.str().c_str()));
        PetscCall(PetscViewerASCIIPopTab(sub));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(sub));
    if (const auto device = dctx->device) {
      PetscCall(PetscViewerASCIIPushTab(sub));
      PetscCall(PetscDeviceView(device, sub));
      PetscCall(PetscViewerASCIIPopTab(sub));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sub));
    PetscCall(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}
