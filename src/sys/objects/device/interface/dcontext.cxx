#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/
#include <petsc/private/viewerimpl.h>
#include <petsc/private/cpputil.hpp>

#include <petsc/private/cpp/flat_map.hpp>
#include <petsc/private/cpp/object_pool.hpp>

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

    constexpr explicit parent_type(PetscDeviceContext dctx) noexcept : parent_type(PetscObjectCast(dctx)->id, PetscObjectCast(dctx)->state) { }

  private:
    // make this private, we do not want to accept any old id and state pairing
    constexpr parent_type(PetscObjectId id_, PetscObjectState state_) noexcept : id(id_), state(state_) { }
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
PETSC_DECLTYPE_AUTO_RETURNS(static_cast<PetscDeviceContextCxxData *>(PetscObjectCast(dctx)->cpp))

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
    PetscCall(PetscHeaderCreate(*dctx, PETSC_DEVICE_CONTEXT_CLASSID, "PetscDeviceContext", "", "Sys", PETSC_COMM_SELF, PetscDeviceContextDestroy, PetscDeviceContextView));
    PetscCallCXX(PetscObjectCast(*dctx)->cpp = new PetscDeviceContextCxxData);
    PetscCall(reset(*dctx, false));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext dctx)) {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying", dctx->numChildren);
    PetscTryTypeMethod(dctx, destroy);
    PetscCall(PetscDeviceDestroy(&dctx->device));
    PetscCall(PetscFree(dctx->childIDs));
    delete CxxDataCast(dctx);
    PetscCall(PetscHeaderDestroy(&dctx));
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
        PetscCall(PetscArrayzero(dctx->ops, 1));
        dctx->data = nullptr;
      }
      PetscCall(PetscHeaderDestroy_Private(PetscObjectCast(dctx), PETSC_TRUE));
      dctx->numChildren = 0;
      dctx->setup       = PETSC_FALSE;
      dctx->contained   = PETSC_FALSE;
      PetscCall(CxxDataCast(dctx)->clear());
      // don't deallocate the child array, rather just zero it out
      PetscCall(PetscArrayzero(dctx->childIDs, dctx->maxNumChildren));
    }
    dctx->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscCall(reset_options_(dctx->options));
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
PetscErrorCode        PetscDeviceContextDestroy(PetscDeviceContext *dctx) {
         PetscFunctionBegin;
         PetscValidPointer(dctx, 1);
         if (!*dctx) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(DCONTEXT_Destroy, *dctx, 0, 0, 0));
         if (--(PetscObjectCast(*dctx)->refct) <= 0) {
           PetscCall(PetscDeviceContextCheckNotOrphaned(*dctx));
           // std::move of the expression of the trivially-copyable type 'PetscDeviceContext' (aka
           // '_n_PetscDeviceContext *') has no effect; remove std::move() [performance-move-const-arg]
           // can't remove std::move, since reclaim only takes r-value reference
           PetscCall(contextPool.reclaim(std::move(*dctx))); // NOLINT (performance-move-const-arg)
  }
         PetscCall(PetscLogEventEnd(DCONTEXT_Destroy, *dctx, 0, 0, 0));
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
    PetscCall(PetscLogEventBegin(DCONTEXT_ChangeStream, dctx, 0, 0, 0));
    PetscUseTypeMethod(dctx, changestreamtype, type);
    PetscCall(PetscLogEventEnd(DCONTEXT_ChangeStream, dctx, 0, 0, 0));
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
static PetscErrorCode PetscDeviceContextSetDevice_Private(PetscDeviceContext dctx, PetscDevice device, PetscBool user_set) {
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change its device
  PetscValidDeviceContext(dctx, 1);
  PetscValidDevice(device, 2);
  if (dctx->device && (dctx->device->id == device->id)) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(DCONTEXT_SetDevice, dctx, 0, 0, 0));
  if (const auto destroy = dctx->ops->destroy) PetscCall((*destroy)(dctx));
  PetscCall(PetscDeviceDestroy(&dctx->device));
  PetscCall(PetscMemzero(dctx->ops, sizeof(*dctx->ops)));
  PetscCall((*device->ops->createcontext)(dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_SetDevice, dctx, 0, 0, 0));
  PetscCall(PetscDeviceReference_Internal(device));
  dctx->device        = device;
  dctx->setup         = PETSC_FALSE;
  dctx->usersetdevice = user_set;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext dctx, PetscDeviceType type) {
  PetscDevice device;

  PetscFunctionBegin;
  PetscCall(PetscDeviceGetDefaultForType_Internal(type, &device));
  PetscCall(PetscDeviceContextSetDevice_Private(dctx, device, PETSC_FALSE));
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
  PetscCall(PetscDeviceContextSetDevice_Private(dctx, device, PETSC_TRUE));
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
  PetscAssert(dctx->device, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " has no attached PetscDevice to get", PetscObjectCast(dctx)->id);
  *device = dctx->device;
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

    PetscCall(PetscInfo(nullptr, "PetscDeviceContext %" PetscInt64_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n", PetscObjectCast(dctx)->id, PetscDeviceTypes[default_dtype]));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, default_dtype));
  }
  PetscCall(PetscLogEventBegin(DCONTEXT_SetUp, dctx, 0, 0, 0));
  PetscUseTypeMethod(dctx, setup);
  PetscCall(PetscLogEventEnd(DCONTEXT_SetUp, dctx, 0, 0, 0));
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextDuplicate_Private(PetscDeviceContext dctx, PetscStreamType stype, PetscDeviceContext *dctxdup) {
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DCONTEXT_Duplicate, dctx, 0, 0, 0));
  PetscCall(PetscDeviceContextCreate(dctxdup));
  PetscCall(PetscDeviceContextSetStreamType(*dctxdup, stype));
  if (const auto device = dctx->device) { PetscCall(PetscDeviceContextSetDevice_Private(*dctxdup, device, dctx->usersetdevice)); }
  (*dctxdup)->options = dctx->options;
  PetscCall(PetscDeviceContextSetUp(*dctxdup));
  PetscCall(PetscLogEventEnd(DCONTEXT_Duplicate, dctx, 0, 0, 0));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_QueryIdle, dctx, 0, 0, 0));
  PetscUseTypeMethod(dctx, query, idle);
  PetscCall(PetscLogEventEnd(DCONTEXT_QueryIdle, dctx, 0, 0, 0));
  PetscCall(PetscInfo(nullptr, "PetscDeviceContext ('%s', id %" PetscInt64_FMT ") %s idle\n", PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", PetscObjectCast(dctx)->id, *idle ? "was" : "was not"));
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
  PetscObject aobj;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxa));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxb));
  PetscCheckCompatibleDeviceContexts(dctxa, 1, dctxb, 2);
  if (dctxa == dctxb) PetscFunctionReturn(0);
  aobj = PetscObjectCast(dctxa);
  PetscCall(PetscLogEventBegin(DCONTEXT_WaitForCtx, dctxa, dctxb, 0, 0));
  PetscUseTypeMethod(dctxa, waitforcontext, dctxb);
  PetscCallCXX(CxxDataCast(dctxa)->upstream[dctxb] = PetscDeviceContextCxxData::parent_type{dctxb});
  PetscCall(PetscLogEventEnd(DCONTEXT_WaitForCtx, dctxa, dctxb, 0, 0));
  PetscCall(PetscDebugInfo("dctx %" PetscInt64_FMT " waiting on dctx %" PetscInt64_FMT "\n", aobj->id, PetscObjectCast(dctxb)->id));
  PetscCall(PetscObjectStateIncrease(aobj));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_Fork, dctx, 0, 0, 0));
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
      auto &childctx = (*dsub)[i];

      /* create the child context in the image of its parent */
      PetscCall(PetscDeviceContextDuplicate_Private(dctx, stype, &childctx));
      PetscCall(PetscDeviceContextWaitForContext(childctx, dctx));
      /* register the child with its parent */
      PetscCall(PetscObjectGetId(PetscObjectCast(childctx), &childID));
      if (PetscDefined(USE_DEBUG_AND_INFO)) {
        PetscCallCXX(idList += std::to_string(childID));
        if (ninput != 1) PetscCallCXX(idList += ", ");
      }
      --ninput;
    }
  }
  PetscCall(PetscLogEventEnd(DCONTEXT_Fork, dctx, 0, 0, 0));
  PetscCall(PetscDebugInfo("Forked %" PetscInt_FMT " children from parent %" PetscInt64_FMT " with IDs: %s\n", n, PetscObjectCast(dctx)->id, idList.c_str()));
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
  PetscCall(PetscLogEventBegin(DCONTEXT_Join, dctx, 0, 0, 0));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx, 1, (*dsub)[i], 4);
    PetscCall(PetscDeviceContextWaitForContext(dctx, (*dsub)[i]));
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      PetscCallCXX(idList += std::to_string(PetscObjectCast((*dsub)[i])->id));
      if (i + 1 < n) PetscCallCXX(idList += ", ");
    }
  }

  /* now we handle the aftermath */
  switch (joinMode) {
  case PETSC_DEVICE_CONTEXT_JOIN_DESTROY: {
    const auto children = dctx->childIDs;
    const auto maxchild = dctx->maxNumChildren;
    auto      &nchild   = dctx->numChildren;
    PetscInt   j        = 0;

    PetscCheck(n <= nchild, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Trying to destroy %" PetscInt_FMT " children of a parent context that only has %" PetscInt_FMT " children, likely trying to restore to wrong parent", n, nchild);
    /* update child count while it's still fresh in memory */
    nchild -= n;
    for (PetscInt i = 0; i < maxchild; ++i) {
      if (children[i] && (children[i] == PetscObjectCast((*dsub)[j])->id)) {
        /* child is one of ours, can destroy it */
        PetscCall(PetscDeviceContextDestroy((*dsub) + j));
        /* reset the child slot */
        children[i] = 0;
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
  PetscCall(PetscLogEventEnd(DCONTEXT_Join, dctx, 0, 0, 0));

  PetscCall(PetscDebugInfo("Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt64_FMT ", mode %s with IDs: %s\n", n, PetscObjectCast(dctx)->id, PetscDeviceContextJoinModes[joinMode], idList.c_str()));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextSyncClearMap(PetscDeviceContext);
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
PetscErrorCode        PetscDeviceContextSynchronize(PetscDeviceContext dctx) {
         PetscFunctionBegin;
         PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
         PetscCall(PetscLogEventBegin(DCONTEXT_Sync, 0, 0, 0, 0));
         /* if it isn't setup there is nothing to sync on */
         if (dctx->setup) PetscCall((*dctx->ops->synchronize)(dctx));
  PetscCall(PetscDeviceContextSyncClearMap(dctx));
         PetscCall(PetscLogEventEnd(DCONTEXT_Sync, 0, 0, 0, 0));
         PetscFunctionReturn(0);
}

class PetscEventAllocator : public Petsc::AllocatorBase<PetscEvent> {
public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(PetscEvent *event, PetscDeviceType dtype)) {
    PetscFunctionBegin;
    PetscCall(PetscNew(event));
    (*event)->type = dtype;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscEvent event)) {
    PetscFunctionBegin;
    if (const auto destroy = event->ops->destroy) PetscCall((*destroy)(event));
    PetscCall(PetscFree(event));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(PetscEvent event, PetscDeviceType dtype)) {
    PetscFunctionBegin;
    if (event->type != dtype) {
      if (const auto destroy = event->ops->destroy) {
        PetscCall((*destroy)(event));
        PetscCall(PetscArrayzero(event->ops, 1));
        event->data = nullptr;
      } else {
        PetscCheck(!event->data, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not sure how to handle event with data ptr but no destroy method!");
      }
      event->type       = dtype;
      event->dctx_id    = 0;
      event->dctx_state = 0;
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) { return 0; }
};

static Petsc::ObjectPool<PetscEvent, PetscEventAllocator> eventPool;

// REVIEW ME
// REMOVE ME
#include <petsc/private/cupminterface.hpp>
static PetscErrorCode PetscEventCreate_Private(PetscDeviceType dtype, PetscEvent *event) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscValidPointer(event, 1);
  PetscCall(eventPool.get(*event, dtype));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventDestroy_Private(PetscEvent *event) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscValidPointer(event, 1);
  if (!*event) PetscFunctionReturn(0);
  PetscCall(eventPool.reclaim(std::move(*event))); // NOLINT
  *event = nullptr;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextRecordEvent_Private(PetscDeviceContext dctx, PetscEvent event) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscValidPointer(event, 1);
  if (dctx->ops->recordevent) {
    const auto       etype = event->type;
    const auto       pobj  = PetscObjectCast(dctx);
    PetscObjectState state;
    PetscObjectId    id;

    PetscCall(PetscObjectGetId(pobj, &id));
    PetscCall(PetscObjectStateGet(pobj, &state));
    // technically state can never be less than event->dctx_state but we include it just in
    // case
    if (id == event->dctx_id && state <= event->dctx_state) PetscFunctionReturn(0);
    // REVIEW ME:
    // TODO maybe move this to impls, as they can determine whether they can interoperate with
    // other device types more readily
    if (etype != PETSC_DEVICE_HOST && PetscDefined(USE_DEBUG)) {
      PetscDeviceType dtype;

      PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
      PetscCheck(etype == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[etype], PetscDeviceTypes[dtype]);
    }
    PetscUseTypeMethod(dctx, recordevent, event);
    event->dctx_id    = id;
    event->dctx_state = state;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceContextWaitForEvent_Private(PetscDeviceContext dctx, PetscEvent event) {
  NVTX_RANGE;
  PetscDeviceType dtype, etype;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  if (!event) PetscFunctionReturn(0);
  PetscValidPointer(event, 1);
  etype = event->type;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (dctx->ops->waitforevent) {
    PetscAssert(event->type == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[event->type], PetscDeviceTypes[dtype]);
    if (PetscObjectCast(dctx)->id == event->dctx_id) PetscFunctionReturn(0);
    // REVIEW ME:
    // TODO move this to impls, as they can determine whether they can interoperate with other
    // device types more readily
    PetscUseTypeMethod(dctx, waitforevent, event);
  } else {
    // if we don't have a wait-for-event impls, then we better check that this event does not
    // expect to be waited on. For this we take existence of the data member as "I expect
    // someone to wait on me"
    PetscCheck(!event->data, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "PetscDeviceContext (of type %s) did not have a wait-for-event implementation, and therefore could not properly handle event (of type %s)", PetscDeviceTypes[dtype], PetscDeviceTypes[etype]);
  }
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
    if (finalize_called_) PetscFunctionReturn(0);
    finalize_called_ = true;
    if (print_) {
      std::size_t        minval = (std::size_t)-1, maxval = 0, samples = 0, weighted_sum = 0;
      std::ostringstream oss;

      for (auto &&[size, count] : s_) {
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
  map_size_counter(std::string name) : print_(PETSC_FALSE), map_name(std::move(name)) { }

  template <typename T>
  PetscErrorCode count(const T &container) noexcept {
    PetscFunctionBegin;
    PetscFunctionReturn(0);
    PetscCall(this->register_finalize());
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

  PetscStackFrame() noexcept = default;

  PetscStackFrame(const char *file_, const char *func_, int line_) : file(split_on_petsc_path_(file_)), function(func_), line(line_) { }

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
  class snapshot_type {
  public:
    using id_type    = PetscObjectId;
    using event_type = PetscEvent;
    using frame_type = PetscStackFrame<PetscDefined(USE_DEBUG)>;

    explicit snapshot_type() noexcept = default;

    explicit snapshot_type(PetscDeviceContext dctx, frame_type frame) noexcept : id_(PetscObjectCast(dctx)->id), frame_(std::move(frame)) {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, init_event_(dctx, &event()));
      PetscFunctionReturnVoid();
    }

    explicit snapshot_type(PetscDeviceContext ctx, const char *file, const char *function, int line) noexcept : snapshot_type(ctx, frame_type{file, function, line}) { }

    snapshot_type(snapshot_type &&other) noexcept : id_(other.dctx_id()), event_(std::move(other.event())), frame_(std::move(other.frame())) {
      PetscCheckAbort(this != &other, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Destroying self");
      other.event() = event_type{};
    }

    ~snapshot_type() noexcept {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Private(&event()));
      PetscFunctionReturnVoid();
    }

    snapshot_type &operator=(snapshot_type &&other) noexcept {
      PetscFunctionBegin;
      if (this == &other) PetscFunctionReturn(other);
      dctx_id() = other.dctx_id();
      PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Private(&event()));
      std::swap(event(), other.event());
      frame() = std::move(other.frame());
      PetscFunctionReturn(*this);
    }

    // not copyable
    snapshot_type &operator=(const snapshot_type &) noexcept = delete;
    snapshot_type(const snapshot_type &other) noexcept       = delete;

    PETSC_NODISCARD id_type          &dctx_id() noexcept { return id_; }
    PETSC_NODISCARD const id_type    &dctx_id() const noexcept { return id_; }
    PETSC_NODISCARD event_type       &event() noexcept { return event_; }
    PETSC_NODISCARD const event_type &event() const noexcept { return event_; }
    PETSC_NODISCARD frame_type       &frame() noexcept { return frame_; }
    PETSC_NODISCARD const frame_type &frame() const noexcept { return frame_; }

    explicit operator bool() const noexcept { return static_cast<bool>(event()); }

    bool operator==(const snapshot_type &other) const noexcept { return dctx_id() == other.dctx_id() && event() == other.event() && frame() == other.frame(); }

    bool operator!=(const snapshot_type &other) const noexcept { return !(*this == other); }

  private:
    id_type    id_{};
    event_type event_{};
    frame_type frame_{};

    static PetscErrorCode init_event_(PetscDeviceContext dctx, event_type *event) noexcept {
      PetscDeviceType dtype;

      PetscFunctionBegin;
      PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
      PetscCall(PetscEventCreate_Private(dtype, event));
      PetscCall(PetscDeviceContextRecordEvent_Private(dctx, *event));
      PetscFunctionReturn(0);
    }
  };

  struct mapped_type {
    using mode_type       = PetscMemoryAccessMode;
    using dependency_type = std::vector<snapshot_type>;

    mode_type       mode = PETSC_MEMORY_ACCESS_READ;
    snapshot_type   last_write{};
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
          // if (!dep.ctx->options.allow_orphans) {
          //   wrote_to_oss_tmp = true;
          //   oss_tmp<<"  ["<<rank<<"] dctx "<<dep.ctx<<" (id "<<dep.dctx_id()<<", state "<<dep.dctx_state<<", intent "<<mode<<' '<<dep.frame()<<")\n";
          // }
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

// A mapping between PetscObjectId (i.e. some PetscObject) to the list of PetscEvent's encoding
// the last time the PetscObject was accessed
static MarkedObjectMap marked_object_map;

namespace {

struct ignore {
  template <typename... T>
  constexpr bool operator()(T &&...) const noexcept {
    return false;
  }
};

} // anonymous namespace

template <typename T = ignore>
static PetscErrorCode PetscDeviceContextMapIterVistor(PetscDeviceContext dctx, T &&callback = T{}) noexcept {
  const auto dctx_id    = PetscObjectCast(dctx)->id;
  auto      &dctx_deps  = CxxDataCast(dctx)->deps;
  auto      &object_map = marked_object_map.map;

  PetscFunctionBegin;
  PetscCall(dependency_ctr.count(dctx_deps));
  PetscCall(upstream_ctr.count(CxxDataCast(dctx)->upstream));
  for (auto &&dep : dctx_deps) {
    const auto mapit = object_map.find(dep);

    // Need this check since the final PetscDeviceContext may run through this *after* the map
    // has been finalized (and cleared), and hence might fail to find its dependencies. This is
    // perfectly valid since the user no longer cares about dangling dependencies after PETSc
    // is finalized
    if (PetscLikely(object_map.cend() != mapit)) {
      auto      &deps = mapit->second.dependencies;
      const auto end  = deps.end();
      const auto it   = std::remove_if(deps.begin(), end, [&](const MarkedObjectMap::snapshot_type &obj) { return obj.dctx_id() == dctx_id; });

      if (!std::is_same<T, ignore>::value) { PetscCall(callback(mapit, deps.cbegin(), static_cast<decltype(deps.cend())>(it))); }
      // remove ourselves
      PetscCallCXX(deps.erase(it, end));
      // continue to next object, but erase this one if it has no more dependencies
      if (deps.empty()) PetscCallCXX(object_map.erase(mapit));
    }
  }
  PetscCallCXX(dctx_deps.clear());
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

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMapIterVistor(dctx, [&](map_iterator_type mapit, dep_iterator_type it, dep_iterator_type end) {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      std::ostringstream oss;
      const auto         mode = PetscMemoryAccessModes(mapit->second.mode);

      oss << "synced dctx " << PetscObjectCast(dctx)->id << ", remaining leaves for obj " << mapit->first << ": {";
      for (; it != end; ++it) {
        oss << "[dctx " << it->dctx_id() << ", " << mode << ' ' << it->frame() << ']';
        if (std::next(it) != end) oss << ", ";
      }
      oss << '}';
      PetscCall(PetscInfo(nullptr, "%s\n", oss.str().c_str()));
    }
    PetscFunctionReturn(0);
  }));
  // aftermath, clear our set of parents (to avoid infinite recursion) and mark ourselves as no
  // longer contained (while the empty graph technically *is* always contained, it is not what
  // we mean by it)
  PetscCall(CxxDataCast(dctx)->clear());
  dctx->contained = PETSC_FALSE;
  for (auto &&upstrm : upstream_copy) {
    // check that this parent still points to what we originally thought it was
    PetscCheck(upstrm.second.id == PetscObjectCast(upstrm.first)->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Upstream dctx %" PetscInt64_FMT " no longer exists, now has id %" PetscInt64_FMT, upstrm.second.id, PetscObjectCast(upstrm.first)->id);
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
    wrote_to_oss = true;
    oss << "- PetscObject (id " << mapit->first << "), intent " << PetscMemoryAccessModes(mapit->second.mode) << ' ' << it->frame();
    if (std::distance(it, end) == 0) oss << " (orphaned)"; // we were the only dependency
    oss << '\n';
    PetscFunctionReturn(0);
  }));
  PetscCheck(!wrote_to_oss, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Destroying PetscDeviceContext ('%s', id %" PetscInt64_FMT ") would leave the following dangling (possibly orphaned) dependants:\n%s\nMust synchronize before destroying it, or allow it to be destroyed with orphans",
             PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", PetscObjectCast(dctx)->id, oss.str().c_str());
  PetscCall(CxxDataCast(dctx)->clear());
  PetscFunctionReturn(0);
}

template <bool use_debug>
static PetscErrorCode PetscDeviceContextMarkIntentFromID_Private(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> frame, const char *name) {
  const auto dctx_id             = PetscObjectCast(dctx)->id;
  auto      &marked              = marked_object_map.map[id];
  auto      &old_mode            = marked.mode;
  auto      &object_dependencies = marked.dependencies;
#define DEBUG_INFO(mess, ...) PetscDebugInfo("dctx %" PetscInt64_FMT " (%s) - obj %" PetscInt64_FMT " (%s): " mess, dctx_id, PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", id, name, ##__VA_ARGS__)

  PetscFunctionBegin;
  if ((mode == PETSC_MEMORY_ACCESS_READ) && (old_mode == mode)) {
    const auto end = object_dependencies.end();
    const auto it  = std::find_if(object_dependencies.begin(), end, [&](const MarkedObjectMap::snapshot_type &obj) { return obj.dctx_id() == dctx_id; });

    PetscCall(DEBUG_INFO("new mode (%s) COMPATIBLE with %s mode (%s), no need to serialize\n", PetscMemoryAccessModes(mode), PetscMemoryAccessModes(old_mode), object_dependencies.empty() ? "default" : "old"));

    if (it != end) {
      // we have been here before, all we must do is update our entry then we can bail
      PetscCall(DEBUG_INFO("found old self as dependency, updating\n"));
      PetscAssert(CxxDataCast(dctx)->deps.contains(id), PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext %" PetscInt64_FMT " listed as dependency for object %" PetscInt64_FMT " (%s), but does not have the object in private dependency list!", dctx_id, id, name);

      it->frame() = std::move(frame);
      PetscCall(PetscDeviceContextRecordEvent_Private(dctx, it->event()));
      PetscFunctionReturn(0);
    }

    // we have not been here before, we need only serialize with the last write event (if it
    // exists)
    PetscCall(PetscDeviceContextWaitForEvent_Private(dctx, marked.last_write.event()));
  } else {
    // we are incompatible with the previous mode
    PetscCall(DEBUG_INFO("new mode (%s) NOT COMPATIBLE with %s mode (%s), serializing then clearing (%zu) %s\n", PetscMemoryAccessModes(mode), object_dependencies.empty() ? "default" : "old", PetscMemoryAccessModes(old_mode), object_dependencies.size(),
                         object_dependencies.size() == 1 ? "dependency" : "dependencies"));
    for (const auto &dep : object_dependencies) {
      if (dep.dctx_id() == dctx_id) {
        PetscCall(DEBUG_INFO("found old self as dependency, skipping\n"));
        continue;
      }
      PetscCall(PetscDeviceContextWaitForEvent_Private(dctx, dep.event()));
    }

    // if the previous mode wrote, bump it to the previous write spot
    if (PetscMemoryAccessWrite(old_mode)) {
      PetscCheck(object_dependencies.size() == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Can only have a single writer as dependency!");
      PetscCall(DEBUG_INFO("moving last write dependency (intent %s)\n", PetscMemoryAccessModes(old_mode)));
      marked.last_write = std::move(object_dependencies.back()); // note front is equivalent
    }

    // clear out the old dependencies and update the mode, we are about to append ourselves
    object_dependencies.clear();
    old_mode = mode;
  }
  // become the new leaf by appending ourselves
  PetscCall(DEBUG_INFO("%s with intent %s\n", object_dependencies.empty() ? "dependency list is empty, creating new leaf" : "appending to existing leaves", PetscMemoryAccessModes(mode)));
  PetscCallCXX(object_dependencies.emplace_back(dctx, std::move(frame)));
  PetscCallCXX(CxxDataCast(dctx)->deps.emplace(id));
  PetscFunctionReturn(0);
#undef DEBUG_INFO
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
    if (static_cast<std::size_t>(devid) >= ctxlist.size()) PetscCallCXX(ctxlist.resize(devid + 1));
    if (PetscUnlikely(!ctxlist[devid])) {
      // we have not seen this device before
      PetscCall(PetscInfo(nullptr, "Initializing null PetscDeviceContext (of type %s) for device %" PetscInt_FMT "\n", PetscDeviceTypes[dtype], devid));
      PetscCall(PetscDeviceContextCreate(dctx));
      {
        const auto pobj = PetscObjectCast(*dctx);
        auto       name = "null context " + std::to_string(devid);

        PetscCall(PetscObjectSetName(pobj, name.c_str()));
        std::replace(name.begin(), name.end(), ' ', '_');
        name += '_';
        PetscCall(PetscObjectSetOptionsPrefix(pobj, name.c_str()));
      }
      PetscCall(PetscDeviceContextSetStreamType(*dctx, PETSC_STREAM_GLOBAL_BLOCKING));
      PetscCall(PetscDeviceContextSetDevice_Private(*dctx, device, user_set_device));
      PetscCall(PetscDeviceContextSetUp(*dctx));
      // would use ctxlist.cbegin() but GCC 4.8 can't handle const iterator insert!
      PetscCallCXX(ctxlist.insert(std::next(ctxlist.begin(), devid), *dctx));
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
+ comm - MPI communicator on which to query the options database (optional)
- dctx - The `PetscDeviceContext` to configure

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Options Database:
+ -device_context_stream_type - type of stream to create inside the `PetscDeviceContext` -
   `PetscDeviceContextSetStreamType()`
- -device_context_device_type - the type of `PetscDevice` to attach by default - `PetscDeviceType`

  Notes:
  The user may pass `MPI_COMM_NULL` for `comm` in which case the communicator of `dctx` is
  used (which is always `PETSC_COMM_SELF`).

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextSetDevice()`
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, PetscDeviceContext dctx) {
  const auto  pobj    = PetscObjectCast(dctx);
  const char *prefix  = nullptr;
  auto        dtype   = std::make_pair(PETSC_DEVICE_DEFAULT(), PETSC_FALSE);
  auto        stype   = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE, PETSC_FALSE);
  auto        orphans = std::make_pair(PETSC_TRUE, PETSC_FALSE);

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 2);
  if (comm == MPI_COMM_NULL) PetscCall(PetscObjectGetComm(pobj, &comm));
  PetscCall(PetscObjectGetOptionsPrefix(pobj, &prefix));
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
    const auto      pobj = PetscObjectCast(dctx);
    MPI_Comm        comm;
    PetscMPIInt     rank;
    PetscStreamType stype;
    PetscViewer     sub;

    PetscCall(PetscObjectGetComm(PetscObjectCast(viewer), &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sub));
    PetscCall(PetscObjectPrintClassNamePrefixType(pobj, sub));
    PetscCall(PetscViewerASCIIPushTab(sub));
    PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
    PetscCall(PetscViewerASCIIPrintf(sub, "stream type: %s\n", PetscStreamTypes[stype]));
    PetscCall(PetscViewerASCIIPushTab(sub));
    PetscCall(PetscViewerASCIIPrintf(sub, "allow orphans: %s\n", PetscBools[dctx->options.allow_orphans]));
    PetscCall(PetscViewerASCIIPopTab(sub));
    PetscCall(PetscViewerASCIIPrintf(sub, "children: %" PetscInt_FMT "\n", dctx->numChildren));
    if (const auto nchild = dctx->numChildren) {
      PetscCall(PetscViewerASCIIPushTab(sub));
      // REVIEW ME: fix this, this is busted
      PetscCall(PetscIntView(nchild, reinterpret_cast<PetscInt *>(dctx->childIDs), sub));
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
          if (id != PetscObjectCast(it->first)->id) oss << " (invalid)";
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
