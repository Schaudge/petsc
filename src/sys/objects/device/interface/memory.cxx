#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/

#include <petsc/private/cpp/register_finalize.hpp>

#include <unordered_map>
#include <cstring> // std::memset

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_AUTO) == 4, "");

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
