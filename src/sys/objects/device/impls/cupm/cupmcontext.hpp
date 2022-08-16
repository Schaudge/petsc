#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#if defined(__cplusplus)
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/logimpl.h>

#include "../segmentedmempool.hpp"
#include "cupmthrustutility.hpp"
#include "cupmallocator.hpp"

#include <array>

namespace Petsc {

namespace device {

namespace cupm {

namespace impl {

template <DeviceType T>
class DeviceContext : BlasInterface<T> {
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t, T);

private:
  template <typename H, std::size_t>
  struct HandleTag {
    using type = H;
  };
  using stream_tag = HandleTag<cupmStream_t, 0>;
  using blas_tag   = HandleTag<cupmBlasHandle_t, 1>;
  using solver_tag = HandleTag<cupmSolverHandle_t, 2>;

  using stream_type = CUPMStream<T>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS {
    stream_type stream{};
    cupmEvent_t event{};
    cupmEvent_t begin{}; // timer-only
    cupmEvent_t end{};   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
    cupmBlasHandle_t   blas{};
    cupmSolverHandle_t solver{};

    PETSC_NODISCARD cupmStream_t get(stream_tag) const noexcept {
      return this->stream.get_stream();
    }
    PETSC_NODISCARD cupmBlasHandle_t get(blas_tag) const noexcept {
      return this->blas;
    }
    PETSC_NODISCARD cupmSolverHandle_t get(solver_tag) const noexcept {
      return this->solver;
    }
  };

private:
  static bool                                                     initialized_;
  static std::array<cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_CXX_COMPAT_DECL(constexpr auto impls_cast_(PetscDeviceContext ptr))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<PetscDeviceContext_IMPLS *>(ptr->data));

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent CUPMBLAS_HANDLE_CREATE()) {
    return T == DeviceType::CUDA ? CUBLAS_HANDLE_CREATE : HIPBLAS_HANDLE_CREATE;
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscLogEvent CUPMSOLVER_HANDLE_CREATE()) {
    return T == DeviceType::CUDA ? CUSOLVER_HANDLE_CREATE : HIPSOLVER_HANDLE_CREATE;
  }

  // this exists purely to satisfy the compiler so the tag-based dispatch works for the other
  // handles
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(stream_tag, PetscDeviceContext)) {
    return 0;
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmBlasHandle_t &handle)) {
    PetscLogEvent event;

    PetscFunctionBegin;
    if (PetscLikely(handle)) PetscFunctionReturn(0);
    PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
    PetscCall(PetscLogEventBegin(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
    for (auto i = 0; i < 3; ++i) {
      auto cberr = cupmBlasCreate(&handle);
      if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) PetscCallCUPMBLAS(cberr);
      if (i != 2) {
        PetscCall(PetscSleep(3));
        continue;
      }
      PetscCheck(cberr == CUPMBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "Unable to initialize %s", cupmBlasName());
    }
    PetscCall(PetscLogEventEnd(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
    PetscCall(PetscLogEventResume_Internal(event));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(blas_tag, PetscDeviceContext dctx)) {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = blashandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCallCUPMBLAS(cupmBlasSetStream(handle, dci->stream.get_stream()));
    dci->blas = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmSolverHandle_t &handle)) {
    PetscLogEvent event;

    PetscFunctionBegin;
    PetscCall(PetscLogPauseCurrentEvent_Internal(&event));
    PetscCall(PetscLogEventBegin(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
    PetscCall(cupmBlasInterface_t::InitializeHandle(handle));
    PetscCall(PetscLogEventEnd(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
    PetscCall(PetscLogEventResume_Internal(event));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(solver_tag, PetscDeviceContext dctx)) {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = solverhandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCall(cupmBlasInterface_t::SetHandleStream(handle, dci->stream.get_stream()));
    dci->solver = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode check_current_device_(PetscDeviceContext dctxl, PetscDeviceContext dctxr)) {
    const auto devidl = dctxl->device->deviceId, devidr = dctxr->device->deviceId;

    PetscFunctionBegin;
    PetscCheck(devidl == devidr, PETSC_COMM_SELF, PETSC_ERR_GPU, "Device contexts must be on the same device; dctx A (id %" PetscInt64_FMT " device id %" PetscInt_FMT ") dctx B (id %" PetscInt64_FMT " device id %" PetscInt_FMT ")",
               PetscObjectCast(dctxl)->id, devidl, PetscObjectCast(dctxr)->id, devidr);
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidl));
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidr));
    PetscCallCUPM(cupmSetDevice(static_cast<int>(devidl)));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(auto check_current_device_(PetscDeviceContext dctx))
  PETSC_DECLTYPE_AUTO_RETURNS(check_current_device_(dctx, dctx));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode finalize_()) {
    PetscFunctionBegin;
    for (auto &&handle : blashandles_) {
      if (handle) {
        PetscCallCUPMBLAS(cupmBlasDestroy(handle));
        handle = nullptr;
      }
    }
    for (auto &&handle : solverhandles_) {
      if (handle) {
        PetscCall(cupmBlasInterface_t::DestroyHandle(handle));
        handle = nullptr;
      }
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  template <typename Allocator, typename PetscType = typename Allocator::value_type>
  PETSC_CXX_COMPAT_DECL(::Petsc::memory::SegmentedMemoryPool<PetscType, stream_type, Allocator> &managed_pool_()) {
    static ::Petsc::memory::SegmentedMemoryPool<PetscType, stream_type, Allocator> pool;
    return pool;
  }

  template <template <DeviceType, typename> class Allocator, typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_managed_values_(PetscDeviceContext, PetscManagedType, PetscMemoryAccessMode, PetscType *&, const PetscType *, PetscOffloadMask, cupmMemcpyKind_t, PetscType **));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode check_memtype_(PetscMemType mtype, const char mess[])) {
    PetscFunctionBegin;
    PetscCheck(PetscMemTypeHost(mtype) || (mtype == PETSC_MEMTYPE_DEVICE) || (mtype == cupmDeviceTypeToPetscMemType()), PETSC_COMM_SELF, PETSC_ERR_SUP, "%s device context can only handle %s (pinned) host or device memory", cupmName(), mess);
    PetscFunctionReturn(0);
  }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize());
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext, PetscBool *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext));
  template <typename Handle_t>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getHandle(PetscDeviceContext, void *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memAlloc(PetscDeviceContext, PetscBool, PetscMemType, std::size_t, void **));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memFree(PetscDeviceContext, PetscMemType, void *));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memCopy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, std::size_t, PetscDeviceCopyMode));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memSet(PetscDeviceContext, PetscMemType, void *, PetscInt, std::size_t));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext, PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext, PetscManagedType, PetscMemType, PetscMemoryAccessMode, PetscType **));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode applyOperatorType(PetscDeviceContext, PetscManagedType, PetscOperatorType, PetscMemType, const PetscType *, PetscManagedType));

  // not a PetscDeviceContext method, this registers the class
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize());

  const struct _DeviceContextOps ops = {
    destroy,
    changeStreamType,
    setUp,
    query,
    waitForContext,
    synchronize,
    getHandle<blas_tag>,
    getHandle<solver_tag>,
    getHandle<stream_tag>,
    beginTimer,
    endTimer,
    memAlloc,
    memFree,
    memCopy,
    memSet,
    destroyManagedType<PetscScalar, PetscManagedScalar>,
    getManagedTypeValues<PetscScalar, PetscManagedScalar>,
    applyOperatorType<PetscScalar, PetscManagedScalar>,
    destroyManagedType<PetscReal, PetscManagedReal>,
    getManagedTypeValues<PetscReal, PetscManagedReal>,
    applyOperatorType<PetscReal, PetscManagedReal>,
    destroyManagedType<PetscInt, PetscManagedInt>,
    getManagedTypeValues<PetscInt, PetscManagedInt>,
    applyOperatorType<PetscInt, PetscManagedInt>,
  };
};

// not a PetscDeviceContext method, this initializes the CLASS
template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::initialize()) {
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) {
    cupmMemPool_t mempool;
    uint64_t      threshold = UINT64_MAX;

    initialized_ = true;
    PetscCallCUPM(cupmDeviceGetMemPool(&mempool, 0));
    PetscCallCUPM(cupmMemPoolSetAttribute(mempool, cupmMemPoolAttrReleaseThreshold, &threshold));
    blashandles_.fill(nullptr);
    solverhandles_.fill(nullptr);
    PetscCall(PetscRegisterFinalize(finalize_));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::initialize()) {
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) {
    initialized_ = true;
    PetscCall(PetscRegisterFinalize(finalize_));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  if (const auto dci = impls_cast_(dctx)) {
    PetscCall(dci->stream.destroy());
    if (dci->event) PetscCallCUPM(cupmEventDestroy(dci->event));
    if (dci->begin) PetscCallCUPM(cupmEventDestroy(dci->begin));
    if (dci->end) PetscCallCUPM(cupmEventDestroy(dci->end));
    delete dci;
    dctx->data = nullptr;
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype)) {
  NVTX_RANGE;
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(dci->stream.destroy());
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto dci   = impls_cast_(dctx);
  auto      &event = dci->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(dci->stream.change_type(dctx->streamType));
  if (!event) PetscCall(cupm_fast_event_pool<T>().get(event));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle)) {
  NVTX_RANGE;
  cupmError_t cerr;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  *idle = PETSC_TRUE;
  cerr  = cupmStreamQuery(impls_cast_(dctx)->stream.get_stream());
  if (cerr == cupmSuccess) PetscFunctionReturn(0);
  // somethings gone wrong
  if (PetscUnlikely(cerr != cupmErrorNotReady)) PetscCallCUPM(cerr);
  *idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)) {
  NVTX_RANGE;
  const auto dcib  = impls_cast_(dctxb);
  const auto event = dcib->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctxa, dctxb));
  PetscCallCUPM(cupmEventRecord(event, dcib->stream.get_stream()));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream.get_stream(), event, 0));
  PetscFunctionReturn(0);
}

static PetscInt nsync            = 0;
static PetscInt nsync_req        = 0;
static auto     nsync_registered = false;

static PetscErrorCode PrintSyncStats() {
  auto print = PETSC_FALSE, flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_stats", &print, &flg));
  print = (PetscBool)(print && flg);
  if (!print) PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_sync_stats", &print, &flg));
  if (print && flg) { PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of synchronizations %" PetscInt_FMT ", required %" PetscInt_FMT "\n", nsync, nsync_req)); }
  nsync            = 0;
  nsync_req        = 0;
  nsync_registered = false;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto dci    = impls_cast_(dctx);
  const auto event  = dci->event;
  const auto stream = dci->stream.get_stream();

  PetscFunctionBegin;
  if (PetscUnlikely(!nsync_registered)) {
    PetscCall(PetscRegisterFinalize(PrintSyncStats));
    nsync_registered = true;
  }
  ++nsync;
  if (cupmStreamQuery(stream) == cupmSuccess) PetscFunctionReturn(0);
  PetscCallCUPM(cupmGetLastError());
  ++nsync_req;
  PetscCallCUPM(cupmStreamSynchronize(stream));
  // if (!idle) {
  // in case anything was queued on the event
  // PetscCallCUPM(cupmEventRecord(event,dci->stream));
  // PetscCallCUPM(cupmEventSynchronize(event));
  //}
  // REMOVE ME (maybe?)
  //PetscCallCUPM(cupmStreamSynchronize(stream));
  // try pruning our managed pools now that we have synchronized
#if 0
  PetscCall(managed_pool_<HostAllocator<PetscScalar>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscScalar>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<HostAllocator<PetscReal>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscReal>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<HostAllocator<PetscInt>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscInt>>().pruneEmptyBlocks(stream));
#endif
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{}, dctx));
  *static_cast<typename handle_t::type *>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx)) {
  NVTX_RANGE;
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    PetscAssert(!dci->end, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Don't have a 'begin' event, but somehow have an end event");
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin, dci->stream.get_stream()));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed)) {
  NVTX_RANGE;
  float      gtime;
  const auto dci = impls_cast_(dctx);
  const auto end = dci->end;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscCallCUPM(cupmEventRecord(end, dci->stream.get_stream()));
  PetscCallCUPM(cupmEventSynchronize(end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime, dci->begin, end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memAlloc(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, std::size_t n, void **dest)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "allocating"));
  if (PetscMemTypeHost(mtype)) {
    PetscCallCUPM(cupmMallocHost(dest, n));
    if (clear) std::memset(*dest, 0, n);
  } else {
    const auto stream = impls_cast_(dctx)->stream.get_stream();

    PetscCallCUPM(cupmMallocAsync(dest, n, stream));
    if (clear) PetscCallCUPM(cupmMemsetAsync(*dest, 0, n, stream));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memFree(PetscDeviceContext dctx, PetscMemType mtype, void *ptr)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "freeing"));
  if (!ptr) PetscFunctionReturn(0);
  if (PetscMemTypeHost(mtype)) {
    PetscCallCUPM(cupmFreeHost(ptr));
  } else {
    PetscCallCUPM(cupmFreeAsync(ptr, impls_cast_(dctx)->stream.get_stream()));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode)) {
  NVTX_RANGE;
  const auto stream = impls_cast_(dctx)->stream.get_stream();

  PetscFunctionBegin;
  // can't use PetscCUPMMemcpyAsync here since we don't know sizeof(*src)...
  if (mode == PETSC_DEVICE_COPY_HTOH) {
    if (cupmStreamQuery(stream) == cupmSuccess) {
      PetscCall(PetscMemcpy(dest, src, n));
      PetscFunctionReturn(0);
    }
    PetscCallCUPM(cupmGetLastError());
  }
  PetscCall(cupmMemcpyAsync(dest, src, n, PetscDeviceCopyModeToCUPMMemcpyKind(mode), stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memSet(PetscDeviceContext dctx, PetscMemType mtype, void *ptr, PetscInt v, std::size_t n)) {
  NVTX_RANGE;
  auto vint = static_cast<int>(v);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "zeroing"));
  if (PetscMemTypeHost(mtype)) {
    // must call public sync to prune the dependency graph
    PetscCall(PetscDeviceContextSynchronize(dctx));
    std::memset(ptr, vint, n);
  } else {
    PetscCallCUPM(cupmMemsetAsync(ptr, vint, n, impls_cast_(dctx)->stream.get_stream()));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroyManagedType(PetscDeviceContext dctx, PetscManagedType scal)) {
  NVTX_RANGE;
  const auto &stream     = impls_cast_(dctx)->stream;
  auto       &host_ptr   = scal->host;
  auto       &device_ptr = scal->device;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  // try returning them to the pool
  PetscCall(managed_pool_<HostAllocator<T, PetscType>>().release(&host_ptr, &stream));
  // not freed, indicating the pool doesn't own it, now check if it is our responsibility to
  // get rid of it
  if (host_ptr && (scal->h_cmode == PETSC_OWN_POINTER)) {
    auto mtype      = PETSC_MEMTYPE_HOST;
    auto registered = PETSC_FALSE, managed = PETSC_FALSE;

    // if the pointer is managed memory we need to call cupmFree() on it
    PetscCall(PetscCUPMGetMemType(host_ptr, &mtype, &registered, &managed));
    if (PetscMemTypeDevice(mtype) || managed) {
      PetscCallCUPM(cupmFreeAsync(host_ptr, stream.get_stream()));
    } else if (registered) {
      PetscCallCUPM(cupmFreeHost(host_ptr));
    } else {
      PetscCall(PetscFree(host_ptr));
    }
  }
  PetscCall(managed_pool_<DeviceAllocator<T, PetscType>>().release(&device_ptr, &stream));
  // same deal with device pointer
  if (device_ptr && (scal->d_cmode == PETSC_OWN_POINTER)) { PetscCallCUPM(cupmFreeAsync(device_ptr, stream.get_stream())); }
  PetscFunctionReturn(0);
}

// this should by all means be a lambda in getManagedTypeValues(), but since you can't make
// templated lambdas until either C++14 or for real in C++20 it is a function instead...
template <DeviceType T>
template <template <DeviceType, typename> class Allocator, typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::copy_managed_values_(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType *&dest, const PetscType *src, PetscOffloadMask requested_mask, cupmMemcpyKind_t direction, PetscType **ptr)) {
  NVTX_RANGE;
  const auto &stream = impls_cast_(dctx)->stream;
  const auto  n      = scal->n;
  auto       &mask   = scal->mask;

  PetscFunctionBegin;
  PetscAssert(requested_mask != PETSC_OFFLOAD_BOTH, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have %s!", PetscOffloadMasks(requested_mask));
  if (PetscUnlikely(!dest)) {
    // no pointer? get one first
    PetscCall(managed_pool_<Allocator<T, PetscType>>().get(n, &dest, &stream));
    if (mask == PETSC_OFFLOAD_UNALLOCATED) {
      PetscAssert(!src, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offload mask is PETSC_OFFLOAD_UNALLOCATED but have src pointer %p!", src);
      mask = requested_mask;
    }
  }
  PetscAssert(mask != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Still have offload unallocated!");
  *ptr = dest;
  // no need to do anything if we already match the desired offload
  if (mask == requested_mask) PetscFunctionReturn(0);
  // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
  // it now
  if (PetscMemoryAccessRead(mode) && src && (mask != PETSC_OFFLOAD_BOTH)) {
    PetscCall(PetscCUPMMemcpyAsync(dest, src, n, direction, stream.get_stream()));
    mask = PETSC_OFFLOAD_BOTH;
  }
  // if we have any kind of write then mask is set to the specific requested version (which
  // must not be OFFLOAD_BOTH)
  if (PetscMemoryAccessWrite(mode)) mask = requested_mask;
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr)) {
  NVTX_RANGE;
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  switch (mtype) {
  case PETSC_MEMTYPE_HOST: PetscCall(copy_managed_values_<HostAllocator>(dctx, scal, mode, scal->host, scal->device, PETSC_OFFLOAD_CPU, cupmMemcpyDeviceToHost, ptr)); break;
  case PETSC_MEMTYPE_DEVICE: PetscCall(copy_managed_values_<DeviceAllocator>(dctx, scal, mode, scal->device, scal->host, PETSC_OFFLOAD_GPU, cupmMemcpyHostToDevice, ptr)); break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemType must be either PETSC_MEMTYPE_HOST (%d) or PETSC_MEMTYPE_DEVICE (%d) not %d", static_cast<int>(PETSC_MEMTYPE_HOST), static_cast<int>(PETSC_MEMTYPE_DEVICE), static_cast<int>(mtype));
    break;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedInt scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscInt **ptr) {
  NVTX_RANGE;
  return PetscManagedIntGetValues(dctx, scal, mtype, mode, sync, ptr);
}

static inline PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedReal scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscReal **ptr) {
  NVTX_RANGE;
  return PetscManagedRealGetValues(dctx, scal, mtype, mode, sync, ptr);
}

static inline PetscErrorCode PetscManagedTypeGetValues(PetscDeviceContext dctx, PetscManagedScalar scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscScalar **ptr) {
  NVTX_RANGE;
  return PetscManagedScalarGetValues(dctx, scal, mtype, mode, sync, ptr);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::applyOperatorType(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, PetscManagedType ret)) {
  NVTX_RANGE;
  const auto in_place   = !ret || ret == scal;
  const auto src_access = in_place ? PETSC_MEMORY_ACCESS_READ_WRITE : PETSC_MEMORY_ACCESS_READ;
  const auto n          = scal->n;
  const auto stream     = impls_cast_(dctx)->stream.get_stream();
  PetscType *ptr = nullptr, *retptr = nullptr;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(PetscManagedTypeGetValues(dctx, scal, mtype, src_access, PETSC_FALSE, &ptr));
  //PetscCall(getManagedTypeValues(dctx,scal,mtype,src_access,&ptr));
  if (in_place) {
    retptr = ptr;
  } else {
    PetscCall(PetscManagedTypeGetValues(dctx, ret, mtype, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &retptr));
    //PetscCall(getManagedTypeValues(dctx,ret,mtype,PETSC_MEMORY_ACCESS_WRITE,&retptr));
  }
  // REVIEW ME: need to somehow handle having rhs be host or device memory!
  switch (otype) {
  case PETSC_OPERATOR_PLUS: PetscCall(ThrustApplyPointwise<T>(stream, make_shift_operator(rhs, thrust::plus<PetscType>{}), n, ptr, retptr)); break;
  case PETSC_OPERATOR_MINUS: PetscCall(ThrustApplyPointwise<T>(stream, make_shift_operator(rhs, thrust::minus<PetscType>{}), n, ptr, retptr)); break;
  case PETSC_OPERATOR_MULTIPLY: PetscCall(ThrustApplyPointwise<T>(stream, make_shift_operator(rhs, thrust::multiplies<PetscType>{}), n, ptr, retptr)); break;
  case PETSC_OPERATOR_DIVIDE: PetscCall(ThrustApplyPointwise<T>(stream, make_shift_operator(rhs, thrust::divides<PetscType>{}), n, ptr, retptr)); break;
  case PETSC_OPERATOR_EQUAL: PetscCall(ThrustSet<T>(stream, n, retptr, rhs)); break;
  }
  PetscFunctionReturn(0);
}

// initialize the static member variables

template <DeviceType T>
bool DeviceContext<T>::initialized_ = false;

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

} // namespace impl

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = impl::DeviceContext<DeviceType::CUDA>;
using CUPMContextHip  = impl::DeviceContext<DeviceType::HIP>;

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS) ::Petsc::device::cupm::impl::DeviceContext<::Petsc::device::cupm::DeviceType::IMPLS>::PetscDeviceContext_IMPLS

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCDEVICECONTEXTCUDA_HPP
