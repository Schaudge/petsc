#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/logimpl.h>

#include "../segmentedmempool.hpp"
#include "cupmthrustutility.hpp"

#include <array>
#include <functional>

namespace Petsc {

namespace Device {

namespace CUPM {

namespace Impl {

template <DeviceType T>
class DeviceContext : BlasInterface<T> {
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t, T);

private:
  // for tag-based dispatch of handle retrieval
  template <typename H, std::size_t>
  struct HandleTag {
    using type = H;
  };
  using stream_tag = HandleTag<cupmStream_t, 0>;
  using blas_tag   = HandleTag<cupmBlasHandle_t, 1>;
  using solver_tag = HandleTag<cupmSolverHandle_t, 2>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS {
    cupmStream_t stream;
    cupmEvent_t  event;
    cupmEvent_t  begin; // timer-only
    cupmEvent_t  end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse;
#endif
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;

    PETSC_NODISCARD auto get(stream_tag) const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->stream);
    PETSC_NODISCARD auto get(blas_tag) const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->blas);
    PETSC_NODISCARD auto get(solver_tag) const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->solver);
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
    PetscCallCUPMBLAS(cupmBlasSetStream(handle, dci->stream));
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
    PetscCall(cupmBlasInterface_t::SetHandleStream(handle, dci->stream));
    dci->solver = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode check_current_device_(PetscDeviceContext dctxl, PetscDeviceContext dctxr)) {
    const auto devidl = dctxl->device->deviceId, devidr = dctxr->device->deviceId;

    PetscFunctionBegin;
    PetscCheck(devidl == devidr, PETSC_COMM_SELF, PETSC_ERR_GPU, "Device contexts must be on the same device; dctx A (id %" PetscInt_FMT " device id %" PetscInt_FMT ") dctx B (id %" PetscInt_FMT " device id %" PetscInt_FMT ")", dctxl->id, devidl,
               dctxr->id, devidr);
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
      if (handle) PetscCallCUPMBLAS(cupmBlasDestroy(handle));
      handle = nullptr;
    }
    for (auto &&handle : solverhandles_) {
      if (handle) PetscCall(cupmBlasInterface_t::DestroyHandle(handle));
      handle = nullptr;
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  struct CUPMAllocatorBase : Petsc::Device::Impl::SegmentedMemoryPoolAllocatorBase<PetscType, std::atomic_flag> {
    using base_type = Petsc::Device::Impl::SegmentedMemoryPoolAllocatorBase<PetscType, std::atomic_flag>;

    template <typename U>
    PETSC_CXX_COMPAT_DECL(PetscErrorCode call(U &&functor, cupmStream_t strm)) {
      PetscFunctionBegin;
      if (cupmStreamQuery(strm) == cupmSuccess) {
        // if the stream isn't busy then no need to do the allocations below, just call the
        // functor immediately
        PetscCall(base_type::call(std::forward<U>(functor), strm));
      } else {
        using function_type = std::function<void(void)>;
        const auto last     = cupmGetLastError();

        if (PetscUnlikely(last != cupmErrorNotReady)) PetscCallCUPM(last);
        PetscCallCUPM(cupmLaunchHostFunc(
          strm,
          [](void *ptr) {
            auto fn = static_cast<function_type *>(ptr);
            (*fn)();
            delete fn;
          },
          new function_type{std::forward<U>(functor)}));
      }
      PetscFunctionReturn(0);
    }
  };

  template <typename PetscType>
  struct HostAllocator : CUPMAllocatorBase<PetscType> {
    using base_type = CUPMAllocatorBase<PetscType>;
    using typename base_type::base_type::real_value_type;
    using typename base_type::base_type::value_type;

    PETSC_CXX_COMPAT_DECL(PetscErrorCode allocate(value_type **ptr, std::size_t n)) {
      PetscFunctionBegin;
      PetscCall(PetscCUPMMallocHost(ptr, n));
      PetscFunctionReturn(0);
    }

    PETSC_CXX_COMPAT_DECL(PetscErrorCode deallocate(value_type *ptr, cupmStream_t)) {
      PetscFunctionBegin;
      PetscCallCUPM(cupmFreeHost(ptr));
      PetscFunctionReturn(0);
    }
  };

  template <typename PetscType>
  struct DeviceAllocator : CUPMAllocatorBase<PetscType> {
    using base_type = CUPMAllocatorBase<PetscType>;
    using typename base_type::base_type::real_value_type;
    using typename base_type::base_type::value_type;

    PETSC_CXX_COMPAT_DECL(PetscErrorCode allocate(value_type **ptr, std::size_t n)) {
      PetscFunctionBegin;
      PetscCall(PetscCUPMMalloc(ptr, n));
      PetscFunctionReturn(0);
    }

    PETSC_CXX_COMPAT_DECL(PetscErrorCode deallocate(value_type *ptr, cupmStream_t strm)) {
      PetscFunctionBegin;
      PetscCallCUPM(cupmFreeAsync(ptr, strm));
      PetscFunctionReturn(0);
    }

    PETSC_CXX_COMPAT_DECL(PetscErrorCode zero(value_type *ptr, std::size_t n, cupmStream_t strm)) {
      PetscFunctionBegin;
      PetscCall(PetscCUPMMemsetAsync(ptr, 0, n, strm));
      PetscFunctionReturn(0);
    }

    PETSC_CXX_COMPAT_DECL(PetscErrorCode setCanary(value_type *ptr, std::size_t n, cupmStream_t strm)) {
      using limit_t           = std::numeric_limits<real_value_type>;
      const value_type canary = limit_t::has_signaling_NaN ? limit_t::signaling_NaN() : limit_t::max();

      PetscFunctionBegin;
      PetscCall(ThrustSet<T>(strm, n, ptr, &canary));
      PetscFunctionReturn(0);
    }
  };

  template <typename Allocator, typename PetscType = typename Allocator::value_type>
  PETSC_CXX_COMPAT_DECL(Petsc::Device::Impl::SegmentedMemoryPool<PetscType, Allocator> &managed_pool_()) {
    static Petsc::Device::Impl::SegmentedMemoryPool<PetscType, Allocator> pool;
    return pool;
  }

  template <template <typename> class Allocator, typename PetscType, typename PetscManagedType>
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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memAlloc(PetscDeviceContext, PetscBool, PetscMemType, std::size_t, void **PETSC_RESTRICT));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memFree(PetscDeviceContext, PetscMemType, void *PETSC_RESTRICT));
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
    initialized_ = true;
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
  PetscFunctionBegin;
  if (const auto dci = impls_cast_(dctx)) {
    if (dci->stream) PetscCallCUPM(cupmStreamDestroy(dci->stream));
    if (dci->event) PetscCallCUPM(cupmEventDestroy(dci->event));
    if (dci->begin) PetscCallCUPM(cupmEventDestroy(dci->begin));
    if (dci->end) PetscCallCUPM(cupmEventDestroy(dci->end));
    PetscCall(PetscFree(dctx->data));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype)) {
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (auto &stream = dci->stream) {
    PetscCallCUPM(cupmStreamDestroy(stream));
    stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx)) {
  const auto dci    = impls_cast_(dctx);
  auto      &stream = dci->stream;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  if (stream) {
    PetscCallCUPM(cupmStreamDestroy(stream));
    stream = nullptr;
  }
  switch (const auto stype = dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING: PetscCallCUPM(cupmStreamCreate(&stream)); break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING: PetscCallCUPM(cupmStreamCreateWithFlags(&stream, cupmStreamNonBlocking)); break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid PetscStreamType %s", PetscStreamTypes[util::integral_value(stype)]); break;
  }
  if (!dci->event) PetscCallCUPM(cupmEventCreateWithFlags(&dci->event, cupmEventDisableTiming));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle)) {
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  {
    const auto cerr = cupmStreamQuery(impls_cast_(dctx)->stream);
    if (cerr == cupmSuccess) *idle = PETSC_TRUE;
    else {
      // somethings gone wrong
      if (PetscUnlikely(cerr != cupmErrorNotReady)) PetscCallCUPM(cerr);
      *idle = PETSC_FALSE;
    }
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)) {
  const auto dcib = impls_cast_(dctxb);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctxa, dctxb));
  PetscCallCUPM(cupmEventRecord(dcib->event, dcib->stream));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream, dcib->event, 0));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx)) {
  const auto dci    = impls_cast_(dctx);
  const auto stream = dci->stream;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  // in case anything was queued on the event
  PetscCallCUPM(cupmStreamWaitEvent(stream, dci->event, 0));
  PetscCallCUPM(cupmStreamSynchronize(stream));
  // try pruning our managed pools now that we have synchronized
  PetscCall(managed_pool_<HostAllocator<PetscScalar>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscScalar>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<HostAllocator<PetscReal>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscReal>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<HostAllocator<PetscInt>>().pruneEmptyBlocks(stream));
  PetscCall(managed_pool_<DeviceAllocator<PetscInt>>().pruneEmptyBlocks(stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle)) {
  PetscFunctionBegin;
  PetscCall(initialize_handle_<handle_t>(dctx));
  *static_cast<typename handle_t::type *>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx)) {
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin, dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed)) {
  float      gtime;
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscCallCUPM(cupmEventRecord(dci->end, dci->stream));
  PetscCallCUPM(cupmEventSynchronize(dci->end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime, dci->begin, dci->end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memAlloc(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, std::size_t n, void **PETSC_RESTRICT dest)) {
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "allocating"));
  if (PetscMemTypeHost(mtype)) {
    PetscCallCUPM(cupmMallocHost(dest, n));
    if (clear) std::memset(*dest, 0, n);
  } else {
    auto stream = impls_cast_(dctx)->stream;

    PetscCallCUPM(cupmMallocAsync(dest, n, stream));
    if (clear) {
      PetscCallCUPM(cupmMemsetAsync(*dest, 0, n, stream));
      // cudaMemsetAsync() is actually still fully async w.r.t. the host even on the null
      // stream so we have to do a sync here
      if (dctx->streamType == PETSC_STREAM_GLOBAL_BLOCKING) PetscCall(synchronize(dctx));
    }
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memFree(PetscDeviceContext dctx, PetscMemType mtype, void *PETSC_RESTRICT ptr)) {
  auto vptr = (void *)ptr;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "freeing"));
  if (PetscMemTypeHost(mtype)) {
    PetscCallCUPM(cupmFreeHost(vptr));
  } else {
    PetscCallCUPM(cupmFreeAsync(vptr, impls_cast_(dctx)->stream));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode)) {
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  // can't use PetscCUPMMemcpyAsync here since we don't know sizeof(*src)...
  PetscCall(cupmMemcpyAsync(dest, src, n, PetscDeviceCopyModeToCUPMMemcpyKind(mode), impls_cast_(dctx)->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::memSet(PetscDeviceContext dctx, PetscMemType mtype, void *ptr, PetscInt v, std::size_t n)) {
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "zeroing"));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(synchronize(dctx));
    std::memset(ptr, static_cast<int>(v), n);
  } else {
    PetscCallCUPM(cupmMemsetAsync(ptr, static_cast<int>(v), n, impls_cast_(dctx)->stream));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroyManagedType(PetscDeviceContext dctx, PetscManagedType scal)) {
  const auto stream     = impls_cast_(dctx)->stream;
  auto      &host_ptr   = scal->host;
  auto      &device_ptr = scal->device;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  // try returning them to the pool
  PetscCall(managed_pool_<HostAllocator<PetscType>>().release(&host_ptr, stream));
  // not freed, indicating the pool doesn't own it, now check if it is our responsibility to
  // get rid of it
  if (host_ptr && (scal->h_cmode == PETSC_OWN_POINTER)) {
    PetscMemType mtype;

    // if the pointer is managed memory we need to call cupmFree() on it
    PetscCall(cupmGetMemType(host_ptr, &mtype));
    if (PetscMemTypeDevice(mtype)) {
      PetscCallCUPM(cupmFreeAsync(host_ptr, stream));
    } else {
      PetscCall(PetscFree(host_ptr));
    }
  }
  PetscCall(managed_pool_<DeviceAllocator<PetscType>>().release(&device_ptr, stream));
  // same deal with device pointer
  if (device_ptr && (scal->d_cmode == PETSC_OWN_POINTER)) { PetscCallCUPM(cupmFreeAsync(device_ptr, stream)); }
  PetscFunctionReturn(0);
}

// this should by all means be a lambda in getManagedTypeValues(), but since you can't make
// templated lambdas until either C++14 or for real in C++20 it is a function instead...
template <DeviceType T>
template <template <typename> class Allocator, typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::copy_managed_values_(PetscDeviceContext dctx, PetscManagedType scal, PetscMemoryAccessMode mode, PetscType *&dest, const PetscType *src, PetscOffloadMask requested_mask, cupmMemcpyKind_t direction, PetscType **ptr)) {
  const auto stream = impls_cast_(dctx)->stream;
  const auto n      = scal->n;
  auto      &mask   = scal->mask;

  PetscFunctionBegin;
  if (!dest) PetscCall(managed_pool_<Allocator<PetscType>>().get(n, &dest, stream));
  *ptr = dest;
  // no need to do anything if we already match the desired offload
  if (mask == requested_mask) PetscFunctionReturn(0);
  mask = requested_mask;
  // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
  // it now
  if (PetscMemoryAccessRead(mode) && src) {
    PetscCall(PetscCUPMMemcpyAsync(dest, src, n, direction, stream));
    // if read-only then update the offloadmask
    if (mode == PETSC_MEMORY_ACCESS_READ) mask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr)) {
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

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::applyOperatorType(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, PetscMemType mtype, const PetscType *rhs, PetscManagedType ret)) {
  const auto src_access = ret ? PETSC_MEMORY_ACCESS_READ : PETSC_MEMORY_ACCESS_READ_WRITE;
  const auto n          = scal->n;
  auto       stream     = impls_cast_(dctx)->stream;
  PetscType *ptr = nullptr, *retptr = nullptr;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(getManagedTypeValues(dctx, scal, PETSC_MEMTYPE_DEVICE, src_access, &ptr));
  if (ret) {
    PetscCall(getManagedTypeValues(dctx, ret, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE, &retptr));
  } else {
    // in place
    retptr = ptr;
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

} // namespace Impl

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = Impl::DeviceContext<DeviceType::CUDA>;
using CUPMContextHip  = Impl::DeviceContext<DeviceType::HIP>;

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS) ::Petsc::Device::CUPM::Impl::DeviceContext<::Petsc::Device::CUPM::DeviceType::IMPLS>::PetscDeviceContext_IMPLS

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif // PETSCDEVICECONTEXTCUDA_HPP
