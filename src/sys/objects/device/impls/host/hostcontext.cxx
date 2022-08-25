#ifndef HOSTCONTEXT_HPP
#define HOSTCONTEXT_HPP

#include <petscmanagedtype.hpp>
#include "../segmentedmempool.hpp"

namespace Petsc {

namespace device {

namespace host {

namespace impl {

class DeviceContext {
  template <typename PetscType, std::size_t ChunkSize = 256, typename PoolType = ::Petsc::memory::SegmentedMemoryPool<PetscType, device::DefaultStream, Petsc::memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>, ChunkSize>>
  PETSC_CXX_COMPAT_DECL(PoolType &managed_pool_()) {
    static PoolType pool;
    return pool;
  }

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext)) { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType)) { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext)) { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext, PetscBool *idle)) {
    PetscFunctionBegin;
    *idle = PETSC_TRUE; // the host is always idle
    PetscFunctionReturn(0);
  }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext)) { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext)) { return 0; }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getBlasHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getSolverHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getStreamHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }

  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext, PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext, PetscManagedType, PetscMemType, PetscMemoryAccessMode, PetscType **));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode applyOperatorType(PetscDeviceContext, PetscManagedType, PetscOperatorType, PetscMemType, const PetscType *, PetscManagedType));

  struct HostAllocator : public Petsc::memory::stream_allocator {
    PETSC_NODISCARD PetscErrorCode do_allocate(PetscDeviceContext, std::size_t bytes, void **ptr) noexcept final {
      static constexpr DefaultStream stream;

      PetscFunctionBegin;
      PetscCall(managed_pool_<char, 256 * sizeof(PetscScalar)>().get(bytes, reinterpret_cast<char **>(ptr), &stream));
      PetscFunctionReturn(0);
    }

    PETSC_NODISCARD PetscErrorCode do_deallocate(PetscDeviceContext, void *ptr) noexcept final {
      static constexpr DefaultStream stream;
      auto                           cptr = static_cast<char *>(ptr);

      PetscFunctionBegin;
      PetscCall(managed_pool_<char, 256 * sizeof(PetscScalar)>().release(&cptr, &stream));
      PetscFunctionReturn(0);
    }
  };

  PETSC_CXX_COMPAT_DECL(PetscErrorCode getAllocator(PetscDeviceContext dctx, PetscMemType mtype, PetscDeviceContextStreamAllocator *alloc)) {
    PetscFunctionBegin;
    if (PetscMemTypeHost(mtype)) {
      static auto host_alloc = std::make_shared<HostAllocator>();

      *alloc = host_alloc;
    }
    PetscFunctionReturn(0);
  }

  const struct _DeviceContextOps ops = {destroy, changeStreamType, setUp, query, waitForContext, synchronize, getBlasHandle, getSolverHandle, getStreamHandle, beginTimer, endTimer, nullptr, nullptr, nullptr, nullptr, destroyManagedType<PetscScalar, PetscManagedScalar>, getManagedTypeValues<PetscScalar, PetscManagedScalar>, applyOperatorType<PetscScalar, PetscManagedScalar>, destroyManagedType<PetscReal, PetscManagedReal>, getManagedTypeValues<PetscReal, PetscManagedReal>, applyOperatorType<PetscReal, PetscManagedReal>, destroyManagedType<PetscInt, PetscManagedInt>, getManagedTypeValues<PetscInt, PetscManagedInt>, applyOperatorType<PetscInt, PetscManagedInt>, nullptr, nullptr, getAllocator};
};

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::destroyManagedType(PetscDeviceContext, PetscManagedType scal)) {
  static constexpr DefaultStream stream;

  PetscFunctionBegin;
  PetscCall(managed_pool_<PetscType>().release(&scal->host, &stream));
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr)) {
  auto &mask = scal->mask;
  auto &sptr = scal->host;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheckCompatibleDeviceTypes(dtype, 1, scal->dtype, 2);
    PetscCheck(PetscMemTypeHost(mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Host device can only access host memory from a managed scalar, not %s", PetscMemTypes(mtype));
    PetscCheck(!PetscOffloadDevice(mask), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Host device can only manage scalars on the host, not %s", PetscOffloadMasks(mask));
  }
  // the only values we can "get" is the host pointer
  if (!sptr) {
    static constexpr DefaultStream stream;

    PetscCall(managed_pool_<PetscType>().get(scal->n, &sptr, &stream));
  }
  *ptr = sptr;
  mask = PETSC_OFFLOAD_CPU; // a host managed scalar is always offloaded on host
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::applyOperatorType(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType, PetscMemType, const PetscType *, PetscManagedType)) {
  PetscDeviceType dtype;

  PetscFunctionBegin;
  // we should never get here
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  PetscCheckCompatibleDeviceTypes(dtype, 1, scal->dtype, 2);
  PetscAssert(PetscOffloadHost(scal->mask), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Host device can only manage scalars on the host");
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host apply operator can only apply a rhs from host memory");
}

} // namespace impl

} // namespace host

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext dctx) {
  static constexpr auto hostctx = ::Petsc::device::host::impl::DeviceContext{};

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(dctx->ops, &hostctx.ops, 1));
  PetscFunctionReturn(0);
}

#endif // HOSTCONTEXT_HPP
