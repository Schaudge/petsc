#ifndef HOSTCONTEXT_HPP
#define HOSTCONTEXT_HPP

#include "../segmentedmempool.hpp"
#include <cstring> // std::memset()

namespace Petsc {

namespace Device {

namespace Host {

namespace Impl {

class DeviceContext {
  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(::Petsc::Device::Impl::SegmentedMemoryPool<PetscType> &managed_pool_()) {
    static ::Petsc::Device::Impl::SegmentedMemoryPool<PetscType> pool;
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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getBlasHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getSolverHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getStreamHandle(PetscDeviceContext, void *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *)) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented"); }
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memalloc(PetscDeviceContext, PetscBool, PetscMemType, std::size_t, void **PETSC_RESTRICT));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memfree(PetscDeviceContext, PetscMemType, void *PETSC_RESTRICT));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memcopy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, std::size_t, PetscDeviceCopyMode));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode memset(PetscDeviceContext, PetscMemType, void *, PetscInt, std::size_t));

  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext, PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext, PetscManagedType, PetscMemType, PetscMemoryAccessMode, PetscType **));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode applyOperatorType(PetscDeviceContext, PetscManagedType, PetscOperatorType, PetscMemType, const PetscType *, PetscManagedType));

  const struct _DeviceContextOps ops = {
    destroy,
    changeStreamType,
    setUp,
    query,
    waitForContext,
    synchronize,
    getBlasHandle,
    getSolverHandle,
    getStreamHandle,
    beginTimer,
    endTimer,
    memalloc,
    memfree,
    memcopy,
    memset,
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

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::synchronize(PetscDeviceContext)) {
  PetscFunctionBegin;
  PetscCall(managed_pool_<PetscScalar>().pruneEmptyBlocks());
  PetscCall(managed_pool_<PetscReal>().pruneEmptyBlocks());
  PetscCall(managed_pool_<PetscInt>().pruneEmptyBlocks());
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::memalloc(PetscDeviceContext, PetscBool clear, PetscMemType mtype, std::size_t n, void **PETSC_RESTRICT ptr)) {
  PetscFunctionBegin;
  PetscCheck(mtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_SUP, "Host context can only handle allocating host memory");
  PetscCall(PetscMallocA(1, clear, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, ptr));
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::memfree(PetscDeviceContext, PetscMemType mtype, void *PETSC_RESTRICT ptr)) {
  PetscFunctionBegin;
  PetscCheck(mtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_SUP, "Host context can only handle freeing host memory");
  PetscCall(PetscFree(ptr));
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::memcopy(PetscDeviceContext, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode)) {
  PetscFunctionBegin;
  PetscCheck(mode == PETSC_DEVICE_COPY_HTOH, PETSC_COMM_SELF, PETSC_ERR_SUP, "Host device context can only copy host-to-host");
  PetscCall(PetscMemcpy(dest, src, n));
  PetscFunctionReturn(0);
}

PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::memset(PetscDeviceContext, PetscMemType mtype, void *ptr, PetscInt v, std::size_t n)) {
  PetscFunctionBegin;
  PetscCheck(mtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_SUP, "Host context can only handle setting host memory");
  std::memset(ptr, static_cast<int>(v), n);
  PetscFunctionReturn(0);
}

template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext::destroyManagedType(PetscDeviceContext, PetscManagedType scal)) {
  PetscFunctionBegin;
  PetscCall(managed_pool_<PetscType>().release(&scal->host));
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
  if (!sptr) PetscCall(managed_pool_<PetscType>().get(scal->n, &sptr));
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

} // namespace Impl

} // namespace Host

} // namespace Device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext dctx) {
  static constexpr auto hostctx = ::Petsc::Device::Host::Impl::DeviceContext{};

  PetscFunctionBegin;
  PetscCall(PetscArraycpy(dctx->ops, &hostctx.ops, 1));
  PetscFunctionReturn(0);
}

#endif // HOSTCONTEXT_HPP
