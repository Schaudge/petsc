#ifndef CUPMALLOCATOR_HPP
#define CUPMALLOCATOR_HPP

#if defined(__cplusplus)
#include <petsc/private/cpp/object_pool.hpp>

#include "../segmentedmempool.hpp"
#include "cupmthrustutility.hpp"

#include <limits> // std::numeric_limits

namespace Petsc {

namespace device {

namespace cupm {

// Allocator class to allocate pinned host memory for use with device
template <DeviceType T, typename PetscType>
struct HostAllocator : impl::Interface<T>, memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);
  using base_type = ::Petsc::memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using typename base_type::real_value_type;
  using typename base_type::value_type;

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

// Allocator class to allocate device memory
template <DeviceType T, typename PetscType>
struct DeviceAllocator : impl::Interface<T>, memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);
  using base_type = ::Petsc::memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using typename base_type::real_value_type;
  using typename base_type::value_type;

  PETSC_CXX_COMPAT_DECL(PetscErrorCode allocate(value_type **ptr, std::size_t n)) {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMallocAsync(ptr, n));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode deallocate(value_type *ptr, cupmStream_t strm)) {
    PetscFunctionBegin;
    PetscCallCUPM(cupmFreeAsync(ptr, strm));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode zero(value_type *ptr, std::size_t n, cupmStream_t strm)) {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemsetAsync(ptr, 0, n, strm, true));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode setCanary(value_type *ptr, std::size_t n, cupmStream_t strm)) {
    using limit_t           = std::numeric_limits<real_value_type>;
    const value_type canary = limit_t::has_signaling_NaN ? limit_t::signaling_NaN() : limit_t::max();

    PetscFunctionBegin;
    PetscCall(impl::ThrustSet<T>(strm, n, ptr, &canary));
    PetscFunctionReturn(0);
  }
};

namespace detail {

template <DeviceType T, unsigned long flags>
struct CUPMEventPoolAllocator : impl::Interface<T>, AllocatorBase<typename impl::Interface<T>::cupmEvent_t> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(cupmEvent_t *event)) {
    PetscFunctionBegin;
    PetscCallCUPM(cupmEventCreateWithFlags(event, flags));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(cupmEvent_t event)) {
    PetscFunctionBegin;
    PetscCallCUPM(cupmEventDestroy(event));
    PetscFunctionReturn(0);
  }
};

} // namespace detail

template <DeviceType T, unsigned long flags>
static inline auto &cupm_event_pool() noexcept {
  static ObjectPool<typename impl::Interface<T>::cupmEvent_t, detail::CUPMEventPoolAllocator<T, flags>> p;
  return p;
}

template <DeviceType T>
static inline auto &cupm_fast_event_pool() noexcept {
  return cupm_event_pool<T, impl::Interface<T>::cupmEventDisableTiming>();
}

// A bare wrapper around a cupmStream_t. The reason it exists is because we need to uniquely
// identify separate cupm streams. This is so that the memory pool can accelerate allocation
// calls as it can just pass back a pointer to memory that was used on the same
// stream. Otherwise it must either serialize with another stream or allocate a new chunk.
// Address of the objects does not suffice since cupmStreams are very likely internally reused.
template <DeviceType T>
class CUPMStream : public StreamBase<CUPMStream<T>>, impl::Interface<T> {
  using crtp_base_type = StreamBase<CUPMStream<T>>;
  friend crtp_base_type;
  class cupm_event;

public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  using stream_type = cupmStream_t;
  using id_type     = typename crtp_base_type::id_type;
  using event_type  = cupm_event;
  using flag_type   = unsigned int;

  PETSC_NODISCARD PetscErrorCode destroy() noexcept {
    PetscFunctionBegin;
    if (stream_) {
      PetscCallCUPM(cupmStreamDestroy(stream_));
      stream_ = cupmStream_t{};
      id_     = 0;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode create(flag_type flags) noexcept {
    PetscFunctionBegin;
    if (stream_) PetscFunctionReturn(0);
    PetscCallCUPM(cupmStreamCreateWithFlags(&stream_, flags));
    id_ = new_id_();
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode change_type(PetscStreamType newtype) noexcept {
    PetscFunctionBegin;
    if (newtype == PETSC_STREAM_GLOBAL_BLOCKING) {
      PetscCall(this->destroy());
    } else {
      const flag_type preferred = newtype == PETSC_STREAM_DEFAULT_BLOCKING ? cupmStreamDefault : cupmStreamNonBlocking;

      if (stream_) {
        flag_type flag;

        PetscCallCUPM(cupmStreamGetFlags(stream_, &flag));
        if ((flag != preferred) || (cupmStreamQuery(stream_) != cupmSuccess)) { PetscCall(this->destroy()); }
      }
      PetscCall(this->create(preferred));
    }
    PetscFunctionReturn(0);
  }

private:
  stream_type stream_{};
  id_type     id_ = 0;

  PETSC_NODISCARD static id_type new_id_() noexcept {
    static id_type id = 0;
    return id++;
  }

  class cupm_event {
  public:
    constexpr cupm_event() noexcept = default;

    explicit operator bool() const noexcept { return event_ != cupmEvent_t{}; }

    ~cupm_event() noexcept {
      PetscFunctionBegin;
      if (event_) { PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().reclaim(std::move(event_))); }
      PetscFunctionReturnVoid();
    }

    cupm_event(cupm_event &&other) noexcept {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, move_assign_(std::move(other)));
      PetscFunctionReturnVoid();
    }

    cupm_event &operator=(cupm_event &&other) noexcept {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, move_assign_(std::move(other)));
      PetscFunctionReturn(*this);
    }

    // event is not copyable
    cupm_event(const cupm_event &)            = delete;
    cupm_event &operator=(const cupm_event &) = delete;

    PETSC_NODISCARD cupmEvent_t get() const noexcept {
      PetscFunctionBegin;
      if (!event_) PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().get(event_));
      PetscFunctionReturn(event_);
    }

  private:
    mutable cupmEvent_t event_{};

    PETSC_NODISCARD PetscErrorCode move_assign_(cupm_event &&other) noexcept {
      PetscFunctionBegin;
      if (event_) {
        PetscCall(cupm_fast_event_pool<T>().reclaim(std::move(event_)));
        event_ = cupmEvent_t{};
      }
      std::swap(event_, other.event_);
      PetscFunctionReturn(0);
    }
  };

  // CRTP implementations
  PETSC_NODISCARD stream_type get_stream_() const noexcept { return stream_; }

  PETSC_NODISCARD id_type get_id_() const noexcept { return id_; }

  PETSC_NODISCARD PetscErrorCode record_event_(const event_type &event) const noexcept {
    PetscFunctionBegin;
    PetscCallCUPM(cupmEventRecord(event.get(), stream_));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode wait_for_(const event_type &event) const noexcept {
    PetscFunctionBegin;
    PetscCallCUPM(cupmStreamWaitEvent(stream_, event.get(), 0));
    PetscFunctionReturn(0);
  }
};

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // CUPMALLOCATOR_HPP
