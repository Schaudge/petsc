#ifndef PETSCMANAGEDTYPE_HPP
#define PETSCMANAGEDTYPE_HPP

// clang-format off
#include "petsc/private/cpp/utility.hpp"
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>

#if defined(__cplusplus)
#include <string>
#include <iostream>
#include <memory>
#include <utility>

static inline  PetscErrorCode PetscDeviceContextGetAllocator(PetscDeviceContext dctx, PetscMemType mtype, PetscDeviceContextStreamAllocator *alloc)
{
  PetscFunctionBegin;
  if (!dctx) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall((*dctx->ops->getallocator)(dctx, mtype, alloc));
  if (*alloc) std::cout<<"got "<< PetscMemTypes(mtype)<<" allocator"<<std::endl;
  PetscFunctionReturn(0);
}

namespace Unused {
#if 0
namespace Petsc
{

// ============================================================================================

struct simple_slice
{
  using size_type = PetscInt;

  size_type          start;
  PetscDeviceContext dctx{nullptr};
};

template <typename T>
struct OwnedPointer
{
  using value_type   = T;
  using pointer_type = T *;
  using mode_type    = PetscCopyMode;

  pointer_type ptr   = nullptr;
  mode_type    cmode = PETSC_OWN_POINTER;

  constexpr operator bool() const noexcept { return ptr != nullptr; }
};

template <typename T, template <class> class MemoryClass>
static inline MemoryClass<T> wrap_memory(T *ptr = nullptr, PetscCopyMode mode = PETSC_OWN_POINTER)
{
  return {ptr, mode};
}

template <typename T>
struct HostMemory : OwnedPointer<T>
{
};

template <typename T>
static inline HostMemory<T> wrap_host_memory(T *ptr = nullptr, PetscCopyMode mode = PETSC_OWN_POINTER)
{
  return {ptr, mode};
}

template <typename T, util::enable_if_t<!std::is_pointer<util::decay_t<T>>::value> * = nullptr>
static inline HostMemory<T> wrap_host_memory(T &val, PetscCopyMode mode = PETSC_USE_POINTER)
{
  return {&val, mode};
}

template <typename T>
struct DeviceMemory : OwnedPointer<T>
{
};

template <typename T>
static inline DeviceMemory<T> wrap_device_memory(T *ptr = nullptr, PetscCopyMode mode = PETSC_OWN_POINTER)
{
  return {ptr, mode};
}

// ============================================================================================

namespace
{

template <typename, template <typename...> typename>
struct is_instance_impl : std::false_type
{
};

template <template <typename...> typename U, typename... Ts>
struct is_instance_impl<U<Ts...>, U> : std::true_type
{
};

} // namespace

template <typename T, template <typename...> typename U>
using is_instance = is_instance_impl<util::decay_t<T>, U>;

template <typename D>
class ManagedTypeExpression
{
public:
  using size_type = PetscInt;

  PETSC_NODISCARD auto operator[](simple_slice idx) noexcept { return underlying().at_impl_(std::move(idx)); }
  PETSC_NODISCARD auto operator[](simple_slice idx) const noexcept { return underlying().at_impl_(std::move(idx)); }

  PETSC_NODISCARD auto operator[](size_type idx) noexcept { return underlying()[{idx, dctx_}]; }
  PETSC_NODISCARD auto operator[](size_type idx) const noexcept { return underlying()[{idx, dctx_}]; }

  PETSC_NODISCARD size_type size() const noexcept { return underlying().size_impl_(); }
  PETSC_NODISCARD size_type size() noexcept { return underlying().size_impl_(); }

  PETSC_NODISCARD ManagedTypeExpression &with(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    dctx_ = dctx;
    PetscFunctionReturn(*this);
  }

  PETSC_NODISCARD PetscDeviceContext dctx() const noexcept { return dctx_; }

protected:
  PetscDeviceContext dctx_{};

  constexpr ManagedTypeExpression() noexcept
  {
    if (is_instance<D, ManagedTypeExpression>::value)
      ;
  }

  const D &underlying() const noexcept { return static_cast<const D &>(*this); }
  D       &underlying() noexcept { return static_cast<D &>(*this); }
};

template <typename LExpr, typename RExpr, typename F>
class binary_op : public ManagedTypeExpression<binary_op<LExpr, RExpr, F>>
{
public:
  using base_type = ManagedTypeExpression<binary_op<LExpr, RExpr, F>>;
  friend base_type;
  using base_type::size;
  using typename base_type::size_type;
  using base_type::operator[];

  explicit constexpr binary_op(const LExpr &lxpr, const RExpr &rxpr, F &&callable = F()) noexcept
    : lhs_(lxpr), rhs_(rxpr), op_(std::forward<F>(callable))
  { }

  PETSC_NODISCARD size_type size_impl_() const noexcept { return lhs_.size(); };

  PETSC_NODISCARD auto at_impl_(simple_slice idx) const noexcept
  {
    const auto  make_idx = [&](auto &seed) { return seed.dctx() ? simple_slice{idx.start, seed.dctx()} : idx; };
    const auto &lhsv     = lhs_[make_idx(lhs_)];

    if ((void *)&lhs_ == (void *)&rhs_) return op_(lhsv, lhsv);
    return op_(lhsv, rhs_[make_idx(rhs_)]);
  }

private:
  const LExpr &lhs_{};
  const RExpr &rhs_{};
  F            op_;
};

template <typename T>
class ManagedType : public ManagedTypeExpression<ManagedType<T>>
{
public:
  using value_type           = T;
  using reference_type       = T &;
  using const_reference_type = const T &;
  using size_type            = PetscInt;

  std::string name;

  void set_name(std::string n) { name = std::move(n); }

  ManagedType() noexcept : initialized_(false) { }

  ManagedType(PetscDeviceContext dctx, value_type *host_ptr, value_type *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask) noexcept : ManagedType()
  {
    const auto comm = dctx ? PetscObjectComm(PetscObjectCast(dctx)) : PETSC_COMM_SELF;

    PetscFunctionBegin;
    PetscCallAbort(comm, this->construct(dctx, host_ptr, device_ptr, n, host_cmode, device_cmode, mask));
    PetscFunctionReturnVoid();
  }

  ManagedType(PetscDeviceContext dctx, HostMemory<T> host, DeviceMemory<T> device, PetscInt n, PetscOffloadMask mask) noexcept : ManagedType(dctx, host.ptr, device.ptr, n, host.cmode, device.cmode, mask) { }

  ManagedType(PetscDeviceContext dctx, HostMemory<T> host, PetscInt n) noexcept : ManagedType(dctx, std::move(host), DeviceMemory<T>{}, n, PETSC_OFFLOAD_CPU){};
  ManagedType(PetscDeviceContext dctx, DeviceMemory<T> device, PetscInt n) noexcept : ManagedType(dctx, HostMemory<T>{}, std::move(device), n, PETSC_OFFLOAD_GPU) { }

  ManagedType(PetscDeviceContext dctx, PetscInt n) noexcept : ManagedType(dctx, HostMemory<T>{}, DeviceMemory<T>{}, n, PETSC_OFFLOAD_UNALLOCATED) { }

  template <typename U>
  ManagedType(const ManagedTypeExpression<U> &expr) noexcept : ManagedType(nullptr, expr.size())
  {
    PetscFunctionBegin;
    for (typename ManagedTypeExpression<U>::size_type i = 0; i < expr.size(); ++i) (*this)[i] = expr[i];
    PetscFunctionReturnVoid();
  }

  ~ManagedType() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->destroy(nullptr));
    PetscFunctionReturnVoid();
  }

  class proxy_reference
  {
  public:
    constexpr proxy_reference(const ManagedType &type, simple_slice slc) noexcept : type_(type), slice_(std::move(slc)) { std::cout << "Proxy reference created" << std::endl; }

    void operator=(value_type val) noexcept
    {
      PetscFunctionBegin;
      get_(PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE) = val;
      PetscFunctionReturnVoid();
    }

    proxy_reference &operator+=(value_type val) noexcept
    {
      PetscFunctionBegin;
      get_(PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE) += val;
      PetscFunctionReturn(*this);
    }

    friend proxy_reference operator+(proxy_reference lhs, value_type val) noexcept
    {
      PetscFunctionBegin;
      lhs += val;
      PetscFunctionReturn(lhs);
    }

    // prefix increment
    proxy_reference &operator++() noexcept
    {
      PetscFunctionBegin;
      ++get_(PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE);
      PetscFunctionReturn(*this);
    }

    // postfix increment
    proxy_reference operator++(int) noexcept
    {
      auto old = *this;

      PetscFunctionBegin;
      this->operator++();
      PetscFunctionReturn(old);
    }

    // prefix decrement
    proxy_reference &operator--() noexcept
    {
      PetscFunctionBegin;
      --get_(PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE);
      PetscFunctionReturn(*this);
    }

    // postfix decrement
    proxy_reference operator--(int) noexcept
    {
      auto old = *this;

      PetscFunctionBegin;
      this->operator--();
      PetscFunctionReturn(old);
    }

    operator value_type() noexcept { return get_(PETSC_MEMORY_ACCESS_READ, PETSC_TRUE); }

  private:
    const ManagedType &type_;
    simple_slice       slice_;
    value_type        *array_{};

    PETSC_NODISCARD reference_type get_(PetscMemoryAccessMode mode, PetscBool sync) noexcept
    {
      PetscFunctionBegin;
      if (!array_) {
        const auto dctx = slice_.dctx;
        const auto comm = dctx ? PetscObjectComm(PetscObjectCast(dctx)) : PETSC_COMM_SELF;

        PetscCallAbort(comm, const_cast<ManagedType &>(type_).get_array(dctx, PETSC_MEMTYPE_HOST, mode, sync, &array_));
      }
      PetscFunctionReturn(array_[slice_.start]);
    }
  };

  PETSC_NODISCARD proxy_reference operator[](simple_slice slc) noexcept { return {*this, std::move(slc)}; }

  PETSC_NODISCARD auto operator[](size_type pos) noexcept { return (*this)[simple_slice{pos, nullptr}]; }

  PETSC_NODISCARD value_type operator[](simple_slice slc) const noexcept
  {
    value_type *array;

    PetscFunctionBegin;
    PetscCall(const_cast<ManagedType *>(this)->get_array(slc.dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &array));
    PetscFunctionReturn(array[slc.start]);
  }

  PETSC_NODISCARD auto operator[](size_type pos) const noexcept { return (*this)[simple_slice{pos, nullptr}]; }

  PETSC_NODISCARD size_type        size() const noexcept { return size_; }
  PETSC_NODISCARD PetscOffloadMask offload_mask() const noexcept { return memory_.mask; }
  PETSC_NODISCARD PetscObjectId    id() const noexcept { return id_; }

  PETSC_NODISCARD PetscErrorCode construct(PetscDeviceContext, value_type *, value_type *, PetscInt, PetscCopyMode, PetscCopyMode, PetscOffloadMask) noexcept;
  PETSC_NODISCARD PetscErrorCode get_array(PetscDeviceContext, PetscMemType, PetscMemoryAccessMode, PetscBool, value_type **) noexcept;
  PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext) noexcept;
  PETSC_NODISCARD PetscErrorCode get_pointer_and_memtype(PetscDeviceContext, PetscMemoryAccessMode, value_type **, PetscMemType *) noexcept;
  PETSC_NODISCARD PetscErrorCode assign(PetscDeviceContext, const ManagedType &) noexcept;

private:
  struct MemoryManager
  {
    using memory_type = OwnedPointer<T>;

    memory_type      host{};
    memory_type      device{};
    PetscOffloadMask mask{PETSC_OFFLOAD_UNALLOCATED};
  };

  MemoryManager   memory_{};
  PetscDeviceType dtype_{PETSC_DEVICE_HOST};
  size_type       size_{0};
  PetscObjectId   id_{init_id_()};
  ManagedType    *parent_{nullptr};
  bool            pure_{true};
  bool            initialized_{true};
#if PetscDefined(USE_DEBUG)
  int lock_cnt_{0};
#endif

  PETSC_NODISCARD static PetscObjectId init_id_() noexcept
  {
    static PetscObjectId id;

    PetscFunctionBegin;
    //PetscCallAbort(PETSC_COMM_SELF,PetscObjectNewId_Internal(&id));
    PetscFunctionReturn(id++);
  }

  PETSC_NODISCARD PetscErrorCode set_purity_(bool purity) noexcept
  {
    PetscFunctionBegin;
    pure_ = purity;
    if (!purity && parent_) PetscCall(parent_->set_purity_(purity));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD bool lock_() const noexcept
  {
#if PetscDefined(USE_DEBUG)
    return lock_cnt_;
#else
    return false;
#endif
  }

  PETSC_NODISCARD PetscErrorCode check_lock_(bool v) const noexcept
  {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      constexpr const char *strings[] = {"unlocked", "locked"};
      const auto            locked    = lock_();

      PetscCheck(locked == v, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Managed type object is %s expected it to be %s", strings[locked], strings[v]);
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode set_mask_(PetscOffloadMask mask) noexcept
  {
    PetscFunctionBegin;
    if (offload_mask() != mask) {
      memory_.mask = mask;
      // should not update the parent if our mask did not change!
      if (parent_) PetscCall(parent_->set_mask_(mask));
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscOffloadMask &offload_mask() noexcept
  {
    return memory_.mask;
  }

  PETSC_NODISCARD bool pure() const noexcept
  {
    return pure_;
  }

  PETSC_NODISCARD bool impure() const noexcept
  {
    return !pure();
  }
};

template <typename T>
inline PetscErrorCode ManagedType<T>::construct(PetscDeviceContext dctx, value_type *host_ptr, value_type *device_ptr, PetscInt n, PetscCopyMode host_cmode, PetscCopyMode device_cmode, PetscOffloadMask mask) noexcept
{
  const auto assign_values = [&](PetscOffloadMask src_mask, value_type *src_ptr, PetscCopyMode src_cmode, typename MemoryManager::memory_type &mem) {
    PetscFunctionBegin;
    if ((mem.cmode = PETSC_COPY_VALUES)) {
      if (const auto n = size()) {
        const auto          mask = offload_mask();
        PetscDeviceCopyMode mode;
        value_type         *ptr;

        PetscCall(this->get_array(dctx, PetscOffloadMaskToMemType(mask), PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &ptr));
        PetscCall(PetscOffloadMaskToDeviceCopyMode(mask, src_mask, &mode));
        PetscCall(PetscDeviceArrayCopy(dctx, ptr, src_ptr, n, mode));
      }
      mem.cmode = PETSC_OWN_POINTER;
    } else {
      mem.ptr = src_ptr;
    }
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
#if 0
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
#endif
  if (host_ptr && n) PetscValidPointer(host_ptr, 2);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot request negative amount of managed memory %" PetscInt_FMT, n);
  if (host_ptr && device_ptr) {
    PetscAssert(mask != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Set both host and device pointer but offloadmask was %s", PetscOffloadMasks(mask));
    // this is the only instance in which we believe whatever the user has fed us
  } else if (host_ptr) {
    // clearly no device_ptr, so we own it
    mask         = PETSC_OFFLOAD_CPU;
    device_cmode = PETSC_OWN_POINTER;
  } else if (device_ptr) {
    // clearly no host_ptr, so we own it
    mask       = PETSC_OFFLOAD_GPU;
    host_cmode = PETSC_OWN_POINTER;
  } else {
    // user gave us nothing, we own everything
    mask       = PETSC_OFFLOAD_UNALLOCATED;
    host_cmode = device_cmode = PETSC_OWN_POINTER;
  }

  // populate known quantities
  size_          = n;
  offload_mask() = mask;
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype_));
  // unallocated is technically a "pure" state
  PetscCall(set_purity_(!PetscOffloadDevice(mask)));

  PetscCall(assign_values(PETSC_OFFLOAD_CPU, host_ptr, host_cmode, memory_.host));
  PetscCall(assign_values(PETSC_OFFLOAD_GPU, device_ptr, device_cmode, memory_.device));

  initialized_ = true;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::get_array(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, value_type **ptr) noexcept
{
  const auto oid = id();

  PetscFunctionBegin;
  std::cout << name << ".get_array(" << (dctx ? PetscObjectCast(dctx)->name : "(unnamed)") << ", " << PetscMemTypes(mtype) << ", " << PetscMemoryAccessModes(mode) << ", " << PetscBools[sync] << ")\n";
#if 0
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,this,2);
#endif
  PetscValidPointer(ptr, 5);
  PetscCall(check_lock_(false));
  PetscCheck(!(PetscOffloadUnallocated(offload_mask()) && PetscMemoryAccessRead(mode)), PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_ARG_WRONG, "Trying to read (using %s) from a managed type (id %" PetscInt64_FMT ") that has not been written to (has offload mask %s)", PetscMemoryAccessModes(mode), oid, PetscOffloadMasks(offload_mask()));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, oid, mode, nullptr));
#if 0
  PetscCall((*dctx->ops->getmanagedvaluesscalar)(dctx,this,mtype,mode,ptr));
#else
  PetscCheck(PetscMemTypeHost(mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Not memtype host");
  if (!memory_.host) {
    memory_.host.ptr = new T[size_];
    memory_.host.cmode = PETSC_OWN_POINTER;
    memory_.mask = PETSC_OFFLOAD_CPU;
  }
  *ptr = memory_.host.ptr;
#endif
  // if user intends to write to device in any capacity then we are impure
  if (PetscMemTypeDevice(mtype) && PetscMemoryAccessWrite(mode)) PetscCall(set_purity_(false));
  // also sets the parents mask if needed
  PetscCall(set_mask_(offload_mask()));
  // REVIEW ME:
  // if we are pure, there is no need to synchronize (I think)
  if (sync && impure()) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    if (PetscMemTypeHost(mtype)) PetscCall(set_purity_(true));
  }
  PetscAssert(*ptr, PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_PLIB, "ManagedType (id %" PetscInt64_FMT ") Returned null pointer for mtype %s", oid, PetscMemTypes(mtype));
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::destroy(PetscDeviceContext dctx) noexcept
{
  const auto check_copy_mode = [](typename MemoryManager::memory_type &mem, const char name[]) {
    const auto cmode = mem.cmode;

    PetscFunctionBegin;
    PetscAssert(cmode != PETSC_COPY_VALUES, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Managed types should not have a %s copy mode of %s", PetscCopyModes[cmode], name);
    if (cmode == PETSC_USE_POINTER) mem.ptr = nullptr;
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
#if 0
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,*scal,2);
#endif
  PetscCall(check_lock_(false));
  // if we don't own it, nullify it since the memory pools may try to return it below
  PetscCall(check_copy_mode(memory_.host, "host"));
  PetscCall(check_copy_mode(memory_.device, "device"));
#if 0
  PetscManagedTypeCallMethod(dctx->ops->destroymanagedtype,dctx,*scal);
#endif
  // if the host pointer still exists at this point it is because it didn't belong to its
  // respective memory pool. If copy mode is PETSC_OWN_POINTER its because we have
  // co-opted the users pointer, so we should free it now.
  if (memory_.host && (memory_.host.cmode == PETSC_OWN_POINTER)) delete[] memory_.host.ptr;
  // cannot handle device pointers though
  if (PetscDefined(USE_DEBUG) && memory_.device) {
    PetscDeviceType dtype;
    PetscObjectId   id;

    PetscCall(PetscObjectGetId(PetscObjectCast(dctx), &id));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheck(memory_.device.cmode != PETSC_OWN_POINTER, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext (id %" PetscInt64_FMT ", device type %s) failed to free the owned device pointer", id, PetscDeviceTypes[dtype]);
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::get_pointer_and_memtype(PetscDeviceContext dctx, PetscMemoryAccessMode mode, value_type **ptr, PetscMemType *mtype) noexcept
{
  PetscMemType retmtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  check_lock_(false);
  PetscValidPointer(ptr, 3);
  if (mtype) PetscValidPointer(mtype, 4);
  switch (const auto mask = offload_mask()) {
    // if both prefer CPU, since we may be able to set purity
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_CPU: retmtype = PETSC_MEMTYPE_HOST; break;
  case PETSC_OFFLOAD_GPU: retmtype = PETSC_MEMTYPE_DEVICE; break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    retmtype = dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  } break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support yet for offloadmask %d", mask);
  }
  PetscCall(this->get_array(dctx, retmtype, mode, PETSC_FALSE, ptr));
  PetscAssert(*ptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values", PetscMemTypes(retmtype));
  if (mtype) *mtype = retmtype;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::assign(PetscDeviceContext dctx, const ManagedType<T> &src) noexcept
{
  const auto dn = size(), sn = src.size();

  PetscFunctionBegin;
  if (this == &src) PetscFunctionReturn(0);
#if 0
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,dest,2);
  PetscCheckManagedTypeCompatibleDeviceContext(dctx,1,src,3);
#endif
  PetscAssert(dn >= sn, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Destination size %" PetscInt_FMT " not large enough for source size %" PetscInt_FMT, dn, sn);
  if (sn) {
    auto        dest_mtype = PETSC_MEMTYPE_DEVICE, src_mtype = PETSC_MEMTYPE_DEVICE;
    value_type *dest_ptr, *src_ptr;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscCall(this->get_array_and_memtype(dctx, PETSC_MEMORY_ACCESS_WRITE, &dest_ptr, &dest_mtype));
    PetscCall(src.get_array_and_memtype(dctx, PETSC_MEMORY_ACCESS_READ, &src_ptr, &src_mtype));
    PetscCall(PetscDeviceArrayCopy(dctx, dest_ptr, src_ptr, sn, PetscMemTypeToDeviceCopyMode(dest_mtype, src_mtype)));
  }
  PetscFunctionReturn(0);
}

template <typename T, typename U>
auto operator+(const ManagedTypeExpression<T> &l, const ManagedTypeExpression<U> &r)
{
  return binary_op<T, U, std::plus<>>(*static_cast<const T *>(&l), *static_cast<const U *>(&r));
}

template <typename T, typename U>
auto operator*(const ManagedTypeExpression<T> &l, const ManagedTypeExpression<U> &r)
{
  return binary_op<T, U, std::multiplies<>>(*static_cast<const T *>(&l), *static_cast<const U *>(&r));
}

using ManagedScalar = ManagedType<PetscScalar>;

} // namespace Petsc
#endif
} // namespace Unused

namespace Petsc {

namespace memory {

struct copy_init_t      { };
struct move_init_t      { };
struct reference_init_t { };

class stream_allocator
{
public:
  using size_type = std::size_t;

  virtual ~stream_allocator() noexcept = default;

  template <typename T>
  PETSC_NODISCARD PetscErrorCode allocate(PetscDeviceContext,size_type,T**) noexcept;

  template <typename T>
  PETSC_NODISCARD PetscErrorCode deallocate(PetscDeviceContext,T**) noexcept;

  template <typename T>
  PETSC_NODISCARD PetscErrorCode reallocate(PetscDeviceContext,size_type,T**) noexcept;

  PETSC_NODISCARD virtual PetscMemType mem_type() const noexcept;

protected:
  constexpr stream_allocator() noexcept = default;

private:
  PETSC_NODISCARD virtual PetscErrorCode do_allocate(PetscDeviceContext,size_type,void**)   noexcept = 0;
  PETSC_NODISCARD virtual PetscErrorCode do_deallocate(PetscDeviceContext,void*)            noexcept = 0;
  PETSC_NODISCARD virtual PetscErrorCode do_reallocate(PetscDeviceContext,size_type,void**) noexcept = 0;
};

template <typename T>
inline PetscErrorCode stream_allocator::allocate(PetscDeviceContext dctx, size_type nelem, T **ptr) noexcept
{
  static_assert(!std::is_void<T>::value, "");

  PetscFunctionBegin;
  PetscValidPointer(ptr, 3);
  *ptr = nullptr;
  if (const auto bytes = nelem * sizeof(T)) {
    PetscCall(do_allocate(dctx, bytes, reinterpret_cast<void **>(ptr)));
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode stream_allocator::deallocate(PetscDeviceContext dctx, T **ptr) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(ptr, 2);
  if (auto &ret = *ptr) {
    PetscCall(do_deallocate(dctx, ret));
    ret = nullptr;
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode stream_allocator::reallocate(PetscDeviceContext dctx, size_type newnelem, T **ptr) noexcept
{
  static_assert(!std::is_void<T>::value,"");

  PetscFunctionBegin;
  PetscValidPointer(ptr,3);
  // there are several scenarios here that we can head off before virtual dispatch
  if (*ptr) {
    if (newnelem) {
      // pointer and nonzero nelem, actually do a realloc()
      PetscCall(do_reallocate(dctx, newnelem * sizeof(T), reinterpret_cast<void **>(ptr)));
    } else {
      // realloc() to zero, in other words free()
      PetscCall(deallocate(dctx, ptr));
    }
  } else if (newnelem) {
    // no pointer and nonzero nelem, in other words malloc()
    PetscCall(allocate(dctx, newnelem, ptr));
  } else {
    // no pointer and zero nelem, a no-op
    (void)ptr;
  }
  PetscFunctionReturn(0);
}

inline PetscMemType stream_allocator::mem_type() const noexcept
{
  return PETSC_MEMTYPE_HOST;
}

template <typename T>
class managed_storage
{
public:
  using value_type        = T;
  using allocator_type    = stream_allocator;
  using size_type         = typename allocator_type::size_type;
  using allocator_pointer = std::shared_ptr<stream_allocator>;

  constexpr managed_storage() noexcept = default;

  template <typename Iterator>
  managed_storage(copy_init_t init, PetscDeviceContext dctx, PetscMemType mtype, Iterator begin, Iterator end) noexcept
    : mtype_(mtype)
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,iterator_init_(init,dctx,begin,std::distance(begin,end)));
    PetscFunctionReturnVoid();
  }

  template <typename Iterator>
  managed_storage(reference_init_t init, PetscDeviceContext dctx, PetscMemType mtype, Iterator begin, Iterator end) noexcept
    : ptr_(begin), mtype_(mtype)
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,iterator_init_(init,dctx,begin,std::distance(begin,end)));
    PetscFunctionReturnVoid();
  }

  template <typename Iterator>
  managed_storage(move_init_t init, PetscDeviceContext dctx, PetscMemType mtype, Iterator begin, Iterator end) noexcept
    : ptr_(begin), mtype_(mtype)
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,iterator_init_(init,dctx,begin,std::distance(begin,end)));
    PetscFunctionReturnVoid();
  }

  ~managed_storage()
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->clear());
    PetscFunctionReturnVoid();
  }

  managed_storage(managed_storage&& other) noexcept
    : ptr_(std::exchange(other.ptr_,nullptr)), mtype_(other.mtype_),
      allocator_(std::move(other.allocator_))
  { }

  managed_storage& operator=(managed_storage&& other) noexcept
  {
    PetscFunctionBegin;
    if (&other != this) {
      // delete our pointer (if we have one)
      PetscCall(this->clear());
      mtype_     = other.mtype_;
      ptr_       = std::exchange(other.ptr_, nullptr);
      allocator_ = std::move(other.allocator_);
    }
    PetscFunctionReturn(*this);
  }

  PETSC_NODISCARD value_type        *data()      const noexcept { return ptr_;       }
  PETSC_NODISCARD PetscMemType       mem_type()  const noexcept { return mtype_;     }
  PETSC_NODISCARD allocator_pointer  allocator() const noexcept { return allocator_; }
  PETSC_NODISCARD allocator_pointer &allocator()       noexcept { return allocator_; }

  explicit operator bool() const noexcept { return data() != nullptr; }

  PETSC_NODISCARD PetscErrorCode get_pointer(PetscDeviceContext, PetscInt, value_type **) noexcept;
  PETSC_NODISCARD PetscErrorCode reserve(PetscDeviceContext, PetscInt) noexcept;
  PETSC_NODISCARD PetscErrorCode clear(PetscDeviceContext = nullptr) noexcept;

private:
  value_type        *ptr_   = nullptr;
  PetscMemType       mtype_ = PETSC_MEMTYPE_HOST;
  allocator_pointer  allocator_{};

  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode iterator_init_(copy_init_t,PetscDeviceContext,Iterator,size_type) noexcept;
  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode iterator_init_(reference_init_t,PetscDeviceContext,Iterator,size_type) noexcept;
  template <typename Iterator>
  PETSC_NODISCARD PetscErrorCode iterator_init_(move_init_t,PetscDeviceContext,Iterator,size_type) noexcept;
};

template <typename T>
template <typename Iterator>
inline PetscErrorCode managed_storage<T>::iterator_init_(copy_init_t, PetscDeviceContext dctx, Iterator begin, size_type n) noexcept
{
  PetscFunctionBegin;
  PetscCall(get_pointer(dctx, n, nullptr));
  PetscCall(PetscDeviceArrayCopy(dctx,data(),static_cast<value_type*>(begin),n,PetscMemTypeToDeviceCopyMode(mem_type(),mem_type())));
  PetscFunctionReturn(0);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode managed_storage<T>::iterator_init_(reference_init_t, PetscDeviceContext, Iterator, size_type) noexcept
{
  return 0;
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode managed_storage<T>::iterator_init_(move_init_t, PetscDeviceContext dctx, Iterator, size_type n) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetAllocator(dctx,mem_type(),&allocator_));
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::get_pointer(PetscDeviceContext dctx, PetscInt n, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Memory amount %" PetscInt_FMT " < 0", n);
  PetscValidPointer(ptr,3);
  if (!data()) {
    if (!allocator()) PetscCall(PetscDeviceContextGetAllocator(dctx, mem_type(), &allocator_));
    PetscCall(allocator()->allocate(dctx, n, &ptr_));
  }
  *ptr = data();
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::reserve(PetscDeviceContext dctx, PetscInt n) noexcept
{
  PetscFunctionBegin;
  if (!allocator()) PetscCall(PetscDeviceContextGetAllocator(dctx, mem_type(), &allocator_));
  if (data()) {
    PetscCall(allocator()->reallocate(dctx, n, &ptr_));
  } else {
    PetscCall(allocator()->allocate(dctx, n, &ptr_));
  }
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode managed_storage<T>::clear(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  // if ptr but no allocator then we never owned the pointer
  if (data()) {
    if (allocator()) PetscCall(allocator()->deallocate(dctx, &ptr_));
    ptr_ = nullptr;
  }
  PetscFunctionReturn(0);
}

} // namespace memory

namespace expr
{

template <typename D>
class ExpressionBase
{
public:
  using size_type = PetscInt;

  PETSC_NODISCARD size_type size() const noexcept { return underlying().size_impl_(); }

  PETSC_NODISCARD auto at(size_type idx) const noexcept { return underlying().at_impl_(idx); }

  PETSC_NODISCARD PetscErrorCode prefetch(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().prefetch_impl_(dctx););
    PetscFunctionReturn(0);
  }

protected:
  PETSC_NODISCARD const D &underlying() const noexcept { return static_cast<const D &>(*this); }
  PETSC_NODISCARD       D &underlying()       noexcept { return static_cast<      D &>(*this); }
};

template <typename L, typename R, typename F>
class BinaryManagedExpression : public ExpressionBase<BinaryManagedExpression<L, R, F>>
{
public:
  using base_type = ExpressionBase<BinaryManagedExpression<L, R, F>>;
  friend base_type;
  using base_type::size;
  using typename base_type::size_type;

  constexpr explicit BinaryManagedExpression(L&& lxpr, R&& rxpr, F &&callable = F()) noexcept
    : lhs_(std::forward<L>(lxpr)), rhs_(std::forward<R>(rxpr)), op_(std::forward<F>(callable))
  {
    PetscAssertAbort(lhs_.size() == rhs_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Left and right operand size mismatch, %zu != %zu", static_cast<std::size_t>(lhs_.size()), static_cast<std::size_t>(rhs_.size()));
  }

  PETSC_NODISCARD size_type size_impl_() const noexcept
  {
    return lhs_.size();
  };

  PETSC_NODISCARD auto at_impl_(size_type idx) const noexcept
  {
    const auto &lhsv = lhs_.at(idx);

    if ((void *)&lhs_ == (void *)&rhs_) return op_(lhsv, lhsv);
    return op_(lhsv, rhs_.at(idx));
  }

  PETSC_NODISCARD PetscErrorCode prefetch_impl_(PetscDeviceContext dctx) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(lhs_.prefetch(dctx));
    PetscCall(rhs_.prefetch(dctx));
    PetscFunctionReturn(0);
  }

private:
  L lhs_;
  R rhs_;
  F op_;
};

template <typename T>
struct EvaluatedManagedExpression : ExpressionBase<T>
{
  template <typename U>
  constexpr EvaluatedManagedExpression(U&& expr, PetscDeviceContext ctx) noexcept
    : ExpressionBase<T>(std::forward<U>(expr)), dctx(ctx)
  { }

  PetscDeviceContext dctx;
};

template <typename T>
PETSC_NODISCARD static inline EvaluatedManagedExpression<util::remove_reference_t<T>> eval(PetscDeviceContext dctx, T&& expr) noexcept
{
  PetscCallAbort(PETSC_COMM_SELF, expr.prefetch(dctx));
  return {std::forward<T>(expr), dctx};
}

template <typename L, typename R>
static inline auto operator*(L&& lhs, R&& rhs) noexcept
{
  return BinaryManagedExpression<L,R,std::multiplies<>>(std::forward<L>(lhs),std::forward<R>(rhs));
}

} // namespace expr

template <typename T>
class ManagedType : public expr::ExpressionBase<ManagedType<T>>
{
  friend expr::ExpressionBase<ManagedType<T>>;

public:
  using value_type           = T;
  using pointer_type         = T *;
  using const_pointer_type   = const T *;
  using reference_type       = T &;
  using const_reference_type = const T &;
  using size_type            = PetscInt;
  using storage_type         = memory::managed_storage<T>;

  explicit ManagedType(PetscDeviceContext dctx, value_type *host_ptr, value_type *device_ptr, size_type n, PetscCopyMode h_cmode, PetscCopyMode d_cmode, PetscOffloadMask mask) noexcept
    : size_(n),
      host_(construct_storage_(h_cmode,dctx,PETSC_MEMTYPE_HOST,host_ptr,host_ptr+n)),
      device_(construct_storage_(d_cmode,dctx,PETSC_MEMTYPE_DEVICE,device_ptr,device_ptr+n)),
      mask_(*this,init_mask_(host_ptr,device_ptr,mask)), pure_(*this,true)
  { }

  explicit ManagedType(PetscDeviceContext dctx, size_type n) noexcept
    : ManagedType(dctx, nullptr, nullptr, n, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED)
  { }

  explicit ManagedType() noexcept : ManagedType(nullptr, 0) { }

  ~ManagedType() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,this->clear());
    PetscFunctionReturnVoid();
  }

  // does not init id, since this is a separate object (and separate id)
  ManagedType(ManagedType&& other) noexcept
    : id_(std::exchange(other.id_, init_id_())), size_(std::exchange(other.size_, 0)),
      host_(std::move(other.host_)), device_(std::move(other.device_)),
      mask_(*this, std::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED)),
      pure_(*this, std::exchange(other.pure_, true)),
      parent_(std::exchange(other.parent_,nullptr))
  { }

  template <typename U>
  ManagedType& operator=(const expr::EvaluatedManagedExpression<U> &expr) noexcept
  {
    using size_type = typename util::remove_reference_t<decltype(expr)>::size_type;
    value_type *arr;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, get_array(expr.dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &arr));
    for (size_type i = 0; i < expr.size(); ++i) arr[i] = expr.at(i);
    PetscFunctionReturnVoid();
  }

  // ==================================================================================== //

  template <typename U>
  ManagedType& operator=(ManagedType<U>&& other) noexcept;

  // ==================================================================================== //

  PETSC_NODISCARD PetscErrorCode get_array(PetscDeviceContext, PetscMemType, PetscMemoryAccessMode, PetscBool, value_type **) noexcept;
  PETSC_NODISCARD PetscErrorCode clear(PetscDeviceContext = nullptr) noexcept;
  PETSC_NODISCARD PetscErrorCode reserve(size_type,PetscDeviceContext = nullptr) noexcept;

  PETSC_NODISCARD PetscObjectId    id()           const noexcept { return id_;   }
  PETSC_NODISCARD size_type        size()         const noexcept { return size_; }
  PETSC_NODISCARD PetscOffloadMask offload_mask() const noexcept { return mask_; }

private:
  template <typename U>
  class inner_type
  {
  public:
    using value_type = U;

    constexpr explicit inner_type(ManagedType &outer, value_type&& value) noexcept
      : outer_(outer), value_(std::forward<value_type>(value))
    { }

    constexpr operator value_type() const noexcept { return value_; }

  protected:
    ManagedType &outer_;
    value_type   value_;
  };

  class purity_type : inner_type<bool>
  {
    using base_type = inner_type<bool>;

  public:
    using value_type = typename base_type::value_type;
    using base_type::base_type;
    using base_type::operator value_type;

    constexpr purity_type& operator=(value_type purity) noexcept
    {
      PetscFunctionBegin;
      this->value_ = purity;
      if (!purity && this->outer_.parent_) this->outer_.parent_->pure_ = purity;
      PetscFunctionReturn(*this);
    }
  };

  class mask_type : inner_type<PetscOffloadMask>
  {
    using base_type = inner_type<PetscOffloadMask>;

  public:
    using value_type = typename base_type::value_type;
    using base_type::base_type;
    using base_type::operator value_type;

    constexpr mask_type& operator=(value_type mask) noexcept
    {
      PetscFunctionBegin;
      if (this->value_ != mask) {
        this->value_ = mask;
        // should not update the parent if the mask did not change!
        if (this->outer_.parent_) this->outer_.parent_->mask_ = mask;
      }
      PetscFunctionReturn(*this);
    }
  };

  PetscObjectId  id_     = init_id_();
  size_type      size_   = 0;
  storage_type   host_{};
  storage_type   device_{};
  mask_type      mask_;
  purity_type    pure_;
  ManagedType   *parent_ = nullptr;

  PETSC_NODISCARD static PetscObjectId init_id_() noexcept {
    static PetscObjectId id = 0;

    PetscFunctionBegin;
    //PetscCallAbort(PETSC_COMM_SELF,PetscObjectNewId_Internal(&id));
    PetscFunctionReturn(id++);
  }

  PETSC_NODISCARD static PetscOffloadMask init_mask_(const value_type *host_ptr, const value_type *device_ptr, PetscOffloadMask in_mask) noexcept
  {
    PetscFunctionBegin;
    if (host_ptr && device_ptr) {
      PetscAssertAbort(in_mask != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Set both host and device pointer but offloadmask was %s", PetscOffloadMasks(in_mask));
      // this is the only instance in which we believe whatever the user has fed us
      PetscFunctionReturn(in_mask);
    } else if (host_ptr) {
      // clearly no device_ptr, so must be on cpu
      PetscFunctionReturn(PETSC_OFFLOAD_CPU);
    } else if (device_ptr) {
      // clearly no host_ptr, so must be on gpu
      PetscFunctionReturn(PETSC_OFFLOAD_GPU);
    }
    // user gave us nothing, we are nowhere
    PetscFunctionReturn(PETSC_OFFLOAD_UNALLOCATED);
  }

  template <typename... Args>
  PETSC_NODISCARD static storage_type construct_storage_(PetscCopyMode mode, Args&&... args ) noexcept
  {
    switch (mode) {
    case PETSC_OWN_POINTER:
      return {memory::move_init_t{},std::forward<Args>(args)...};
    case PETSC_USE_POINTER:
      return {memory::reference_init_t{},std::forward<Args>(args)...};
    case PETSC_COPY_VALUES:
      return {memory::copy_init_t{},std::forward<Args>(args)...};
    }
  }

  PETSC_NODISCARD bool pure()   const noexcept { return pure_;   }
  PETSC_NODISCARD bool impure() const noexcept { return !pure(); }

  PETSC_NODISCARD auto at_impl_(size_type idx) const noexcept
  {
    PetscCheckAbort(pure(),PETSC_ERR_ORDER,PETSC_COMM_SELF,"did not prime the type\n");
    return host_.data()[idx];
  }

  PETSC_NODISCARD PetscErrorCode prefetch_impl_(PetscDeviceContext dctx) noexcept
  {
    value_type *unused;

    PetscFunctionBegin;
    PetscCall(get_array(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &unused));
    PetscFunctionReturn(0);
  }
};

template <typename T>
template <typename U>
inline ManagedType<T>& ManagedType<T>::operator=(ManagedType<U>&& other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    PetscCall(this->clear());
    id_     = std::exchange(other.id_, init_id_());
    size_   = std::exchange(other.size_, 0);
    host_   = std::move(other.host_);
    device_ = std::move(other.device_);
    mask_   = mask_type(*this, std::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED));
    pure_   = purity_type(*this, std::exchange(other.pure_, true));
    parent_ = std::exchange(other.parent_, nullptr);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::clear(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
#if 0
  PetscCall(PetscManagedTypeCheckLock_Private(*scal,PETSC_FALSE));
#endif
  PetscCall(host_.clear(dctx));
  PetscCall(device_.clear(dctx));
  size_ = 0;
  pure_ = true;
  mask_ = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::get_array(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, value_type **ptr) noexcept
{
  const auto oid                    = id();
  const auto get_array_from_storage = [&](storage_type &dest, const storage_type &src, PetscOffloadMask requested_mask, PetscDeviceCopyMode direction)
  {
    PetscFunctionBegin;
    PetscAssert(requested_mask != PETSC_OFFLOAD_BOTH, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have %s!", PetscOffloadMasks(requested_mask));
    PetscCall(dest.get_pointer(dctx, size(), ptr));
    if (offload_mask() == PETSC_OFFLOAD_UNALLOCATED) {
      PetscAssert(!src, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offload mask is PETSC_OFFLOAD_UNALLOCATED but have src pointer %p!", src.data());
      mask_ = requested_mask;
    }
    // no need to do anything if we already match the desired offload
    if (offload_mask() == requested_mask) PetscFunctionReturn(0);
    // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
    // it now
    if (PetscMemoryAccessRead(mode) && src && (offload_mask() != PETSC_OFFLOAD_BOTH)) {
      mask_ = PETSC_OFFLOAD_BOTH;
      PetscCall(PetscDeviceArrayCopy(dctx, dest.data(), src.data(), size(), direction));
    }
    // if we have any kind of write then mask is set to the specific requested version (which
    // must not be OFFLOAD_BOTH)
    if (PetscMemoryAccessWrite(mode)) mask_ = requested_mask;
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
  std::cout << "get_array(" << (dctx ? PetscObjectCast(dctx)->name : "(unnamed)") << ", " << PetscMemTypes(mtype) << ", " << PetscMemoryAccessModes(mode) << ", " << PetscBools[sync] << ")\n";
  //PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(ptr, 5);
  //PetscCall(check_lock_(false));
  PetscCheck(!(PetscOffloadUnallocated(offload_mask()) && PetscMemoryAccessRead(mode)), PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_ARG_WRONG, "Trying to read (using %s) from a managed type (id %" PetscInt64_FMT ") that has not been written to (has offload mask %s)", PetscMemoryAccessModes(mode), oid, PetscOffloadMasks(offload_mask()));
  *ptr = nullptr;
  if (!size()) PetscFunctionReturn(0);
  // we will actually touch the values so mark them now
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, oid, mode, nullptr));

  // retrieve the pointer
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(get_array_from_storage(host_, device_, PETSC_OFFLOAD_CPU, PETSC_DEVICE_COPY_DTOH));
    break;
  case PETSC_MEMTYPE_DEVICE:
    PetscCall(get_array_from_storage(device_, host_, PETSC_OFFLOAD_GPU, PETSC_DEVICE_COPY_HTOD));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemType must be either PETSC_MEMTYPE_HOST (%d) or PETSC_MEMTYPE_DEVICE (%d) not %d", static_cast<int>(PETSC_MEMTYPE_HOST), static_cast<int>(PETSC_MEMTYPE_DEVICE), static_cast<int>(mtype));
    break;
  }

  // if user intends to write to device in any capacity then we are impure
  if (PetscMemTypeDevice(mtype) && PetscMemoryAccessWrite(mode)) pure_ = false;
  // also sets the parents mask if needed
  mask_ = offload_mask();
  // REVIEW ME:
  // if we are pure, there is no need to synchronize (I think)
  if (sync && impure()) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    if (PetscMemTypeHost(mtype)) pure_ = true;
  }
  PetscAssert(*ptr, PetscObjectComm(PetscObjectCast(dctx)), PETSC_ERR_PLIB, "ManagedType (id %" PetscInt64_FMT ") Returned null pointer for mtype %s", oid, PetscMemTypes(mtype));
  PetscFunctionReturn(0);
}

template <typename T>
inline PetscErrorCode ManagedType<T>::reserve(size_type n, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (size() >= n) PetscFunctionReturn(0);
  size_ = n;
  PetscCall(host_.reserve(dctx,n));
  PetscFunctionReturn(0);
}

using ManagedReal = ManagedType<PetscReal>;

template class ManagedType<PetscReal>;

} // namespace Petsc

#endif // __cplusplus
// clang-format on
#endif // PETSCMANAGEDTYPE_HPP
