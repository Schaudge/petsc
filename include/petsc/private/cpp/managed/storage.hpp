#ifndef PETSC_CPP_MANAGED_STORAGE_HPP
#define PETSC_CPP_MANAGED_STORAGE_HPP

#ifdef __cplusplus
  #include <petsc/private/deviceimpl.h>
  #include <petsc/private/cpp/type_traits.hpp>
  #include <petsc/private/cpp/utility.hpp>

  #include <iterator>  // std::distance()
  #include <algorithm> // std::copy_n()

namespace Petsc
{

struct copy_init_t { };
struct move_init_t { };
struct reference_init_t { };

namespace iterator
{

template <typename T>
class BasicPointerIterator {
public:
  #if PETSC_CPP_VERSION >= 20
  using iterator_category = std::contiguous_iterator_tag;
  #else
  using iterator_category = std::random_access_iterator_tag;
  #endif
  using difference_type = std::ptrdiff_t;
  using value_type      = T;
  using pointer         = value_type *;
  using reference       = value_type &;

  // all basic_pointer_iterators are friends :)
  template <typename>
  friend class basic_pointer_iterator;

  constexpr explicit BasicPointerIterator(pointer p = nullptr) noexcept : ptr_{p} { }
  constexpr BasicPointerIterator(const BasicPointerIterator &) noexcept            = default;
  constexpr BasicPointerIterator(BasicPointerIterator &&) noexcept                 = default;
  constexpr BasicPointerIterator &operator=(const BasicPointerIterator &) noexcept = default;
  constexpr BasicPointerIterator &operator=(BasicPointerIterator &&) noexcept      = default;

  // conversion from another pointer iterator, this also includes non-const-to-const
  // conversion!
  template <typename U, util::enable_if_t<std::is_convertible<U, value_type>::value, int> = 0>
  constexpr BasicPointerIterator(const BasicPointerIterator<U> &other) noexcept : ptr_{other.base()}
  {
  }

  PETSC_NODISCARD constexpr pointer base() const noexcept { return ptr_; }

  constexpr reference operator*() const noexcept { return *ptr_; }

  constexpr pointer                    operator->() noexcept { return ptr_; }
  constexpr util::add_const_t<pointer> operator->() const noexcept { return ptr_; }

  // Prefix decrement
  constexpr BasicPointerIterator &operator--() noexcept
  {
    --ptr_;
    return *this;
  }

  // Postfix decrement
  constexpr BasicPointerIterator operator--(int) noexcept
  {
    BasicPointerIterator tmp(*this);

    --(*this);
    return tmp;
  }

  // Prefix increment
  constexpr BasicPointerIterator &operator++() noexcept
  {
    ++ptr_;
    return *this;
  }

  // Postfix increment
  constexpr BasicPointerIterator operator++(int) noexcept
  {
    BasicPointerIterator tmp(*this);

    ++(*this);
    return tmp;
  }

  constexpr BasicPointerIterator &operator+=(difference_type diff) noexcept
  {
    ptr_ += diff;
    return *this;
  }

  constexpr BasicPointerIterator &operator-=(difference_type diff) noexcept
  {
    ptr_ -= diff;
    return *this;
  }

  constexpr BasicPointerIterator operator+(difference_type diff) const noexcept
  {
    BasicPointerIterator tmp(*this);

    tmp += diff;
    return tmp;
  }

  constexpr BasicPointerIterator operator-(difference_type diff) const noexcept { return *this + (-diff); }

private:
  pointer ptr_ = nullptr;
};

template <typename L, typename R>
constexpr typename BasicPointerIterator<L>::difference_type operator-(BasicPointerIterator<L> lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return lhs.base() - rhs.base();
}

template <typename L, typename R>
constexpr bool operator==(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return lhs.base() == rhs.base();
}

template <typename L, typename R>
constexpr bool operator!=(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return !(lhs == rhs);
}

template <typename L, typename R>
constexpr bool operator<(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return lhs.base() < rhs.base();
}

template <typename L, typename R>
constexpr bool operator>(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return rhs < lhs;
}

template <typename L, typename R>
constexpr bool operator<=(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return !(lhs > rhs);
}

template <typename L, typename R>
constexpr bool operator>=(const BasicPointerIterator<L> &lhs, const BasicPointerIterator<R> &rhs) noexcept
{
  return !(lhs < rhs);
}

template <typename T>
constexpr BasicPointerIterator<T> operator+(typename BasicPointerIterator<T>::difference_type diff, BasicPointerIterator<T> rhs) noexcept
{
  rhs += diff;
  return rhs;
}

template <typename T>
constexpr BasicPointerIterator<T> operator-(typename BasicPointerIterator<T>::difference_type diff, BasicPointerIterator<T> rhs) noexcept
{
  rhs -= diff;
  return rhs;
}

} // namespace iterator

namespace memory
{

// ==========================================================================================
// ManagedStorage
// ==========================================================================================

template <typename T>
class ManagedStorage {
public:
  using value_type     = T;
  using size_type      = std::size_t;
  using const_iterator = iterator::BasicPointerIterator<const value_type>;
  using iterator       = iterator::BasicPointerIterator<value_type>;

  ManagedStorage() noexcept = default;
  ~ManagedStorage() noexcept;

  constexpr explicit ManagedStorage(PetscMemType) noexcept;
  ManagedStorage(PetscDeviceContext, PetscMemType, size_type) noexcept;

  template <typename Iterator>
  ManagedStorage(copy_init_t, PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &) noexcept;
  template <typename Iterator>
  ManagedStorage(reference_init_t, PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &) noexcept;
  template <typename Iterator>
  ManagedStorage(move_init_t, PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &) noexcept;

  // ASYNC TODO
  ManagedStorage(const ManagedStorage &) noexcept            = delete;
  ManagedStorage &operator=(const ManagedStorage &) noexcept = delete;

  ManagedStorage(ManagedStorage &&) noexcept;
  ManagedStorage &operator=(ManagedStorage &&) noexcept;

  PETSC_NODISCARD value_type       *data() noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *cdata() const noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *data() const noexcept { return cdata(); }
  PETSC_NODISCARD bool              empty() const noexcept { return capacity() == 0; }
  PETSC_NODISCARD size_type         capacity() const noexcept { return attr_.size / sizeof(value_type); }
  PETSC_NODISCARD PetscMemType      mem_type() const noexcept { return attr_.mtype; }

  PETSC_NODISCARD iterator begin() noexcept { return iterator{data()}; }
  PETSC_NODISCARD iterator end(size_type size) noexcept { return begin() + static_cast<typename iterator::difference_type>(size); }

  PETSC_NODISCARD const_iterator cbegin() const noexcept { return const_iterator{data()}; }
  PETSC_NODISCARD const_iterator cend(size_type size) const noexcept { return cbegin() + static_cast<typename const_iterator::difference_type>(size); }

  PETSC_NODISCARD const_iterator begin() const noexcept { return cbegin(); }
  PETSC_NODISCARD const_iterator end(size_type size) const noexcept { return cend(size); }

  PetscErrorCode mark_begin(PetscDeviceContext, PetscMemoryAccessMode) const noexcept;
  PetscErrorCode mark_end(PetscDeviceContext, PetscMemoryAccessMode) const noexcept;
  PetscErrorCode reserve(PetscDeviceContext, size_type) noexcept;
  PetscErrorCode destroy(PetscDeviceContext = nullptr) noexcept;
  template <typename Iterator>
  PetscErrorCode assign(PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &, bool = false) noexcept;
  PetscErrorCode assign(PetscDeviceContext, const ManagedStorage &) noexcept;
  PetscErrorCode copy_from(PetscDeviceContext, const ManagedStorage &) noexcept;

  void        swap(ManagedStorage &) noexcept;
  friend void swap(ManagedStorage &lhs, ManagedStorage &rhs) noexcept { lhs.swap(rhs); }

private:
  value_type                    *ptr_{nullptr};
  mutable PetscPointerAttributes attr_{};
  bool                           own_ptr_{true};

  template <typename Iterator>
  ManagedStorage(Iterator, Iterator, const PetscPointerAttributes &, bool) noexcept;

  PetscErrorCode ensure_ptr_attr_() const noexcept;

  template <typename Iterator>
  PetscErrorCode assign_(PetscDeviceContext, Iterator, Iterator, size_type, const PetscPointerAttributes &, std::random_access_iterator_tag) noexcept;
  // ASYNC TODO
  template <typename Iterator>
  PetscErrorCode assign_(PetscDeviceContext, Iterator, Iterator, size_type, const PetscPointerAttributes &, ...) noexcept = delete;
};

// ==========================================================================================
// ManagedStorage - Private API
// ==========================================================================================

template <typename T>
template <typename Iterator>
inline ManagedStorage<T>::ManagedStorage(Iterator begin, Iterator end, const PetscPointerAttributes &attr, bool own) noexcept : ptr_{begin}, attr_{attr}, own_ptr_{own}
{
  PetscFunctionBegin;
  if ((attr_.id == PETSC_UNKNOWN_MEMORY_ID) || (attr_.id == PETSC_DELETED_MEMORY_ID)) {
    auto found = PETSC_FALSE;

    if (ptr_) PetscCallAbort(PETSC_COMM_SELF, PetscDeviceGetPointerAttributes(ptr_, &attr_, &found));
    if (!found) {
      if (ptr_) {
        // if we did not find the pointer then we can only assume this to be host stack memory.
        PetscCheckAbort(PetscMemTypeHost(attr_.mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Don't know ID of unknown pointer to %s", PetscMemTypeToString(attr_.mtype));
        attr_.id = PETSC_STACK_MEMORY_ID;
      } else {
        attr_.id = PETSC_DELETED_MEMORY_ID;
      }
      attr_.size  = std::max(std::distance(begin, end) * sizeof(value_type), attr_.size);
      attr_.align = std::max(alignof(value_type), attr_.align);
    }
  }
  PetscFunctionReturnVoid();
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::ensure_ptr_attr_() const noexcept
{
  PetscFunctionBegin;
  if ((attr_.id == PETSC_DELETED_MEMORY_ID) || (attr_.id == PETSC_UNKNOWN_MEMORY_ID)) {
    PetscBool found;

    PetscCheck(data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Do not have a pointer to get attributes for!");
    PetscCall(PetscDeviceGetPointerAttributes(data(), &attr_, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find attributes for pointer %p", (void *)data());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode ManagedStorage<T>::assign_(PetscDeviceContext dctx, Iterator begin, Iterator, size_type n, const PetscPointerAttributes &attr, std::random_access_iterator_tag) noexcept
{
  PetscFunctionBegin;
  static_assert(sizeof(value_type) == sizeof(*begin), "");
  PetscCall(ensure_ptr_attr_());
  PetscCall(PetscDeviceMemcpy(dctx, data(), std::addressof(*begin), n * sizeof(value_type), &attr_, &attr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// ManagedStorage - Public API
// ==========================================================================================

// memtype constructor
template <typename T>
constexpr inline ManagedStorage<T>::ManagedStorage(PetscMemType mtype) noexcept : attr_{mtype}
{
}

// size constructor
template <typename T>
inline ManagedStorage<T>::ManagedStorage(PetscDeviceContext dctx, PetscMemType mtype, size_type n) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PetscObjectComm((PetscObject)dctx), PetscDeviceMalloc(dctx, mtype, n, &ptr_));
  PetscCallAbort(PetscObjectComm((PetscObject)dctx), PetscDeviceGetPointerAttributes(data(), &attr_, nullptr));
  PetscFunctionReturnVoid();
}

// copy constructor
template <typename T>
template <typename Iterator>
inline ManagedStorage<T>::ManagedStorage(copy_init_t, PetscDeviceContext dctx, Iterator begin, Iterator end, const PetscPointerAttributes &attr) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PetscObjectComm((PetscObject)dctx), assign(dctx, std::move(begin), std::move(end), attr));
  PetscFunctionReturnVoid();
}

// use constructor
template <typename T>
template <typename Iterator>
inline ManagedStorage<T>::ManagedStorage(reference_init_t, PetscDeviceContext, Iterator begin, Iterator end, const PetscPointerAttributes &attr) noexcept : ManagedStorage{std::move(begin), std::move(end), attr, false}
{
}

// move constructor
template <typename T>
template <typename Iterator>
inline ManagedStorage<T>::ManagedStorage(move_init_t, PetscDeviceContext, Iterator begin, Iterator end, const PetscPointerAttributes &attr) noexcept : ManagedStorage{std::move(begin), std::move(end), attr, true}
{
}

template <typename T>
inline ManagedStorage<T>::~ManagedStorage() noexcept
{
  PetscFunctionBegin;
  if (ptr_ && own_ptr_) {
    PetscBool init;

    PetscCallAbort(PETSC_COMM_SELF, PetscInitialized(&init));
    if (PetscLikely(init)) {
      PetscDeviceContext dctx;

      PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextGetCurrentContext(&dctx));
      PetscCallAbort(PETSC_COMM_SELF, PetscDeviceFree(dctx, ptr_));
    }
  }
  PetscFunctionReturnVoid();
}

// template <typename T>
// inline ManagedStorage<T>::ManagedStorage(const ManagedStorage &other) noexcept : ManagedStorage{copy_init_t{}, nullptr, other.mem_type(), other.cbegin(), other.cend(other.capacity())}
// {
// }

// template <typename T>
// inline ManagedStorage<T> &ManagedStorage<T>::operator=(const ManagedStorage &other) noexcept
// {
//   PetscFunctionBegin;
//   if (this != &other) PetscCallAbort(PETSC_COMM_SELF, assign(nullptr, other));
//   PetscFunctionReturn(*this);
// }

template <typename T>
inline ManagedStorage<T>::ManagedStorage(ManagedStorage &&other) noexcept
  // clang-format off
  : ptr_{util::exchange(other.ptr_, nullptr)},
    attr_{util::exchange(other.attr_, PetscPointerAttributes{other.mem_type()})},
    own_ptr_{util::exchange(other.own_ptr_, true)}
// clang-format on
{
}

template <typename T>
inline ManagedStorage<T> &ManagedStorage<T>::operator=(ManagedStorage &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    // delete our pointer (if we have one)
    PetscAssertAbort(other.mem_type() == mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(mem_type()), PetscMemTypeToString(other.mem_type()));
    PetscCallAbort(PETSC_COMM_SELF, destroy());
    ptr_     = util::exchange(other.ptr_, nullptr);
    attr_    = util::exchange(other.attr_, PetscPointerAttributes{attr_.mtype});
    own_ptr_ = util::exchange(other.own_ptr_, true);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::mark_begin(PetscDeviceContext dctx, PetscMemoryAccessMode mode) const noexcept
{
  PetscFunctionBegin;
  PetscCall(ensure_ptr_attr_());
  PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, attr_.id, mode, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::mark_end(PetscDeviceContext dctx, PetscMemoryAccessMode mode) const noexcept
{
  PetscFunctionBegin;
  PetscCall(ensure_ptr_attr_());
  PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, attr_.id, mode, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::reserve(PetscDeviceContext dctx, size_type n) noexcept
{
  PetscFunctionBegin;
  if ((n == 0) || (capacity() >= n)) PetscFunctionReturn(PETSC_SUCCESS);
  // It is assumed that the user provided enough capacity in the case where we don't own the
  // pointer. We cannot resize it. We have no way of knowing if the pointer is dynamically
  // allocated or pointer to a stack variable. Furthermore, we simply do. not. own. it.
  PetscCheck(own_ptr_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot increase capacity (from %zu to %zu) for a pointer that is not owned!", capacity(), n);
  if (data()) {
    PetscCall(PetscDeviceRealloc(dctx, n, &ptr_));
  } else {
    PetscCall(PetscDeviceMalloc(dctx, mem_type(), n, &ptr_));
  }
  PetscCall(PetscDeviceGetPointerAttributes(data(), &attr_, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (ptr_ && own_ptr_) PetscCall(PetscDeviceFree(dctx, ptr_));
  ptr_     = nullptr;
  attr_    = PetscPointerAttributes{attr_.mtype};
  own_ptr_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode ManagedStorage<T>::assign(PetscDeviceContext dctx, Iterator begin, Iterator end, const PetscPointerAttributes &attr, bool fast_assign) noexcept
{
  const auto n = std::distance(begin, end);

  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Begin < end");
  PetscCall(reserve(dctx, n));
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  if (fast_assign || (PetscMemTypeHost(attr.mtype) && (attr.id == PETSC_STACK_MEMORY_ID))) {
    PetscAssert(PetscMemTypeHost(attr.mtype) && PetscMemTypeHost(mem_type()), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot fast assign from %s to %s", PetscMemTypeToString(attr.mtype), PetscMemTypeToString(mem_type()));
    PetscCallCXX(std::copy_n(std::move(begin), n, this->begin()));
  } else {
    PetscCall(assign_(dctx, std::move(begin), std::move(end), static_cast<size_type>(n), attr, typename std::iterator_traits<Iterator>::iterator_category{}));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::assign(PetscDeviceContext dctx, const ManagedStorage &other) noexcept
{
  PetscFunctionBegin;
  PetscAssert(other.mem_type() == mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(mem_type()), PetscMemTypeToString(other.mem_type()));
  PetscCall(copy_from(dctx, other));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::copy_from(PetscDeviceContext dctx, const ManagedStorage &other) noexcept
{
  PetscFunctionBegin;
  if (&other == this) PetscFunctionReturn(PETSC_SUCCESS);
  // ASYNC TODO this should really be calling some kind of iterator version of "copy()",
  // currently assign() and copy() do pretty much the same things.
  PetscCall(other.ensure_ptr_attr_());
  PetscCall(assign(dctx, other.cbegin(), other.cend(other.capacity()), other.attr_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline void ManagedStorage<T>::swap(ManagedStorage &other) noexcept
{
  using std::swap;

  swap(ptr_, other.ptr_);
  swap(attr_, other.attr_);
  swap(own_ptr_, other.own_ptr_);
}

} // namespace memory

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_MANAGED_STORAGE_HPP
