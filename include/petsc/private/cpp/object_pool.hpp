#ifndef PETSCOBJECTPOOL_HPP
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>

#if defined(__cplusplus)
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/register_finalize.hpp>
#include <stack>

namespace Petsc {

// Allocator ABC for interoperability with C ctors and dtors.
template <typename T>
class AllocatorBase {
public:
  using value_type = T;

  PETSC_NODISCARD PetscErrorCode        create(value_type *) noexcept  = delete;
  PETSC_NODISCARD PetscErrorCode        destroy(value_type &) noexcept = delete;
  PETSC_NODISCARD static PetscErrorCode reset(value_type &) noexcept { return 0; }
  PETSC_NODISCARD static PetscErrorCode finalize() noexcept { return 0; }

protected:
  // make the constructor protected, this forces this class to be derived from to ever be
  // instantiated
  AllocatorBase() noexcept = default;
};

// Default allocator that performs the bare minimum of petsc object creation and
// desctruction
template <typename T>
class CAllocator : public AllocatorBase<T> {
public:
  using allocator_type = AllocatorBase<T>;
  using value_type     = typename allocator_type::value_type;

  PETSC_NODISCARD PetscErrorCode create(value_type *obj) const noexcept {
    PetscFunctionBegin;
    PetscCall(PetscNew(obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(value_type &obj) const noexcept {
    PetscFunctionBegin;
    PetscUseTypeMethod(obj, destroy);
    PetscCall(PetscHeaderDestroy(&obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(value_type &obj) const noexcept {
    PetscFunctionBegin;
    PetscCall(this->destroy(obj));
    PetscCall(this->create(&obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode finalize() const noexcept { return 0; }
};

namespace detail {

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <typename T, class Allocator>
class ObjectPoolBase {
public:
  using allocator_type = Allocator;
  using value_type     = typename allocator_type::value_type;

protected:
  allocator_type alloc_;

  PETSC_NODISCARD allocator_type       &allocator() noexcept { return alloc_; }
  PETSC_NODISCARD const allocator_type &callocator() const noexcept { return alloc_; }

  // default constructor
  constexpr ObjectPoolBase() noexcept(std::is_nothrow_default_constructible<allocator_type>::value) : alloc_() { }

  // const copy constructor
  explicit ObjectPoolBase(const allocator_type &alloc) : alloc_(alloc) { }

  // move constructor
  explicit ObjectPoolBase(allocator_type &&alloc) noexcept(std::is_nothrow_move_constructible<allocator_type>::value) : alloc_(std::move(alloc)) { }

  static_assert(std::is_base_of<AllocatorBase<value_type>, Allocator>::value, "");
};

} // namespace detail

// default implementation, use the petsc c allocator
template <typename T, class Allocator = CAllocator<T>>
class ObjectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locallity. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, class Allocator>
class ObjectPool : detail::ObjectPoolBase<T, Allocator>, public RegisterFinalizeable<ObjectPool<T, Allocator>> {
protected:
  using base_type = detail::ObjectPoolBase<T, Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type     = typename base_type::value_type;
  using stack_type     = std::stack<value_type>;
  using base_type::allocator;
  using base_type::callocator;

  // default constructor
  constexpr ObjectPool() noexcept(std::is_nothrow_default_constructible<allocator_type>::value &&std::is_nothrow_default_constructible<stack_type>::value) : stack_() { }

  // destructor
  ~ObjectPool() noexcept(std::is_nothrow_destructible<stack_type>::value &&std::is_nothrow_destructible<allocator_type>::value) { PetscCallAbort(PETSC_COMM_SELF, finalize_()); }

  // copy constructor
  ObjectPool(ObjectPool &other) noexcept(std::is_nothrow_copy_constructible<stack_type>::value) : stack_(other.stack_) { }

  // const copy constructor
  ObjectPool(const ObjectPool &other) noexcept(std::is_nothrow_copy_constructible<stack_type>::value) : stack_(other.stack_) { }

  // move constructor
  ObjectPool(ObjectPool &&other) noexcept(std::is_nothrow_move_constructible<stack_type>::value) : stack_(std::move(other.stack_)) { }

  // copy constructor with allocator
  explicit ObjectPool(const allocator_type &alloc) : base_type(alloc) { }

  // move constructor with allocator
  explicit ObjectPool(allocator_type &&alloc) noexcept(std::is_nothrow_move_constructible<allocator_type>::value) : base_type(std::move(alloc)) { }

  // Retrieve an object from the pool, if the pool is empty a new object is created instead
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode get(value_type &, Args &&...) noexcept;
  // Return an object to the pool, the object need not necessarily have been created by
  // the pool, note this only accepts r-value references. The pool takes ownership of all
  // managed objects.
  PETSC_NODISCARD PetscErrorCode reclaim(value_type &&) noexcept;
  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;

  // operators
  template <typename T_, class A_>
  PetscBool friend operator==(const ObjectPool<T_, A_> &, const ObjectPool<T_, A_> &) noexcept;

  template <typename T_, class A_>
  PetscBool friend operator<(const ObjectPool<T_, A_> &, const ObjectPool<T_, A_> &) noexcept;

private:
  stack_type stack_;
};

template <typename T, class Allocator>
inline PetscBool operator==(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return static_cast<PetscBool>(l.stack_ == r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator!=(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return !(l.stack_ == r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator<(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return static_cast<PetscBool>(l.stack_ < r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator>(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return l.stack_ > r.stack_;
}

template <typename T, class Allocator>
inline PetscBool operator>=(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return !(l.stack_ < r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator<=(const ObjectPool<T, Allocator> &l, const ObjectPool<T, Allocator> &r) noexcept {
  return !(r.stack_ < l.stack_);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T, Allocator>::finalize_() noexcept {
  PetscFunctionBegin;
  while (!stack_.empty()) {
    PetscCall(this->allocator().destroy(stack_.top()));
    PetscCallCXX(stack_.pop());
  }
  stack_ = stack_type{};
  PetscCall(this->allocator().finalize());
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
template <typename... Args>
inline PetscErrorCode ObjectPool<T, Allocator>::get(value_type &obj, Args &&...args) noexcept {
  PetscFunctionBegin;
  PetscCall(this->register_finalize());
  if (stack_.empty()) {
    PetscCall(this->allocator().create(&obj, std::forward<Args>(args)...));
  } else {
    PetscCallCXX(obj = std::move(stack_.top()));
    PetscCallCXX(stack_.pop());
    PetscCall(this->allocator().reset(obj, std::forward<Args>(args)...));
  }
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T, Allocator>::reclaim(value_type &&obj) noexcept {
  PetscFunctionBegin;
  if (PetscLikely(this->registered_)) {
    // allows const allocator_t& to be used if allocator defines a const reset
    PetscCallCXX(stack_.push(std::move(obj)));
  } else {
    // this is necessary if an object is "reclaimed" within another PetscFinalize() registered
    // cleanup after this object pool has returned from it's finalizer. In this case, instead
    // of pushing onto the stack we just destroy the object directly
    PetscCall(this->allocator().destroy(std::move(obj)));
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCOBJECTPOOL_HPP */
