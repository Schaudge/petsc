#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include "objpool.hpp"

// gcc 4.XXX supports all of C++11 _except_ std::is_trivially_copyable, so we guard
// against it here. Note the check for __GLIBCXX__ -- clang may use gcc's libstdc++ on
// certain systems so it isn't enough to check __GNUC__ (which clang defines anyways)
#if !defined(__GNUC__) || __GNUC__ >= 5 || !defined(__GLIBCXX__)
#define PETSC_HAVE_TRIVIALLY_COPYABLE 1
#include <type_traits> // std::is_trivially_copyable
#endif

template <typename T>
struct PetscManagedTypeAllocator : Petsc::AllocatorBase<T> {
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create(T *mscal)) {
    PetscFunctionBegin;
    PetscCall(PetscNew(mscal));
    PetscCall(reset(*mscal, false));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(T mscal)) {
    PetscFunctionBegin;
    PetscCall(PetscFree(mscal));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode reset(T mscal, bool zero = true)) {
    PetscFunctionBegin;
    if (zero) {
#if defined(PETSC_HAVE_TRIVIALLY_COPYABLE)
      static_assert(std::is_trivially_copyable<Petsc::util::remove_pointer_t<T>>::value, "");
#endif
      memset(mscal, 0, sizeof(*mscal));
    }
    mscal->h_cmode = PETSC_OWN_POINTER;
    mscal->d_cmode = PETSC_OWN_POINTER;
    mscal->pure    = PETSC_TRUE;
    static_assert(Petsc::util::integral_value(PETSC_OWN_POINTER) != 0, "");
    static_assert(Petsc::util::integral_value(PETSC_DEVICE_HOST) == 0, "");
    static_assert(Petsc::util::integral_value(PETSC_OFFLOAD_UNALLOCATED) == 0, "");
    static_assert(Petsc::util::integral_value(PETSC_FALSE) == 0, "");
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscErrorCode finalize()) {
    return 0;
  }
};

// wrapper to make the static pool declaration "automatic"
template <typename T>
struct PetscManagedTypePool {
  using pool_type = Petsc::ObjectPool<T, PetscManagedTypeAllocator<T>>;

  static pool_type pool;
};

template <typename T>
typename PetscManagedTypePool<T>::pool_type PetscManagedTypePool<T>::pool;

#define PetscManagedTypeAllocate(scal)   PetscManagedTypePool<PetscManagedType>::pool.get(*(scal))
#define PetscManagedTypeDeallocate(scal) PetscManagedTypePool<PetscManagedType>::pool.reclaim(std::move(scal))

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Scalar
#define PetscTypeSuffix_L scalar
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Real
#define PetscTypeSuffix_L real
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Int
#define PetscTypeSuffix_L int
#include "managedtype.inl"
