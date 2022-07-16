#ifndef PETSC_CPP_TYPE_TRAITS_HPP
#define PETSC_CPP_TYPE_TRAITS_HPP

#if defined(__cplusplus)
#include <petsc/private/petscimpl.h> // _p_PetscObject
#include <petsc/private/cpp/macros.hpp>

#include <type_traits>

namespace Petsc {

namespace util {

#if __cplusplus >= 201703L // C++17
using std::void_t;
#else  // C++17
template <class...>
using void_t = void;
#endif // C++17

#if __cplusplus >= 201402L // C++14
using std::add_const_t;
using std::add_pointer_t;
using std::conditional_t;
using std::decay_t;
using std::enable_if_t;
using std::remove_const_t;
using std::remove_cv_t;
using std::remove_pointer_t;
using std::underlying_type_t;
#else  // C++14
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;
template <class T>
using remove_const_t = typename std::remove_const<T>::type;
template <class T>
using add_const_t = typename std::add_const<T>::type;
template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;
template <class T>
using underlying_type_t = typename std::underlying_type<T>::type;
template <class T>
using remove_pointer_t = typename std::remove_pointer<T>::type;
template <class T>
using add_pointer_t = typename std::add_pointer<T>::type;
template <class T>
using decay_t = typename std::decay<T>::type;
#endif // C++14

template <typename... T>
struct always_false : std::false_type { };

namespace detail {

template <typename T, typename U = _p_PetscObject>
struct is_petsc_object_impl : std::false_type { };

template <typename T>
struct is_petsc_object_impl<T, PetscObject> : std::true_type { };

template <typename T>
struct is_petsc_object_impl<T, decltype(T::hdr)> : conditional_t<(!std::is_pointer<T>::value) && std::is_class<T>::value && std::is_standard_layout<T>::value, std::true_type, std::false_type> { };

} // namespace detail

template <typename T>
using is_petsc_object = detail::is_petsc_object_impl<remove_pointer_t<decay_t<T>>>;

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr underlying_type_t<T> integral_value(T value)) {
  static_assert(std::is_enum<T>::value, "");
  return static_cast<underlying_type_t<T>>(value);
}

} // namespace util

} // namespace Petsc

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr Petsc::util::remove_const_t<T> &PetscRemoveConstCast(T &object)) {
  return const_cast<Petsc::util::remove_const_t<T> &>(object);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr T &PetscRemoveConstCast(const T &object)) {
  return const_cast<T &>(object);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr T *&PetscRemoveConstCast(const T *&object)) {
  return const_cast<T *&>(object);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr Petsc::util::add_const_t<T> &PetscAddConstCast(T &object)) {
  return const_cast<Petsc::util::add_const_t<T> &>(std::forward<T>(object));
}

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr Petsc::util::add_const_t<T> *&PetscAddConstCast(T *&object)) {
  static_assert(!std::is_const<T>::value, "");
  return const_cast<Petsc::util::add_const_t<T> *&>(std::forward<T>(object));
}

// PetscObjectCast() - Cast an object to PetscObject
//
// input param:
// object - the object to cast
//
// output param:
// [return value] - The resulting PetscObject
//
// notes:
// This function checks that the object passed in is in fact a PetscObject, and hence requires
// the full definition of the object. This means you must include the appropriate header
// containing the _p_<object> struct definition
//
//   not available from Fortran
template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr PetscObject &PetscObjectCast(T &object)) {
  static_assert(Petsc::util::is_petsc_object<T>::value, "If this is a PetscObject then the private definition of the struct must be visible for this to work");
  return reinterpret_cast<PetscObject &>(object);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(constexpr const PetscObject &PetscObjectCast(const T &object)) {
  static_assert(Petsc::util::is_petsc_object<T>::value, "If this is a PetscObject then the private definition of the struct must be visible for this to work");
  return reinterpret_cast<const PetscObject &>(object);
}
#else // __cplusplus

#define PetscObjectCast(...) ((PetscObject)(__VA_ARGS__))

#endif // __cplusplus

#endif // PETSC_CPP_TYPE_TRAITS_HPP
