#ifndef PETSC_CPP_TYPE_TRAITS_HPP
#define PETSC_CPP_TYPE_TRAITS_HPP

#if defined(__cplusplus)
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

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_TYPE_TRAITS_HPP
