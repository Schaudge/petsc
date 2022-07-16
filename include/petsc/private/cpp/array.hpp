#ifndef PETSC_CPP_ARRAY_HPP
#define PETSC_CPP_ARRAY_HPP

#if defined(__cplusplus)
#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/type_traits.hpp>

#include <array>

namespace Petsc {

namespace util {

namespace detail {

template <class D, class...>
struct return_type_helper {
  using type = D;
};
template <class... T>
struct return_type_helper<void, T...> : std::common_type<T...> { };

template <class D, class... T>
using array_return_type = std::array<typename return_type_helper<D, T...>::type, sizeof...(T)>;

template <typename T, std::size_t NL, std::size_t... IL, std::size_t NR, std::size_t... IR>
static inline constexpr std::array<T, NL + NR> concat_array_impl(const std::array<T, NL> &l, const std::array<T, NR> &r, index_sequence<IL...>, index_sequence<IR...>) noexcept(noexcept(std::is_nothrow_copy_constructible<T>::value)) {
  return {l[IL]..., r[IR]...};
}

} // namespace detail

template <class D = void, class... T>
static inline constexpr detail::array_return_type<D, T...> make_array(T &&...t) noexcept(noexcept(std::is_nothrow_constructible<detail::array_return_type<D, T...>>::value)) {
  return {std::forward<T>(t)...};
}

template <typename T, std::size_t NL, std::size_t NR>
static inline constexpr auto concat_array(const std::array<T, NL> &l, const std::array<T, NR> &r) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(detail::concat_array_impl(l, r, make_index_sequence<NL>{}, make_index_sequence<NR>{}));

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_ARRAY_HPP
