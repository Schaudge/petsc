#ifndef PETSC_CPP_FUNCTIONAL_HPP
#define PETSC_CPP_FUNCTIONAL_HPP

#if defined(__cplusplus)
#include <petsc/private/cpp/type_traits.hpp> // decay_t
#include <petsc/private/cpp/tuple.hpp>       // tuple_element_t

namespace Petsc {

namespace util {

namespace detail {

struct can_call_test {
  template <typename F, typename... A>
  static decltype(std::declval<F>()(std::declval<A>()...), std::true_type()) f(int);

  template <typename F, typename... A>
  static std::false_type f(...);
};

// generic template
template <typename T>
struct func_traits_impl : func_traits_impl<decltype(&T::operator())> { };

// function pointers
template <typename Ret, typename... Args>
struct func_traits_impl<Ret (*)(Args...)> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

// class-like operator()
template <typename C, typename Ret, typename... Args>
struct func_traits_impl<Ret (C::*)(Args...) const> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

template <typename C, typename Ret, typename... Args>
struct func_traits_impl<Ret (C::*)(Args...)> {
  using result_type = Ret;

  template <std::size_t ix>
  struct arg {
    using type = util::tuple_element_t<ix, std::tuple<Args...>>;
  };
};

} // namespace detail

template <typename F, typename... A>
struct can_call : decltype(detail::can_call_test::f<F, A...>(0)) { };

template <typename... A, typename F>
static inline constexpr can_call<F, A...> is_callable_with(F &&) noexcept {
  return can_call<F, A...>{};
}

template <typename T>
struct func_traits : detail::func_traits_impl<decay_t<T>> {
  template <std::size_t idx>
  using arg_t = typename detail::func_traits_impl<decay_t<T>>::template arg<idx>::type;
};

} // namespace util

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_FUNCTIONAL_HPP
