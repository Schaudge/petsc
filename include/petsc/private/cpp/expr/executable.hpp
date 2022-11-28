#ifndef PETSC_CPP_EXPR_EXECUTABLE_HPP
#define PETSC_CPP_EXPR_EXECUTABLE_HPP

#include <petscdevice_cupm.h>
#include <petscmanagedmemory_fwd.hpp>

#include <petsc/private/deviceimpl.h>
#if PetscDefined(HAVE_CUPM)
  #include <petsc/private/cupminterface.hpp>
  #include "../src/sys/objects/device/impls/cupm/kernels.hpp"
#endif

#include <petsc/private/cpp/expr/base.hpp>
#include <petsc/private/cpp/expr/expression.hpp>
#include <petsc/private/cpp/expr/operators.hpp>

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/tuple.hpp>
#include <petsc/private/cpp/memory.hpp> // std::addressof
#include <petsc/private/cpp/array.hpp>

#include <cstddef> // std::size_t
#include <vector>
#include <algorithm> // std::sort

namespace Petsc
{

namespace expr
{

template <typename, typename...>
class Expression;

// ==========================================================================================
// ExecutableExpression
// ==========================================================================================

template <typename T>
class ExecutableExpression {
public:
  static_assert(is_expression<T>::value, "");
  using expression_type = T;
  using size_type       = typename T::size_type;

  template <typename U>
  ExecutableExpression(U &&, PetscDeviceContext) noexcept;
  ExecutableExpression()                                        = delete;
  ExecutableExpression(const ExecutableExpression &)            = delete;
  ExecutableExpression(ExecutableExpression &&)                 = delete;
  ExecutableExpression &operator=(const ExecutableExpression &) = delete;
  ExecutableExpression &operator=(ExecutableExpression &&)      = delete;

  PETSC_NODISCARD const expression_type &expr() const noexcept;
  PETSC_NODISCARD PetscDeviceContext     dctx() const noexcept;
  PETSC_NODISCARD size_type              size() const noexcept;

  template <typename U>
  PetscErrorCode Execute(ManagedMemory<U> &, PetscDeviceContext) const noexcept;

  template <typename U>
  PetscErrorCode Execute(ManagedMemory<U> &) const noexcept;

private:
  expression_type    expr_;
  PetscDeviceContext dctx_;
};

// ==========================================================================================
// ExecutableExpression -- Public API
// ==========================================================================================

template <typename T>
template <typename U>
inline ExecutableExpression<T>::ExecutableExpression(U &&expr, PetscDeviceContext dctx) noexcept : expr_{std::forward<U>(expr)}, dctx_{dctx}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextGetOptionalNullContext_Internal(&dctx_));
  PetscFunctionReturnVoid();
}

template <typename T>
inline PetscDeviceContext ExecutableExpression<T>::dctx() const noexcept
{
  return dctx_;
}

template <typename T>
inline typename ExecutableExpression<T>::size_type ExecutableExpression<T>::size() const noexcept
{
  return this->expr().size();
}

template <typename T>
inline const typename ExecutableExpression<T>::expression_type &ExecutableExpression<T>::expr() const noexcept
{
  return expr_;
}

namespace detail
{

#if PetscDefined(USING_CUPMCC)
template <typename T>
PETSC_KERNEL_DECL PETSC_LAUNCH_BOUNDS(1) void ExecuteLambda(std::size_t N, T lambda)
{
  ::Petsc::device::cupm::kernels::util::grid_stride_1D(N, std::move(lambda));
  return;
}
#endif

template <typename size_type, typename T>
inline PetscErrorCode CUPMExec(PetscDeviceContext dctx, PetscDeviceType dtype, size_type N, T &&lambda) noexcept
{
  // ASYNC TODO: clean this up
  void *stream;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
  if (dtype == PETSC_DEVICE_CUDA) {
#if PetscDefined(HAVE_CUDA) && PetscDefined(USING_CUPMCC)
    using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>;

  #if PetscDefined(USING_NVCC)
    static_assert(__nv_is_extended_device_lambda_closure_type(T), "");
  #endif
    PetscCall(CUPMInterface::PetscCUPMLaunchKernel1D(1, 0, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), ExecuteLambda<T>, N, std::forward<T>(lambda)));
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trying to launch a CUDA kernel using a PetscDeviceContext of type %s even though you did not configure with CUDA support. This should not happen!", PetscDeviceTypes[dtype]);
#endif
  } else if (dtype == PETSC_DEVICE_HIP) {
#if PetscDefined(HAVE_HIP) && PetscDefined(USING_CUPMCC)
    using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::HIP>;

    PetscCall(CUPMInterface::PetscCUPMLaunchKernel1D(1, 0, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), ExecuteLambda<T>, N, std::forward<T>(lambda)));
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trying to launch a HIP kernel using a PetscDeviceContext of type %s even though you did not configure with HIP support. This should not happen!", PetscDeviceTypes[dtype]);
#endif
  } else {
    static_cast<void>(N);
    static_cast<void>(lambda);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled device type %s", PetscDeviceTypes[dtype]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
struct Collector {
  using vec_type = std::vector<const ManagedMemory<T> *>;

  void operator()(const ManagedMemory<T> &expr) noexcept { this->list.emplace_back(std::addressof(expr)); }

  template <typename... U>
  void operator()(const Expression<U...> &expr) noexcept
  {
    util::tuple_for_each(expr.expr(), *this);
  }

  template <typename U>
  util::enable_if_t<!std::is_same<T, U>::value> operator()(const ManagedMemory<U> &expr) noexcept
  {
    this->list.emplace_back((const ManagedMemory<T> *)std::addressof(expr));
  }

  PETSC_CONSTEXPR_14 void operator()(...) const noexcept { }

  static Collector MakeCollector() noexcept
  {
    static vec_type slist{};

    slist.clear();
    return {slist};
  }

  vec_type &get_unique() noexcept
  {
    PetscCallCXXAbort(PETSC_COMM_SELF, {
      std::sort(this->list.begin(), list.end());
      this->list.erase(std::unique(this->list.begin(), this->list.end()), this->list.end());
    });
    return list;
  }

  vec_type &list;
};

template <typename Expr, typename T>
inline PetscErrorCode Dispatch(PetscDeviceContext dctx, PetscDeviceType dtype, PetscMemType mtype, Expr &&expr, ManagedMemory<T> &dest)
{
  using value_type          = typename ManagedMemory<T>::value_type;
  using size_type           = typename ManagedMemory<T>::size_type;
  const size_type size      = dest.size();
  const auto      dest_ptr  = std::addressof(dest);
  auto            dest_mode = PETSC_MEMORY_ACCESS_WRITE;
  value_type     *array;

  PetscFunctionBegin;
  auto &ptrs = util::tuple_for_each(expr.expr(), Collector<T>::MakeCollector()).get_unique();

  for (auto &&man : ptrs) {
    const value_type *dummy;

    if (man == dest_ptr) {
      dest_mode = PETSC_MEMORY_ACCESS_READ_WRITE;
    } else {
      PetscCall(man->GetArrayRead(dctx, mtype, PETSC_FALSE, &dummy));
    }
  }
  PetscCall(dest.GetArray(dctx, mtype, dest_mode, PETSC_FALSE, &array));
  {
    auto ptr_expr = expr.Freeze(mtype);

    switch (dtype) {
    case PETSC_DEVICE_HOST:
      for (size_type i = 0; i < size; ++i) array[i] = ptr_expr[i];
      break;
    case PETSC_DEVICE_CUDA:
    case PETSC_DEVICE_HIP:
      // clang-format off
      PetscCall(
        CUPMExec(
          dctx, dtype, size,
          [array, ptr_expr] PETSC_DEVICE_DECL(size_type i) { array[i] = ptr_expr[i]; }
        )
      );
      // clang-format on
    default:
      break;
    }
  }
  for (auto &&man : ptrs) {
    const value_type *dummy;

    if (man != dest_ptr) PetscCall(man->RestoreArrayRead(dctx, mtype, PETSC_FALSE, &dummy));
  }
  PetscCall(dest.RestoreArray(dctx, mtype, dest_mode, PETSC_FALSE, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode Dispatch(PetscDeviceContext dctx, PetscDeviceType dtype, PetscMemType mtype, const Expression<operators::identity, const ManagedMemory<T> &> &expr, ManagedMemory<T> &dest)
{
  using ptr_type       = typename ManagedMemory<T>::value_type;
  const auto      size = dest.size();
  auto          &&src  = std::get<0>(expr.expr());
  const ptr_type *src_ptr;
  ptr_type       *dest_ptr;

  PetscFunctionBegin;
  if (std::addressof(src) == std::addressof(dest)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(src.GetArrayRead(dctx, mtype, PETSC_FALSE, &src_ptr));
  PetscCall(dest.GetArray(dctx, mtype, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &dest_ptr));
  if (dtype == PETSC_DEVICE_HOST) {
    PetscCall(PetscArraycpy(dest_ptr, src_ptr, size));
  } else {
    void *stream = nullptr;

    PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
    switch (dtype) {
#if PetscDefined(HAVE_CUDA)
    case PETSC_DEVICE_CUDA:
      using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::CUDA>;

      PetscCall(CUPMInterface::PetscCUPMMemcpyAsync(dest_ptr, src_ptr, size, CUPMInterface::cupmMemcpyDeviceToDevice, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), true));
      break;
#endif
#if PetscDefined(HAVE_HIP)
    case PETSC_DEVICE_HIP:
      using CUPMInterface = Petsc::device::cupm::impl::Interface<Petsc::device::cupm::DeviceType::HIP>;

      PetscCall(CUPMInterface::PetscCUPMMemcpyAsync(dest_ptr, src_ptr, size, CUPMInterface::cupmMemcpyDeviceToDevice, *static_cast<typename CUPMInterface::cupmStream_t *>(stream), true));
      break;
#endif
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unhandled device type %s", PetscDeviceTypes[dtype]);
      break;
    }
  }
  PetscCall(src.RestoreArrayRead(dctx, mtype, PETSC_FALSE, &src_ptr));
  PetscCall(dest.RestoreArray(dctx, mtype, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &dest_ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace detail

template <typename T>
template <typename V>
inline PetscErrorCode ExecutableExpression<T>::Execute(ManagedMemory<V> &dest, PetscDeviceContext dctx) const noexcept
{
  PetscMemType    mtype = PETSC_MEMTYPE_DEVICE;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  switch (dtype) {
  case PETSC_DEVICE_HOST:
    mtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_DEVICE_CUDA:
  case PETSC_DEVICE_HIP:
  case PETSC_DEVICE_SYCL:
    mtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_DEVICE_MAX:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid device type %d", static_cast<int>(dtype));
  }
  PetscCall(detail::Dispatch(dctx, dtype, mtype, this->expr(), dest));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename V>
inline PetscErrorCode ExecutableExpression<T>::Execute(ManagedMemory<V> &dest) const noexcept
{
  PetscFunctionBegin;
  PetscCall(this->Execute(dest, this->dctx()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
PETSC_NODISCARD inline auto ident(T &&expr) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(MakeExpression<operators::identity>(std::forward<T>(expr)))

template <typename T, typename... E>
PETSC_NODISCARD inline std::pair<ManagedMemory<T> &, Expression<E...>> make_expr_pair(ManagedMemory<T> &dest, Expression<E...> &&expr) noexcept
{
  return {dest, std::move(expr)};
}

template <typename T, typename U = T>
PETSC_NODISCARD inline std::pair<ManagedMemory<T> &, Expression<operators::identity, const ManagedMemory<U> &>> make_expr_pair(ManagedMemory<T> &dest, const ManagedMemory<U> &src) noexcept
{
  return {dest, ident(src)};
}

} // namespace expr

// Force evaluation of the expression, but don't allow ManagedMemory (which is handled with a
// special overload)
template <typename T>
PETSC_NODISCARD inline util::enable_if_t<!util::is_instance<util::decay_t<T>, ManagedMemory>::value, expr::ExecutableExpression<util::remove_cvref_t<T>>> Eval(T &&expr, PetscDeviceContext dctx = nullptr) noexcept
{
  static_assert(expr::is_expression<T>::value, "");
  return {std::forward<T>(expr), dctx};
}

// wrap the managed type in a identity "expression" and pass it to the primary eval template
template <typename... T>
PETSC_NODISCARD inline auto Eval(const ManagedMemory<T...> &expr, PetscDeviceContext dctx = nullptr) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(Eval(expr::ident(expr), dctx))

namespace detail
{

template <typename T>
constexpr inline expr::detail::Collector<T> &MultiEvalCollect(expr::detail::Collector<T> &collecter) noexcept
{
  return collecter;
}

template <typename T, typename Top, typename... Rest>
PETSC_NODISCARD inline expr::detail::Collector<T> &MultiEvalCollect(expr::detail::Collector<T> &collector, Top &&top, Rest &&...rest) noexcept
{
  return MultiEvalCollect(util::tuple_for_each(top.second.expr(), collector), std::forward<Rest>(rest)...);
}

} // namespace detail

template <typename T, typename Expr, typename... ManagedMemoryExprPair>
inline PetscErrorCode MultiEval(PetscDeviceContext dctx, std::pair<ManagedMemory<T> &, Expr> expr1, ManagedMemoryExprPair &&...expr_n) noexcept
{
  using size_type              = typename ManagedMemory<T>::size_type;
  const size_type size         = expr1.first.size();
  const auto      out_man_ptrs = util::make_array(std::addressof(expr1.first), std::addressof(expr_n.first)...);
  auto            collector    = expr::detail::Collector<T>::MakeCollector();
  PetscMemType    mtype        = PETSC_MEMTYPE_DEVICE;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  switch (dtype) {
  case PETSC_DEVICE_HOST:
    mtype = PETSC_MEMTYPE_HOST;
    break;
  case PETSC_DEVICE_CUDA:
  case PETSC_DEVICE_HIP:
  case PETSC_DEVICE_SYCL:
    mtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_DEVICE_MAX:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Invalid device type %d", static_cast<int>(dtype));
  }

  auto &ptrs = detail::MultiEvalCollect(collector, expr1, expr_n...).get_unique();

  std::array<PetscMemoryAccessMode, out_man_ptrs.size()> out_modes;

  out_modes.fill(PETSC_MEMORY_ACCESS_WRITE);
  // Need to remove any instances of the output pointers from the list of input pointers. This
  // saves us calling GetArray() (and by extension marking them) twice.
  // clang-format off
  PetscCallCXX(
    const auto out_begin = out_man_ptrs.cbegin();
    const auto out_end   = out_man_ptrs.cend();

    ptrs.erase(
      std::remove_if(
        ptrs.begin(), ptrs.end(),
        [&] (const ManagedMemory<T> *ptr) {
          const auto out_it = std::find(out_begin, out_end, ptr);

          if (out_it == out_end) {
            // pointer found not in the output, is a pure read-only operation for it
            const T *array;

            PetscCallAbort(PETSC_COMM_SELF, ptr->GetArrayRead(dctx, mtype, PETSC_FALSE, &array));
            return false;
          }
          // pointer found in both the input and output, read-write
          out_modes[std::distance(out_begin, out_it)] = PETSC_MEMORY_ACCESS_READ_WRITE;
          return true;
        }
      ),
      ptrs.end()
    )
  );
  // clang-format on

  std::array<T *, out_man_ptrs.size()> out_ptrs;

  for (std::size_t i = 0; i < out_man_ptrs.size(); ++i) PetscCall(out_man_ptrs[i]->GetArray(dctx, mtype, out_modes[i], PETSC_FALSE, &out_ptrs[i]));
  {
    auto ptr_exprs = std::make_tuple(expr1.second.Freeze(mtype), expr_n.second.Freeze(mtype)...);

    switch (dtype) {
    case PETSC_DEVICE_HOST:
      for (size_type i = 0; i < size; ++i) {
        std::size_t idx = 0;

        util::tuple_for_each(ptr_exprs, [&](auto &&expr) { out_ptrs[idx++][i] = expr[i]; });
      }
      break;
    case PETSC_DEVICE_CUDA:
    case PETSC_DEVICE_HIP:
      // clang-format off
      PetscCall(
        expr::detail::CUPMExec(
          dctx, dtype, size,
          [out_ptrs, ptr_exprs] PETSC_DEVICE_DECL(size_type i) {
            std::size_t idx = 0;

            util::tuple_for_each(ptr_exprs, [&](auto &&expr) { out_ptrs[idx++][i] = expr[i]; });
          }
        )
      );
      // clang-format on
    default:
      break;
    }
  }

  for (auto &&ptr : ptrs) {
    const T *array;

    PetscCall(ptr->RestoreArrayRead(dctx, mtype, PETSC_FALSE, &array));
  }
  for (std::size_t i = 0; i < out_man_ptrs.size(); ++i) PetscCall(out_man_ptrs[i]->RestoreArray(dctx, mtype, out_modes[i], PETSC_FALSE, &out_ptrs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace Petsc

#endif // PETSC_CPP_EXPR_EXECUTABLE_HPP
