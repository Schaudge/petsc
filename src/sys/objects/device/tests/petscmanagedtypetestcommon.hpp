#ifndef PETSCMANAGEDTYPETESTCOMMON_HPP
#define PETSCMANAGEDTYPETESTCOMMON_HPP

#include "petscdevicetestcommon.h"

#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/array.hpp>
#include <typeinfo>
#include <sstream>
#include <array>
#include <memory>

struct NoOp {
  constexpr PetscErrorCode operator()(...) const noexcept { return 0; }
};

template <typename PetscType_, typename PetscManagedType_, typename CreateT, typename CreateDefaultT, typename DestroyT, typename GetValuesT, typename SetValuesT, typename GetPointerAndMemTypeT, typename CopyT, typename GetSizeT>
struct ManagedTypeInterface {
  using PetscType        = PetscType_;
  using PetscManagedType = PetscManagedType_;

  PetscBool                   _logging = PETSC_FALSE;
  const CreateT               PetscManagedTypeCreate;
  const CreateDefaultT        PetscManagedTypeCreateDefault;
  const DestroyT              PetscManagedTypeDestroy;
  const GetValuesT            PetscManagedTypeGetValues;
  const SetValuesT            PetscManagedTypeSetValues;
  const GetPointerAndMemTypeT PetscManagedTypeGetPointerAndMemType;
  const CopyT                 PetscManagedTypeCopy;
  const GetSizeT              PetscManagedTypeGetSize;

  ManagedTypeInterface(CreateT &&c, CreateDefaultT &&cd, DestroyT &&d, GetValuesT &&gv, SetValuesT &&sv, GetPointerAndMemTypeT &&gpamt, CopyT &&cp, GetSizeT &&gs) :
    PetscManagedTypeCreate(std::forward<CreateT>(c)), PetscManagedTypeCreateDefault(std::forward<CreateDefaultT>(cd)), PetscManagedTypeDestroy(std::forward<DestroyT>(d)), PetscManagedTypeGetValues(std::forward<GetValuesT>(gv)), PetscManagedTypeSetValues(std::forward<SetValuesT>(sv)), PetscManagedTypeGetPointerAndMemType(std::forward<GetPointerAndMemTypeT>(gpamt)), PetscManagedTypeCopy(std::forward<CopyT>(cp)), PetscManagedTypeGetSize(std::forward<GetSizeT>(gs)) {
    PetscCallAbort(PETSC_COMM_WORLD, [&] {
      PetscBool set;

      PetscFunctionBegin;
      PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Common PetscManagedType test options", nullptr);
      PetscCall(PetscOptionsBool("-debug_logging", "Enable debug logging", nullptr, _logging, &_logging, &set));
      PetscOptionsEnd();
      _logging = static_cast<PetscBool>(_logging && set);
      PetscFunctionReturn(0);
    }());
  }

  template <typename... T>
  PetscErrorCode DebugPrintf(MPI_Comm comm, const char mess[], T &&...args) const {
    PetscFunctionBegin;
    if (_logging) {
      std::string s{"[DEBUG_OUTPUT:%d]: (PetscType = %s) "};
      char       *demangled;
      int         rank;

      s += mess;
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscDemangleSymbol(typeid(PetscType).name(), &demangled));
      PetscCall(PetscPrintf(comm, s.c_str(), rank, demangled, std::forward<T>(args)...));
      PetscCall(PetscFree(demangled));
    }
    PetscFunctionReturn(0);
  }

  template <typename T>
  PetscErrorCode TestCreateAndOpSingleton(PetscDeviceContext dctx, T &&TestOp, PetscInt size = 1, PetscInt nmax = 2) const noexcept {
    using namespace Petsc::util;

    constexpr auto   masks    = make_array(PETSC_OFFLOAD_CPU, PETSC_OFFLOAD_GPU);
    PetscType       *host_ptr = nullptr, *host_own_ptr = nullptr;
    PetscType       *device_ptr = nullptr, *device_own_ptr = nullptr;
    PetscManagedType scal;
    PetscDeviceType  dtype;

    PetscFunctionBegin;
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== begin %s()\n", PETSC_FUNCTION_NAME));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_HOST, size, &host_own_ptr));
    // single size
    auto host_ptrs = make_array(std::make_pair(host_own_ptr, PETSC_COPY_VALUES), std::make_pair(host_ptr, PETSC_OWN_POINTER), std::make_pair(host_own_ptr, PETSC_USE_POINTER));
    if (dtype != PETSC_DEVICE_HOST) { PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_DEVICE, size, &device_own_ptr)); }
    auto device_ptrs = make_array(std::make_pair(device_own_ptr, PETSC_COPY_VALUES), std::make_pair(device_ptr, dtype == PETSC_DEVICE_HOST ? PETSC_USE_POINTER : PETSC_OWN_POINTER), std::make_pair(device_own_ptr, PETSC_USE_POINTER));
    for (auto &host : host_ptrs) {
      const auto host_alloc = host.second == PETSC_OWN_POINTER;
      for (auto &device : device_ptrs) {
        const auto device_alloc = device.second == PETSC_OWN_POINTER;
        for (auto &&mask : masks) {
          for (PetscInt k = 0; k < nmax; ++k) {
            // need to keep reallocating the host pointer since we will pass over ownership
            if (host_alloc) PetscCall(PetscMalloc1(size, &host.first));
            if (device_alloc) PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, size, &device.first));
            PetscCall(DebugPrintf(PETSC_COMM_WORLD, "host ptr %p (alloced? %s) device ptr %p (alloced? %s) size %" PetscInt_FMT " host copy mode %s device copy mode %s PetscOffloadMask %s\n", host.first, PetscBools[host_alloc], device.first, PetscBools[device_alloc], size,
                                  PetscCopyModes[host.second], PetscCopyModes[device.second], PetscOffloadMasks(mask)));
            PetscCall(PetscManagedTypeCreate(dctx, host.first, device.first, size, host.second, device.second, mask, &scal));
            if (size) PetscCall(TestOp(dctx, scal));
            PetscCall(PetscManagedTypeDestroy(dctx, &scal));
          }
        }
      }
    }
    PetscCall(PetscDeviceFree(dctx, host_own_ptr));
    PetscCall(PetscDeviceFree(dctx, device_own_ptr));
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== end %s()\n", PETSC_FUNCTION_NAME));
    PetscFunctionReturn(0);
  }

  template <PetscInt n_scal = 4, typename T>
  PetscErrorCode TestCreateAndOpGroup(PetscDeviceContext dctx, T &&TestOp, PetscInt alloc_size = 1) const noexcept {
    static_assert(n_scal > 0, "");
    static_assert(n_scal % 2 == 0, "");
    PetscManagedType scal_arr[n_scal];

    PetscFunctionBegin;
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== begin %s()\n", PETSC_FUNCTION_NAME));
    // destroy in original order
    PetscCheck(alloc_size > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Allocation size %" PetscInt_FMT " must be > 0", alloc_size);
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(DebugPrintf(PETSC_COMM_WORLD, "original order: create i %" PetscInt_FMT "\n", i));
      PetscCall(PetscManagedTypeCreateDefault(dctx, alloc_size, scal_arr + i));
      PetscCall(TestOp(dctx, scal_arr[i]));
    }
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(DebugPrintf(PETSC_COMM_WORLD, "original order: destroy i %" PetscInt_FMT "\n", i));
      PetscCall(PetscManagedTypeDestroy(dctx, scal_arr + i));
    }

    // destroy in reverse order
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(DebugPrintf(PETSC_COMM_WORLD, "reverse order: create i %" PetscInt_FMT "\n", i));
      PetscCall(PetscManagedTypeCreateDefault(dctx, PetscMax(100 * i, alloc_size), scal_arr + i));
      PetscCall(TestOp(dctx, scal_arr[i]));
    }
    for (PetscInt i = n_scal - 1; i >= 0; --i) {
      PetscCall(DebugPrintf(PETSC_COMM_WORLD, "reverse order: destroy i %" PetscInt_FMT "\n", i));
      PetscCall(PetscManagedTypeDestroy(dctx, scal_arr + i));
    }

    // destroy as we create
    for (PetscInt i = 0, j = 0; i < n_scal + (n_scal / 2); ++i) {
      if (i < n_scal) {
        PetscCall(DebugPrintf(PETSC_COMM_WORLD, "create and destroy: create i %" PetscInt_FMT ", j %" PetscInt_FMT "\n", i, j));
        PetscCall(PetscManagedTypeCreateDefault(dctx, PetscMax(i, alloc_size), scal_arr + i));
        PetscCall(TestOp(dctx, scal_arr[i]));
      }
      if (i >= n_scal / 2) {
        PetscCall(DebugPrintf(PETSC_COMM_WORLD, "create and destroy: destroy i %" PetscInt_FMT ", j %" PetscInt_FMT "\n", i, j));
        PetscCall(PetscManagedTypeDestroy(dctx, scal_arr + j));
        ++j;
      }
    }
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== end %s()\n", PETSC_FUNCTION_NAME));
    PetscFunctionReturn(0);
  }

  template <typename T, typename U = NoOp>
  PetscErrorCode TestGetValuesAndOp(PetscDeviceContext dctx, PetscManagedType scal, T &&TestOp, U &&SetupOp = U{}) const noexcept {
    using namespace Petsc::util;

    constexpr auto  syncs  = make_array(PETSC_FALSE, PETSC_TRUE);
    constexpr auto  mtypes = make_array(PETSC_MEMTYPE_HOST, PETSC_MEMTYPE_DEVICE);
    constexpr auto  modes  = make_array(PETSC_MEMORY_ACCESS_WRITE, PETSC_MEMORY_ACCESS_READ, PETSC_MEMORY_ACCESS_READ_WRITE);
    PetscDeviceType dtype;

    PetscFunctionBegin;
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== begin %s()\n", PETSC_FUNCTION_NAME));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    for (auto &&sync : syncs) {
      for (auto &&mtype : mtypes) {
        // host device cannot handle device memory
        if (mtype == PETSC_MEMTYPE_DEVICE && dtype == PETSC_DEVICE_HOST) continue;
        for (auto &&mode : modes) {
          PetscType *ptr = nullptr;

          PetscCall(DebugPrintf(PETSC_COMM_WORLD, "PetscMemType %s PetscMemoryAccessMode %s sync %s\n", PetscMemTypes(mtype), PetscMemoryAccessModes(mode), PetscBools[sync]));
          PetscCall(SetupOp(dctx, scal, mtype, mode, sync, ptr, dtype));
          PetscCall(PetscManagedTypeGetValues(dctx, scal, mtype, mode, sync, &ptr));
          PetscCall(TestOp(dctx, scal, mtype, mode, sync, ptr, dtype));
        }
      }
    }
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== end %s()\n", PETSC_FUNCTION_NAME));
    PetscFunctionReturn(0);
  }

  template <typename T, typename U>
  PetscErrorCode TestSetValuesAndOp(PetscDeviceContext dctx, PetscManagedType scal, T &&TestOpSetUp, U &&TestOp) const noexcept {
    PetscDeviceType dtype;
    PetscInt        n;

    PetscFunctionBegin;
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== begin %s()\n", PETSC_FUNCTION_NAME));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCall(PetscManagedTypeGetSize(scal, &n));
    for (auto &&mtype_write : {PETSC_MEMTYPE_HOST, PETSC_MEMTYPE_DEVICE}) {
      // host device cannot handle device memory
      if (mtype_write == PETSC_MEMTYPE_DEVICE && dtype == PETSC_DEVICE_HOST) {
        PetscCall(DebugPrintf(PETSC_COMM_WORLD, "PetscMemType %s skipped (device of type %s cannot handle memtype %s)\n", PetscMemTypes(mtype_write), PetscDeviceTypes[dtype], PetscMemTypes(mtype_write)));
      } else {
        PetscType *ptr = nullptr;

        PetscCall(DebugPrintf(PETSC_COMM_WORLD, "PetscMemType %s\n", PetscMemTypes(mtype_write)));
        PetscCall(PetscDeviceMalloc(dctx, mtype_write, n, &ptr));
        // this should set the contents of ptr to some value
        PetscCall(TestOpSetUp(dctx, mtype_write, n, ptr));
        // now set it
        PetscCall(PetscManagedTypeSetValues(dctx, scal, mtype_write, ptr, n));
        // and check it
        PetscCall(TestOp(dctx, scal, mtype_write, n, ptr));
        PetscCall(PetscDeviceFree(dctx, ptr));
      }
    }
    PetscCall(DebugPrintf(PETSC_COMM_WORLD, "==== end %s()\n", PETSC_FUNCTION_NAME));
    PetscFunctionReturn(0);
  }
};

#define PETSC_MANAGED_TYPE_INTERFACE_HEADER(...) \
  using base_type = ManagedTypeInterface<__VA_ARGS__>; \
  using typename base_type::PetscManagedType; \
  using typename base_type::PetscType; \
  using base_type::base_type; \
  using base_type::DebugPrintf; \
  using base_type::PetscManagedTypeCreate; \
  using base_type::PetscManagedTypeCreateDefault; \
  using base_type::PetscManagedTypeDestroy; \
  using base_type::PetscManagedTypeGetValues; \
  using base_type::PetscManagedTypeSetValues; \
  using base_type::PetscManagedTypeGetPointerAndMemType; \
  using base_type::PetscManagedTypeCopy; \
  using base_type::PetscManagedTypeGetSize

template <typename PetscType, typename PetscManagedType, typename... FunctionTypes>
static auto make_managed_interface(FunctionTypes &&...fns) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(ManagedTypeInterface<PetscType, PetscManagedType, FunctionTypes...>{std::forward<FunctionTypes>(fns)...});

template <template <typename...> class T, typename PT, typename PMT, typename... Args>
static auto make_managed_test(Args &&...functions) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(std::unique_ptr<T<PT, PMT, Args...>>{new T<PT, PMT, Args...>{std::forward<Args>(functions)...}});

template <template <typename...> class T>
static auto make_managed_scalar_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(make_managed_test<T, PetscScalar, PetscManagedScalar>(PetscManagedScalarCreate, PetscManagedScalarCreateDefault, PetscManagedScalarDestroy, PetscManagedScalarGetValues, PetscManagedScalarSetValues, PetscManagedScalarGetPointerAndMemType, PetscManagedScalarCopy, PetscManagedScalarGetSize));

template <template <typename...> class T>
static auto make_managed_real_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(make_managed_test<T, PetscReal, PetscManagedReal>(PetscManagedRealCreate, PetscManagedRealCreateDefault, PetscManagedRealDestroy, PetscManagedRealGetValues, PetscManagedRealSetValues, PetscManagedRealGetPointerAndMemType, PetscManagedRealCopy, PetscManagedRealGetSize));

template <template <typename...> class T>
static auto make_managed_int_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(make_managed_test<T, PetscInt, PetscManagedInt>(PetscManagedIntCreate, PetscManagedIntCreateDefault, PetscManagedIntDestroy, PetscManagedIntGetValues, PetscManagedIntSetValues, PetscManagedIntGetPointerAndMemType, PetscManagedIntCopy, PetscManagedIntGetSize));
#endif // PETSCMANAGEDTYPETESTCOMMON_HPP
