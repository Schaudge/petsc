static const char help[] = "Tests PetscManagedScalar/Real/IntShiftPointer().\n\n";

#include "petscmanagedtypetestcommon.hpp"

template <typename... T>
class TestManagedTypeShiftPointer : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);
  PetscRandom rand_;

  // called by the other routines
  PetscErrorCode TestShift(PetscDeviceContext dctx, PetscManagedType scal) const noexcept {
    PetscInt        n, diff, shift = 0;
    PetscType      *host = nullptr, *device = nullptr;
    PetscDeviceType dtype;

    const auto setup = [&](PetscDeviceContext, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode, PetscBool, PetscType *, PetscDeviceType) {
      PetscReal val;

      PetscFunctionBegin;
      PetscCall(PetscRandomGetValueReal(rand_, &val));
      diff = static_cast<PetscInt>(val);
      if (shift + diff > n) diff = -diff;
      while (shift + diff < 0) ++diff;
      PetscCheck(shift + diff <= n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Test ill-formed, shift %" PetscInt_FMT " + diff %" PetscInt_FMT " > n %" PetscInt_FMT, shift, diff, n);
      DebugPrintf(PETSC_COMM_WORLD, "Shift %" PetscInt_FMT " Diff %" PetscInt_FMT " n %" PetscInt_FMT "\n", shift, diff, n);
      PetscCall(PetscManagedTypeShiftPointer(scal, diff));
      shift += diff;
      PetscFunctionReturn(0);
    };

    const auto op = [&](PetscDeviceContext, PetscManagedType, PetscMemType mtype, PetscMemoryAccessMode, PetscBool, PetscType *ptr, PetscDeviceType) {
      const auto check = [&](PetscType *expected_ptr, const char name[]) {
        const auto expected = std::next(expected_ptr, shift);

        PetscFunctionBegin;
        PetscCheck(ptr == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s pointer ptr %p != expected ptr %p for shift of %" PetscInt_FMT, name, ptr, expected, shift);
        PetscFunctionReturn(0);
      };

      PetscFunctionBegin;
      if (PetscMemTypeHost(mtype)) PetscCall(check(host, "Host"));
      else PetscCall(check(device, "Device"));
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCall(PetscManagedTypeGetSize(scal, &n));
    PetscCall(PetscManagedTypeResetShift(scal));
    PetscCall(PetscManagedTypeGetArray(dctx, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &host));
    if (dtype != PETSC_DEVICE_HOST) PetscCall(PetscManagedTypeGetArray(dctx, scal, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ, PETSC_FALSE, &device));
    PetscCall(PetscRandomSetInterval(rand_, 0, n));
    PetscCall(this->TestGetValuesAndOp(dctx, scal, std::move(op), std::move(setup)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestShift(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGroup(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestShift(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpGroup(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

public:
  PetscErrorCode run(PetscDeviceContext dctx, PetscRandom rand) noexcept {
    PetscFunctionBegin;
    rand_ = rand;
    PetscCall(TestSingleton(dctx));
    PetscCall(TestGroup(dctx));
    PetscFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {
  auto              &comm = PETSC_COMM_WORLD;
  PetscDeviceContext dctx;
  PetscRandom        rand;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(make_managed_scalar_test<TestManagedTypeShiftPointer>()->run(dctx, rand));
  PetscCall(make_managed_real_test<TestManagedTypeShiftPointer>()->run(dctx, rand));
  PetscCall(make_managed_int_test<TestManagedTypeShiftPointer>()->run(dctx, rand));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
 build:
   # TODO
   requires: todo

 testset:
   # TODO switch to getsubrange
   output_file: ./output/ExitSuccess.out
   filter: grep -v DEBUG_OUTPUT
   args: -device_enable {{lazy eager}}                                                         \
         -root_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -default_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -default_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -default_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -default_device_type sycl
     suffix: sycl

TEST*/
