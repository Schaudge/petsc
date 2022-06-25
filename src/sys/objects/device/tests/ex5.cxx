static const char help[] = "Tests creation and destruction of PetscManagedScalar/Real/Int.\n\n";

#include "petscmanagedtypetestcommon.hpp"

template <typename... T>
class TestManagedTypeCreate : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);

  // need to test getvalues in creation/destruction since the act of getting the values
  // actually allocates the respective memory pools
  PetscErrorCode TestGetValues(PetscDeviceContext dctx, PetscManagedType scal) const noexcept {
    const auto op = [&](PetscDeviceContext, PetscManagedType, PetscMemType mtype, PetscMemoryAccessMode, PetscBool, PetscType *ptr, PetscDeviceType) {
      PetscFunctionBegin;
      if (PetscMemTypeHost(mtype)) PetscValidPointer(ptr, 6);
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(this->TestGetValuesAndOp(dctx, scal, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingletonDefault(PetscDeviceContext dctx, PetscInt nmax = 4) const noexcept {
    PetscManagedType scal;

    PetscFunctionBegin;
    // single size
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx, 1, &scal));
      PetscCall(TestGetValues(dctx, scal));
      PetscCall(PetscManagedTypeDestroy(dctx, &scal));
    }

    // single large size
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx, 50, &scal));
      PetscCall(TestGetValues(dctx, scal));
      PetscCall(PetscManagedTypeDestroy(dctx, &scal));
    }

    // different sizes
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx, i, &scal));
      if (i) PetscCall(TestGetValues(dctx, scal));
      PetscCall(PetscManagedTypeDestroy(dctx, &scal));
    }

    // different large sizes
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx, 100 * i, &scal));
      if (i) PetscCall(TestGetValues(dctx, scal));
      PetscCall(PetscManagedTypeDestroy(dctx, &scal));
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGroupDefault(PetscDeviceContext dctx) const noexcept {
    const auto op = [&](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpGroup(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [&](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

public:
  PetscErrorCode run(PetscDeviceContext dctx) const noexcept {
    PetscFunctionBegin;
    PetscCall(TestSingletonDefault(dctx));
    PetscCall(TestGroupDefault(dctx));
    PetscCall(TestSingleton(dctx));
    PetscFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(make_managed_scalar_test<TestManagedTypeCreate>()->run(dctx));
  PetscCall(make_managed_real_test<TestManagedTypeCreate>()->run(dctx));
  PetscCall(make_managed_int_test<TestManagedTypeCreate>()->run(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 testset:
   output_file: ./output/ExitSuccess.out
   filter: grep -v DEBUG_OUTPUT
   args: -device_enable {{lazy eager}} \
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
