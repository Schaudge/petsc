static const char help[] = "Tests PetscManagedScalar/Real/IntSetValues().\n\n";

#include "petscmanagedtypetestcommon.hpp"

// TODO:
// test not just reading/writing from the same side
// test explicitly setting values on device as well...
template <typename... T>
class TestManagedTypeSetValues : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);
  PetscRandom rand_;

  // called by the other routines
  PetscErrorCode TestSetValues(PetscDeviceContext dctx, PetscManagedType scal) const noexcept {
    const auto op_before = [this](PetscDeviceContext dctx, PetscMemType mtype, PetscInt n, PetscType *ptr) {
      PetscFunctionBegin;
      if (PetscMemTypeHost(mtype)) {
        for (PetscInt i = 0; i < n; ++i) {
          PetscReal val;

          PetscCall(PetscRandomGetValueReal(rand_, &val));
          ptr[i] = static_cast<PetscType>(val);
        }
      } else {
        PetscCall(PetscDeviceArrayZero(dctx, mtype, ptr, n));
      }
      PetscFunctionReturn(0);
    };
    const auto op_after = [this](PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscInt n, PetscType *ptr) {
      PetscType *scalptr;

      PetscFunctionBegin;
      PetscCall(PetscManagedTypeGetValues(dctx, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &scalptr));
      for (PetscInt i = 0; i < n; ++i) {
        const PetscType actual   = scalptr[i];
        const PetscType expected = PetscMemTypeHost(mtype) ? ptr[i] : 0;

        PetscCheck(actual == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Set value scal[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g, memtype %s", i, (double)PetscRealPart(static_cast<PetscScalar>(actual)), i, (double)PetscRealPart(static_cast<PetscScalar>(expected)), PetscMemTypes(mtype));
      }
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(this->TestSetValuesAndOp(dctx, scal, std::move(op_before), std::move(op_after)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSetValuesSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestSetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSetValuesGroup(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestSetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpGroup(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

public:
  PetscErrorCode run(PetscRandom rand, PetscDeviceContext dctx) noexcept {
    PetscFunctionBegin;
    rand_ = rand;
    PetscCall(TestSetValuesSingleton(dctx));
    PetscCall(TestSetValuesGroup(dctx));
    PetscFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {
  auto              &comm = PETSC_COMM_WORLD;
  PetscDeviceContext dctx;
  PetscRandom        rand;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -100.0, 100.0));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(make_managed_scalar_test<TestManagedTypeSetValues>()->run(rand, dctx));
  PetscCall(make_managed_real_test<TestManagedTypeSetValues>()->run(rand, dctx));
  PetscCall(make_managed_int_test<TestManagedTypeSetValues>()->run(rand, dctx));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
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
