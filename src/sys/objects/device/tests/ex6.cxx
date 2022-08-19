static const char help[] = "Tests PetscManagedScalar/Real/IntGetValues().\n\n";

#include "petscmanagedtypetestcommon.hpp"

// REVIEW ME: TODO
// 1. check values on the device directly
// 2. test partially writing then reading (a la VecSeqCUPM::mdot_async(), when cublas is used)
template <typename... T>
class TestManagedTypeGetValues : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);

  // called by the other routines
  PetscErrorCode TestGetValues(PetscDeviceContext dctx, PetscManagedType scal) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, PetscType *ptr, PetscDeviceType dtype) {
      PetscFunctionBegin;
      if (PetscMemTypeHost(mtype)) {
        static auto have_written = false;
        PetscInt    n;

        PetscValidPointer(ptr, 6);
        PetscCall(PetscManagedTypeGetSize(scal, &n));
        // use this loop to write
        if (PetscMemoryAccessWrite(mode)) {
          have_written = true;
          for (PetscInt i = 0; i < n; ++i) ptr[i] = static_cast<PetscType>(i);
          // ensure the values are piped down to the device if we have one
          if (dtype != PETSC_DEVICE_HOST) { PetscCall(PetscManagedTypeGetArray(dctx, scal, PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_FALSE, &ptr)); }
        }
        // now check what we've written is coherent
        if (PetscMemoryAccessRead(mode)) {
          PetscCheck(have_written, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trying to read from pointer before writing to it, this test is ill-formed!");
          PetscCall(PetscManagedTypeGetArray(dctx, scal, PETSC_MEMTYPE_HOST, mode, PETSC_TRUE, &ptr));
          for (PetscInt i = 0; i < n; ++i) {
            PetscCheck(ptr[i] == static_cast<PetscType>(i), PETSC_COMM_SELF, PETSC_ERR_PLIB, "ptr[%" PetscInt_FMT "] %g != %g", i, static_cast<double>(std::abs(static_cast<PetscScalar>(ptr[i]))), static_cast<double>(i));
          }
        }
      }
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(this->TestGetValuesAndOp(dctx, scal, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGetValuesSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGetValuesGroup(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetValues(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpGroup(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

public:
  PetscErrorCode run(PetscDeviceContext dctx) const noexcept {
    PetscFunctionBegin;
    PetscCall(TestGetValuesSingleton(dctx));
    PetscCall(TestGetValuesGroup(dctx));
    PetscFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(make_managed_scalar_test<TestManagedTypeGetValues>()->run(dctx));
  PetscCall(make_managed_real_test<TestManagedTypeGetValues>()->run(dctx));
  PetscCall(make_managed_int_test<TestManagedTypeGetValues>()->run(dctx));

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
