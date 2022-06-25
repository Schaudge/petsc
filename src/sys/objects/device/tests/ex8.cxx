static const char help[] = "Tests PetscManagedScalar/Real/IntGetPointerAndMemType().\n\n";

#include "petscmanagedtypetestcommon.hpp"

template <typename... T>
class TestManagedTypeGetPointerAndMemType : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);

  // called by the other routines
  PetscErrorCode TestGetPointerAndMemType(PetscDeviceContext dctx, PetscManagedType scal) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype_expect, PetscMemoryAccessMode mode, PetscBool, PetscType *ptr_expect, PetscDeviceType) {
      PetscType   *ptr;
      const auto   mask_before = scal->mask;
      PetscMemType mtype, mtype_explicit_check;

      PetscFunctionBegin;
      PetscCall(PetscManagedTypeGetPointerAndMemType(dctx, scal, mode, &ptr, &mtype));
      PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Returned pointer is NULL (%p)", ptr);
      PetscCheck(mtype == mtype_expect, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected memtype %d, got %d instead", mtype_expect, mtype);
      PetscCheck(ptr == ptr_expect, PETSC_COMM_SELF, PETSC_ERR_POINTER, "PetscManagedTypeGetPointerAndMemType returned different pointer than PetscMnagedTypeGetValues. Expected %p got %p", ptr_expect, ptr);
      PetscCheck(scal->mask == mask_before, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscOffloadMask changed during call, had %d before, now %d", mask_before, scal->mask);
      PetscCall(PetscGetMemType(ptr, &mtype_explicit_check));
      if (PetscMemTypeHost(mtype)) {
        PetscValidPointer(ptr, 2);
        PetscCheck(mtype == mtype_explicit_check, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected PETSC_MEMTYPE_HOST (%d), but PetscGetMemType() returned %d", PETSC_MEMTYPE_HOST, mtype_explicit_check);
      } else {
        PetscCheck(PetscMemTypeDevice(mtype_explicit_check), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected some kind of device PetscMemType but PetscGetMemType() returned %d (PetscMemTypeDevice(ptr) = %d)", mtype_explicit_check, PetscMemTypeDevice(mtype_explicit_check));
      }
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(this->TestGetValuesAndOp(dctx, scal, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetPointerAndMemType(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGroup(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestGetPointerAndMemType(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpGroup(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

public:
  PetscErrorCode run(PetscDeviceContext dctx) const noexcept {
    PetscFunctionBegin;
    PetscCall(TestSingleton(dctx));
    PetscCall(TestGroup(dctx));
    PetscFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(make_managed_scalar_test<TestManagedTypeGetPointerAndMemType>()->run(dctx));
  PetscCall(make_managed_real_test<TestManagedTypeGetPointerAndMemType>()->run(dctx));
  PetscCall(make_managed_int_test<TestManagedTypeGetPointerAndMemType>()->run(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 testset:
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
