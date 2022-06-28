static const char help[] = "Tests PetscManagedScalar/Real/IntCopy().\n\n";

#include "petscmanagedtypetestcommon.hpp"

template <typename... T>
class TestManagedTypeCopy : ManagedTypeInterface<T...> {
  PETSC_MANAGED_TYPE_INTERFACE_HEADER(T...);
  PetscRandom rand_;

  // called by the other routines
  PetscErrorCode TestCopy(PetscDeviceContext dctx, PetscManagedType reference) const noexcept {
    PetscType      *refcpy, *refarr;
    PetscInt        n;
    PetscDeviceType dtype;

    const auto op = [&](PetscDeviceContext dctx, PetscManagedType copy) {
      PetscMemType     destloc;
      PetscOffloadMask refbefore;

      PetscFunctionBegin;
      for (PetscInt i = 0; i < n; ++i) {
        // can access the naked host array since we know it has not (read: should not) change
        const auto ref    = static_cast<PetscScalar>(refcpy[i]);
        const auto actual = static_cast<PetscScalar>(reference->host[i]);

        // the values can be nan if they come from unitialized memory
        PetscCheck((ref == actual) || (PetscIsInfOrNanScalar(ref) && PetscIsInfOrNanScalar(actual)), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reference array has changed, expected ref[%" PetscInt_FMT "] %g != actual[%" PetscInt_FMT "] %g", i, (double)PetscAbsScalar(ref), i, (double)PetscAbsScalar(actual));
      }
      if (dtype == PETSC_DEVICE_HOST) {
        destloc = PETSC_MEMTYPE_HOST;
      } else {
        PetscReal r;

        PetscCall(PetscRandomGetValueReal(rand_, &r));
        destloc = (static_cast<int>(r) % 2) ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
      }
      PetscCall(DebugPrintf(PetscObjectComm(reinterpret_cast<PetscObject>(rand_)), "Location of reference array %s\n", PetscMemTypes(destloc)));
      // move the reference array to some random location
      PetscCall(PetscManagedTypeGetValues(dctx, reference, destloc, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_FALSE, &refarr));
      refbefore = reference->mask;
      // do the copy
      PetscCall(PetscManagedTypeCopy(dctx, copy, reference));
      PetscCheck(reference->mask == refbefore, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Copying should not change the offloadmask of the source, went from %s to %s", PetscOffloadMasks(refbefore), PetscOffloadMasks(reference->mask));
      PetscFunctionReturn(0);
    };

    PetscFunctionBegin;
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    // test self-copy is OK
    PetscCall(PetscManagedTypeCopy(dctx, reference, reference));
    PetscCall(PetscManagedTypeGetSize(reference, &n));
    // set up the real test
    PetscCall(PetscManagedTypeGetValues(dctx, reference, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE, &refarr));
    // zero out the memory
    PetscCall(PetscDeviceArrayZero(dctx, PETSC_MEMTYPE_HOST, refarr, n));
    PetscCall(PetscDeviceCalloc(dctx, PETSC_MEMTYPE_HOST, n, &refcpy));
    PetscCall(this->TestCreateAndOpGroup(dctx, op, n));
    PetscCall(PetscDeviceFree(dctx, refcpy));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingleton(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestCopy(dctx, scal); };

    PetscFunctionBegin;
    PetscCall(this->TestCreateAndOpSingleton(dctx, std::move(op)));
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestGroup(PetscDeviceContext dctx) const noexcept {
    const auto op = [this](PetscDeviceContext dctx, PetscManagedType scal) { return TestCopy(dctx, scal); };

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
  // we want to ensure memcopies can still be in flight
  PetscCall(PetscDeviceContextSetOption(dctx, PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS, PETSC_TRUE));

  PetscCall(make_managed_scalar_test<TestManagedTypeCopy>()->run(dctx, rand));
  PetscCall(make_managed_real_test<TestManagedTypeCopy>()->run(dctx, rand));
  PetscCall(make_managed_int_test<TestManagedTypeCopy>()->run(dctx, rand));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscPrintf(comm, "EXIT_SUCCESS\n"));
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
