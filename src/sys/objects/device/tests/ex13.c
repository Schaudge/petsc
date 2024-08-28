const char help[] = "Test PetscDeviceMemcmp()";

#include <petscdevice.h>

static PetscErrorCode TestMemcmp(PetscDeviceContext dctx, PetscMemType mtype1, PetscMemType mtype2)
{
  size_t n = 10;
  char *str1, *str2;
  PetscBool e;

  PetscFunctionBegin;
  PetscCall(PetscDeviceMalloc(dctx, mtype1, n, &str1));
  PetscCall(PetscDeviceMalloc(dctx, mtype2, n, &str2));
  PetscCall(PetscDeviceMemset(dctx, str1, (PetscInt) 'a', n));
  PetscCall(PetscDeviceMemset(dctx, str2, (PetscInt) 'a', n));
  PetscCall(PetscDeviceMemcmp(dctx, &str1[1], &str2[1], n-1, &e));
  PetscCheck(e == PETSC_TRUE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "equal strings compared inequal");
  PetscCall(PetscDeviceMemset(dctx, str2, (PetscInt) 'b', n));
  PetscCall(PetscDeviceMemcmp(dctx, &str1[1], &str2[1], n-1, &e));
  PetscCheck(e == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "inequal strings compared inequal");

  PetscCall(PetscDeviceFree(dctx, str2));
  PetscCall(PetscDeviceFree(dctx, str1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscDeviceContext dctx;
  PetscDevice        device;
  PetscDeviceType    type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDevice(dctx, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(TestMemcmp(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMTYPE_HOST));
  if (type != PETSC_DEVICE_HOST) {
    PetscCall(TestMemcmp(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMTYPE_DEVICE));
    PetscCall(TestMemcmp(dctx, PETSC_MEMTYPE_DEVICE, PETSC_MEMTYPE_HOST));
    PetscCall(TestMemcmp(dctx, PETSC_MEMTYPE_DEVICE, PETSC_MEMTYPE_DEVICE));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
