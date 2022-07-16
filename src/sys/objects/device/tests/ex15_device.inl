#include <petsc/private/deviceimpl.h>
#include <petscdevice_cupm.h>

static PETSC_KERNEL_DECL void wasteSomeTimeKernel(clock_t *global_now, PetscInt max_cycles)
{
  clock_t start = clock();
  clock_t now;
  for (;;) {
    now = clock();
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= max_cycles) break;
  }
  // Stored "now" in global memory here to prevent the
  // compiler from optimizing away the entire loop.
  *global_now = now;
}

static PetscErrorCode WasteSomeTime(PetscDeviceContext dctx, PetscInt ncycles, clock_t *global_now)
{
  PetscDeviceType dtype;
  cupmStream_t    stream;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetDeviceType(dctx,&dtype));
  if (dtype == PETSC_DEVICE_HOST) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  wasteSomeTimeKernel<<<1024,2,0,stream>>>(global_now,ncycles);
  PetscFunctionReturn(0);
}
