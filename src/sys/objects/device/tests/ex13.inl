static const char help[] = "Tests PetscManagedTypeEnsureOffload().\n\n";

#include <petsc/private/deviceimpl.h>
#include <petscdevice_cupm.h>

// kernels only valid for a single device thread!
static PETSC_KERNEL_DECL void check(const PetscInt *values, PetscInt size, PetscInt *errors_found)
{
  PetscInt errors = 0;

  for (PetscInt i = 0; i < size; ++i) {
    if (values[i] != i) {
      ++errors;
      printf("PETSC_ERROR: device_values[%" PetscInt_FMT "] %" PetscInt_FMT " != expected[%" PetscInt_FMT "] %" PetscInt_FMT "\n",i,values[i],i,i);
    }
  }
  errors_found[0] = errors;
  return;
}

static PETSC_KERNEL_DECL void set(PetscInt *values, PetscInt size)
{
  for (PetscInt i = 0; i < size; ++i) values[i] = size-i;
  return;
}

static PetscErrorCode ForceOffload(PetscDeviceContext dctx, PetscManagedInt scal, PetscMemType mtype)
{
  PetscInt *unused;

  PetscFunctionBegin;
  // this exists purely to force the managed type to consider mtype to be most up to date
  PetscCall(PetscManagedIntGetArray(dctx,scal,mtype,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&unused));
  PetscFunctionReturn(0);
}

// set on device, check on host
static PetscErrorCode CheckOffloadCPU(PetscDeviceContext dctx, PetscManagedInt scal, PetscInt *host_values, PetscInt *device_values)
{
  PetscInt     n;
  cupmStream_t strm;

  PetscFunctionBegin;
  // set
  PetscCall(PetscManagedIntGetSize(scal,&n));
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&strm));
  PetscCall(ForceOffload(dctx,scal,PETSC_MEMTYPE_DEVICE));
  set<<<1,1,0,strm>>>(device_values,n);

  // and check
  PetscCall(PetscManagedIntEnsureOffload(dctx,scal,PETSC_OFFLOAD_CPU,PETSC_TRUE));
  for (PetscInt i = 0; i < n; ++i) {
    const auto val = host_values[i];
    const auto exp = n-i;

    PetscCheck(val == exp,PETSC_COMM_SELF,PETSC_ERR_PLIB,"host_values[%" PetscInt_FMT "] %" PetscInt_FMT " != expected[%" PetscInt_FMT "] %" PetscInt_FMT,i,val,i,exp);
  }
  PetscFunctionReturn(0);
}

// set on host, check on device
static PetscErrorCode CheckOffloadGPU(PetscDeviceContext dctx, PetscManagedInt scal, PetscInt *host_values, PetscInt *device_values)
{
  PetscInt        *err_count;
  PetscInt         n;
  cupmStream_t     strm;
  PetscManagedInt  errors;

  PetscFunctionBegin;
  // set values directly through the host pointer
  PetscCall(PetscManagedIntGetSize(scal,&n));
  // make sure the managed type knows about this as well
  PetscCall(ForceOffload(dctx,scal,PETSC_MEMTYPE_HOST));
  for (PetscInt i = 0; i < n; ++i) host_values[i] = i;

  // and check
  PetscCall(PetscManagedIntCreateDefault(dctx,1,&errors));
  PetscCall(PetscManagedIntGetArray(dctx,errors,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&err_count));
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&strm));
  // the last thing we do is copy the values to GPU
  PetscCall(PetscManagedIntEnsureOffload(dctx,scal,PETSC_OFFLOAD_GPU,PETSC_FALSE));
  check<<<1,1,0,strm>>>(device_values,n,err_count);
  PetscCall(PetscManagedIntGetArray(dctx,errors,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&err_count));
  PetscCheck(!*err_count,PETSC_COMM_SELF,PETSC_ERR_PLIB,"%" PetscInt_FMT " errors in PetscManagedIntEnsureOffload(PETSC_OFFLOAD_GPU)!",*err_count);
  PetscCall(PetscManagedIntDestroy(dctx,&errors));
  PetscFunctionReturn(0);
}

template <typename T>
static PetscErrorCode Check(PetscDeviceContext dctx, PetscInt n, T&& CheckOffload)
{
  PetscInt        *host,*device;
  PetscManagedInt  scal;

  PetscFunctionBegin;
  // create a host array that is co-opted by the managed type
  PetscCall(PetscDeviceMalloc(dctx,PETSC_MEMTYPE_HOST,n,&host));
  // create a device array that is co-opted by the managed type
  PetscCall(PetscDeviceMalloc(dctx,PETSC_MEMTYPE_DEVICE,n,&device));
  PetscCall(PetscManagedIntCreate(dctx,host,device,n,PETSC_USE_POINTER,PETSC_USE_POINTER,PETSC_OFFLOAD_CPU,&scal));
  PetscCall(CheckOffload(dctx,scal,host,device));
  PetscCall(PetscManagedIntDestroy(dctx,&scal));
  PetscCall(PetscDeviceFree(dctx,device));
  PetscCall(PetscDeviceFree(dctx,host));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  auto&              comm = PETSC_COMM_WORLD;
  PetscDeviceContext dctx;
  PetscInt           n    = 50;

  PetscCall(PetscInitialize(&argc,&argv,nullptr,help));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscOptionsBegin(comm,nullptr,"Test Options",nullptr);
  PetscCall(PetscOptionsInt("-n","Size of managed scalars. Should be large enough to hide host to device memcopies",nullptr,n,&n,nullptr));
  PetscOptionsEnd();

  PetscCall(Check(dctx,n,CheckOffloadCPU));
  PetscCall(Check(dctx,n,CheckOffloadGPU));

  PetscCall(PetscPrintf(comm,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}
