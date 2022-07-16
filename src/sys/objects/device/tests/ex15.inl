static const char help[] = "Tests that the PetscManagedType allocator does not copy-construct memory chunks\n\n";

#include <petscdevice.h>
#include <time.h> // clock_t

int main(int argc, char *argv[])
{
  PetscManagedReal   *scal_arr;
  PetscDeviceContext  dctx[2];
  PetscDeviceType     dtype;
  const PetscInt      bucket_size = 256;
  PetscInt            n           = 524,size = bucket_size/2,ncycles = 10000,m;
  clock_t            *global_now;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Test Options",NULL);
  PetscCall(PetscOptionsInt("-n","Number of outer loops",NULL,n,&n,NULL));
  PetscCall(PetscOptionsRangeInt("-size","Size of managedtypes",NULL,size,&size,NULL,1,bucket_size));
  PetscCall(PetscOptionsInt("-cycles","Number of clock cycles in the 'waste time' kernel",NULL,ncycles,&ncycles,NULL));
  PetscOptionsEnd();

  m = bucket_size/size;
  PetscCall(PetscDeviceContextGetCurrentContext(dctx));
  // we absolutely want to orphan some of the transfers, this is the point of this test
  PetscCall(PetscDeviceContextSetOption(dctx[0],PETSC_DEVICE_CONTEXT_ALLOW_ORPHANS,PETSC_TRUE));
  PetscCall(PetscDeviceContextDuplicate(dctx[0],dctx+1));
  PetscCall(PetscDeviceContextGetDeviceType(dctx[0],&dtype));
  PetscCall(PetscDeviceMalloc(dctx[0],dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE,1,&global_now));
  PetscCall(PetscMalloc1(n*m,&scal_arr));
  // Internally the managedtype allocator relies on a std::deque<std::deque<MemoryChunk>>. The
  // MemoryChunks are not copyable and will error if is attempted. The goal here is to try and
  // trigger that error when the std::deque implementation allocates new buckets on
  // expansion. If it extends and copies, then the error will fire.
  //
  // By default std::deque buckets are the following sizes:
  //
  // listdc++: 8 times the object size on 64-bit
  // libc++: 16 times the object size or 4096 bytes, whichever is larger, on 64-bit
  //
  // since each deque holds pointers (usually size of 8), we need 512 to trigger a copy, but
  // for safety we do 1024
  for (PetscInt i = 0 ; i < n; ++i) {
    for (PetscInt j = 0; j < m; ++j) {
      const PetscInt  idx = i*m+j;
      PetscReal      *host;

      // do a create->instantiate->destroy over 2 different contexts, this will create "dead"
      // allocations that need to be returned on the stream (note its backwards because dctx 0
      // is the "root")
      for (PetscInt k = 1; k >= 0; --k) {
        PetscCall(PetscManagedRealCreateDefault(dctx[k],size,scal_arr+idx));
        PetscCall(PetscManagedRealGetValues(dctx[k],scal_arr[idx],PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&host));
        if (dtype != PETSC_DEVICE_HOST) {
          PetscReal *device;

          PetscCall(PetscManagedRealGetValues(dctx[k],scal_arr[idx],PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_FALSE,&device));
        }
        PetscCall(PetscManagedRealGetValues(dctx[k],scal_arr[idx],PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&host));
        if (k) {
          PetscCall(WasteSomeTime(dctx[k],ncycles,global_now));
          PetscCall(PetscManagedRealDestroy(dctx[k],scal_arr+idx));
        }
      }
      *host = idx;
    }
  }

  for (PetscInt i = 0; i < n*m; ++i) {
    PetscReal *host;

    PetscCall(PetscManagedRealGetValues(dctx[0],scal_arr[i],PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&host));
    // if we have copy-constructed or the flag variables have failed somehow this might also
    // fail
    PetscCheck((PetscInt)(*host) == i,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"actual[%" PetscInt_FMT "] %" PetscInt_FMT " != expected[%" PetscInt_FMT "] %" PetscInt_FMT ". This may indicate an error in the handling of the pools atomic marker flags, i.e. that blocks are being given back out before they are ready!",i,(PetscInt)(*host),i,i);
  }
  // do these in a separate loop since they may hide errors in the previous by taking up time
  for (PetscInt i = 0; i < n*m; ++i) PetscCall(PetscManagedRealDestroy(dctx[0],scal_arr+i));

  PetscCall(PetscDeviceFree(dctx[0],global_now));
  PetscCall(PetscDeviceContextDestroy(dctx+1));
  PetscCall(PetscFree(scal_arr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}
