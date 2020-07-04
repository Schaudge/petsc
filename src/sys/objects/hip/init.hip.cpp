/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot known
  what malloc is being used until it has already processed the input.
*/

#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <petsc/private/petscimpl.h>
#include <petscvalgrind.h>
#include <petscviewer.h>
#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#endif

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#include <hip/hip_runtime.h>
#include <petschipblas.h>

/*@C
     PetscHIPInitialize - Initializes the HIP device and hipBLAS on the device

     Logically collective

  Input Parameter:
  comm - the MPI communicator that will utilize the HIP devices

  Options Database:
+  -hip_initialize <default yes,no> - do the initialization in PetscInitialize(). If -hip_initialize no is used then the default initialization is done automatically
                               when the first HIP call is made unless you call PetscHIPInitialize() before any HIP operations are performed
.  -hip_view - view information about the HIP devices
.  -hip_synchronize - wait at the end of asynchronize HIP calls so that their time gets credited to the current event; default with -log_view
-  -hip_set_device <gpu> - integer number of the device

  Level: beginner

  Notes:
   Initializing cuBLAS takes about 1/2 second there it is done by default in PetscInitialize() before logging begins

@*/
PetscErrorCode PetscHIPInitialize(MPI_Comm comm)
{
  PetscErrorCode        ierr;
  PetscInt              deviceOpt = 0;
  PetscBool             hip_view_flag = PETSC_FALSE,flg;
  struct hipDeviceProp prop;
  int                   devCount,device,devicecnt;
  hipError_t           err = hipSuccess;
  PetscMPIInt           rank,size;

  PetscFunctionBegin;
  /*
     If collecting logging information, by default, wait for GPU to complete its operations
     before returning to the CPU in order to get accurate timings of each event
  */
  ierr = PetscOptionsHasName(NULL,NULL,"-log_summary",&PetscHIPSynchronize);CHKERRQ(ierr);
  if (!PetscHIPSynchronize) {
    ierr = PetscOptionsHasName(NULL,NULL,"-log_view",&PetscHIPSynchronize);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(comm,NULL,"HIP options","Sys");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-hip_set_device","Set all MPI ranks to use the specified HIP device",NULL,deviceOpt,&deviceOpt,&flg);CHKERRQ(ierr);
  device = (int)deviceOpt;
  ierr = PetscOptionsBool("-hip_synchronize","Wait for the GPU to complete operations before returning to the CPU",NULL,PetscHIPSynchronize,&PetscHIPSynchronize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDeprecated("-hip_show_devices","-hip_view","3.12",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsName("-hip_view","Display HIP device information and assignments",NULL,&hip_view_flag);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!PetscHIPInitialized) {
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size>1 && !flg) {
      /* check to see if we force multiple ranks to hit the same GPU */
      /* we're not using the same GPU on multiple MPI threads. So try to allocated different   GPUs to different processes */

      /* First get the device count */
      err   = hipGetDeviceCount(&devCount);
      if (err != hipSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in hipGetDeviceCount %s",hipGetErrorString(err));

      /* next determine the rank and then set the device via a mod */
      ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      device = rank % devCount;
    }
    err = hipSetDevice(device);
    if (err != hipSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in hipSetDevice %s",hipGetErrorString(err));

    /* set the device flags so that it can map host memory */
    err = hipSetDeviceFlags(hipDeviceMapHost);
    if (err != hipSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in hipSetDeviceFlags %s",hipGetErrorString(err));

    ierr = PetscCUBLASInitializeHandle();CHKERRQ(ierr);
    ierr = PetscCUSOLVERDnInitializeHandle();CHKERRQ(ierr);
    PetscHIPInitialized = PETSC_TRUE;
  }
  if (hip_view_flag) {
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    err  = hipGetDeviceCount(&devCount);
    if (err != hipSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in hipGetDeviceCount %s",hipGetErrorString(err));
    for (devicecnt = 0; devicecnt < devCount; ++devicecnt) {
      err = hipGetDeviceProperties(&prop,devicecnt);
      if (err != hipSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"error in hipGetDeviceProperties %s",hipGetErrorString(err));
      ierr = PetscPrintf(comm, "HIP device %d: %s\n", devicecnt, prop.name);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(comm,"[%d] Using HIP device %d.\n",rank,device);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
