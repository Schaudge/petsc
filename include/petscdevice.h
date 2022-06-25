#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscdevicetypes.h>

/* SUBMANSEC = Sys */

PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscGetMemType(const void *, PetscMemType *);

/* Cannot use the device context api without C++ */
#if PetscDefined(HAVE_CXX)
/* PetscDevice */
PETSC_EXTERN PetscErrorCode PetscDeviceInitialize(PetscDeviceType);
PETSC_EXTERN PetscBool      PetscDeviceInitialized(PetscDeviceType);
PETSC_EXTERN PetscErrorCode PetscDeviceCreate(PetscDeviceType, PetscInt, PetscDevice *);
PETSC_EXTERN PetscErrorCode PetscDeviceConfigure(PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceView(PetscDevice, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceGetType(PetscDevice, PetscDeviceType *);
PETSC_EXTERN PetscErrorCode PetscDeviceGetDeviceId(PetscDevice, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDeviceDestroy(PetscDevice *);
PETSC_EXTERN PetscErrorCode PetscDeviceGetDeviceId(PetscDevice, PetscInt *);

/* PetscDeviceContext */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext, PetscDevice *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext, PetscStreamType *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext, PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextFork(PetscDeviceContext, PetscInt, PetscDeviceContext **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContext **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm, const char[], PetscDeviceContext);
#else
#define PetscDeviceInitialize(...)  0
#define PetscDeviceInitialized(...) PETSC_FALSE
#endif /* PETSC_HAVE_CXX */

#endif /* PETSCDEVICE_H */
