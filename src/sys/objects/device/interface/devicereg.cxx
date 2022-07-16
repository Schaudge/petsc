#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

PetscClassId PETSC_DEVICE_CLASSID, PETSC_DEVICE_CONTEXT_CLASSID;

PetscLogEvent CUBLAS_HANDLE_CREATE, CUSOLVER_HANDLE_CREATE;
PetscLogEvent HIPSOLVER_HANDLE_CREATE, HIPBLAS_HANDLE_CREATE;

static auto registered = false;

static PetscErrorCode PetscDeviceRegisterEvent_Private(const char name[], PetscClassId id, PetscLogEvent *event) {
  PetscFunctionBegin;
  PetscCall(PetscLogEventRegister(name, id, event));
  PetscCall(PetscLogEventSetCollective(*event, PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the `PetscDevice`
  package. It is called from `PetscFinalize()`.

  Developer Notes:
  This function is automatically registered to be called during `PetscFinalize()` by
  `PetscDeviceInitializePackage()` so there should be no need to call it yourself.

  Level: developer

.seealso: `PetscFinalize()`, `PetscDeviceInitializePackage()`
@*/
PetscErrorCode PetscDeviceFinalizePackage(void) {
  registered = false;
  return 0;
}

/*@C
  PetscDeviceInitializePackage - This function initializes everything in the `PetscDevice`
  package. It is called on the first call to `PetscDeviceContextCreate()` or
  `PetscDeviceCreate()` when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`, `PetscDeviceFinalizePackage()`, `PetscDeviceContextCreate()`,
`PetscDeviceCreate()`
@*/
PetscErrorCode PetscDeviceInitializePackage(void) {
  PetscFunctionBegin;
  PetscCheck(PetscDeviceConfiguredFor_Internal(PETSC_DEVICE_DEFAULT()), PETSC_COMM_SELF, PETSC_ERR_SUP, "PETSc is not configured with device support (PETSC_DEVICE_DEFAULT = '%s')", PetscDeviceTypes[PETSC_DEVICE_DEFAULT()]);
  if (PetscLikely(registered)) PetscFunctionReturn(0);
  registered = true;
  PetscCall(PetscRegisterFinalize(PetscDeviceFinalizePackage));
  // class registration
  PetscCall(PetscClassIdRegister("PetscDevice", &PETSC_DEVICE_CLASSID));
  PetscCall(PetscClassIdRegister("PetscDeviceContext", &PETSC_DEVICE_CONTEXT_CLASSID));
  // events
  if (PetscDefined(HAVE_CUDA)) {
    PetscCall(PetscDeviceRegisterEvent_Private("cuBLAS Init", PETSC_DEVICE_CONTEXT_CLASSID, &CUBLAS_HANDLE_CREATE));
    PetscCall(PetscDeviceRegisterEvent_Private("cuSolver Init", PETSC_DEVICE_CONTEXT_CLASSID, &CUSOLVER_HANDLE_CREATE));
  }
  if (PetscDefined(HAVE_HIP)) {
    PetscCall(PetscDeviceRegisterEvent_Private("hipBLAS Init", PETSC_DEVICE_CONTEXT_CLASSID, &HIPBLAS_HANDLE_CREATE));
    PetscCall(PetscDeviceRegisterEvent_Private("hipSolver Init", PETSC_DEVICE_CONTEXT_CLASSID, &HIPSOLVER_HANDLE_CREATE));
  }
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxCreate", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Create));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxDestroy", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Destroy));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxChangeStream", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_ChangeStream));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSetUp", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_SetUp));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSetDevice", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_SetDevice));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxDuplicate", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Duplicate));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxQueryIdle", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_QueryIdle));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxWaitForCtx", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_WaitForCtx));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxFork", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Fork));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxJoin", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Join));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxMarkIntent", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Mark));
  PetscCall(PetscDeviceRegisterEvent_Private("DCtxSync", PETSC_DEVICE_CONTEXT_CLASSID, &DCONTEXT_Sync));
  PetscFunctionReturn(0);
}
