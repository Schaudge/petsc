#ifndef PETSCDEVICEIMPL_H
#define PETSCDEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petsc/private/cpp/type_traits.hpp>

#include <petscdevice.h>

/* logging support */
PETSC_INTERN PetscLogEvent CUBLAS_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent CUSOLVER_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent HIPSOLVER_HANDLE_CREATE;
PETSC_INTERN PetscLogEvent HIPBLAS_HANDLE_CREATE;

PETSC_INTERN PetscLogEvent DCONTEXT_Create;
PETSC_INTERN PetscLogEvent DCONTEXT_Destroy;
PETSC_INTERN PetscLogEvent DCONTEXT_ChangeStream;
PETSC_INTERN PetscLogEvent DCONTEXT_SetDevice;
PETSC_INTERN PetscLogEvent DCONTEXT_SetUp;
PETSC_INTERN PetscLogEvent DCONTEXT_Duplicate;
PETSC_INTERN PetscLogEvent DCONTEXT_QueryIdle;
PETSC_INTERN PetscLogEvent DCONTEXT_WaitForCtx;
PETSC_INTERN PetscLogEvent DCONTEXT_Fork;
PETSC_INTERN PetscLogEvent DCONTEXT_Join;
PETSC_INTERN PetscLogEvent DCONTEXT_Mark;
PETSC_INTERN PetscLogEvent DCONTEXT_Sync;

/* type cast macros for some additional type-safety in C++ land */
#if defined(__cplusplus)
#define PetscStreamTypeCast(...)     static_cast<PetscStreamType>(__VA_ARGS__)
#define PetscDeviceTypeCast(...)     static_cast<PetscDeviceType>(__VA_ARGS__)
#define PetscDeviceInitTypeCast(...) static_cast<PetscDeviceInitType>(__VA_ARGS__)
#else
#define PetscStreamTypeCast(...)     ((PetscStreamType)(__VA_ARGS__))
#define PetscDeviceTypeCast(...)     ((PetscDeviceType)(__VA_ARGS__))
#define PetscDeviceInitTypeCast(...) ((PetscDeviceInitType)(__VA_ARGS__))
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T>
void PetscValidDeviceType(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDeviceTypes(T, int, U, int);
template <typename T>
void PetscValidDevice(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDevices(T, int, U, int);
template <typename T>
void PetscValidStreamType(T, int);
template <typename T>
void PetscValidDeviceContext(T, int);
template <typename T, typename U>
void PetscCheckCompatibleDeviceContexts(T, int, U, int);
#elif PetscDefined(HAVE_CXX) && (PetscDefined(USE_DEBUG) || PetscDefined(DEVICE_KEEP_ERROR_CHECKING_MACROS))
#define PetscValidDeviceType(dtype, argno) \
  do { \
    PetscDeviceType pvdt_dtype_ = PetscDeviceTypeCast(dtype); \
    int             pvdt_argno_ = (int)(argno); \
    PetscCheck(((int)pvdt_dtype_ >= (int)PETSC_DEVICE_HOST) && ((int)pvdt_dtype_ <= (int)PETSC_DEVICE_MAX), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDeviceType '%d': Argument #%d", pvdt_dtype_, pvdt_argno_); \
    if (PetscUnlikely(!PetscDeviceConfiguredFor_Internal(pvdt_dtype_))) { \
      PetscCheck((int)pvdt_dtype_ != (int)PETSC_DEVICE_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid PetscDeviceType '%s': Argument #%d", PetscDeviceTypes[pvdt_dtype_], pvdt_argno_); \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, \
              "Not configured for PetscDeviceType '%s': Argument #%d;" \
              " run configure --help %s for available options", \
              PetscDeviceTypes[pvdt_dtype_], pvdt_argno_, PetscDeviceTypes[pvdt_dtype_]); \
    } \
  } while (0)

#define PetscCheckCompatibleDeviceTypes(dtype1, argno1, dtype2, argno2) \
  do { \
    PetscDeviceType pccdt_dtype1_ = PetscDeviceTypeCast(dtype1); \
    PetscDeviceType pccdt_dtype2_ = PetscDeviceTypeCast(dtype2); \
    PetscValidDeviceType(pccdt_dtype1_, 1); \
    PetscValidDeviceType(pccdt_dtype2_, 2); \
    PetscCheck(pccdt_dtype1_ == pccdt_dtype2_, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "PetscDeviceTypes are incompatible: Arguments #%d and #%d. Expected PetscDeviceType '%s' but has '%s' instead", argno1, argno2, PetscDeviceTypes[pccdt_dtype1_], PetscDeviceTypes[pccdt_dtype2_]); \
  } while (0)

#define PetscValidDevice(dev, argno) \
  do { \
    PetscDevice pvd_dev_   = dev; \
    int         pvd_argno_ = (int)(argno); \
    PetscValidPointer(pvd_dev_, pvd_argno_); \
    PetscValidDeviceType(pvd_dev_->type, pvd_argno_); \
    PetscCheck(pvd_dev_->id >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDevice: Argument #%d; id %" PetscInt_FMT " < 0", pvd_argno_, pvd_dev_->id); \
    PetscCheck(pvd_dev_->refcnt >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDevice: Argument #%d; negative reference count %" PetscInt_FMT, pvd_argno_, pvd_dev_->refcnt); \
  } while (0)

/*
  for now just checks strict equality, but this can be changed as some devices (i.e. kokkos and
  any cupm should be compatible once implemented)
*/
#define PetscCheckCompatibleDevices(dev1, argno1, dev2, argno2) \
  do { \
    PetscDevice pccd_dev1_ = (dev1), pccd_dev2_ = (dev2); \
    int         pccd_argno1_ = (int)(argno1), pccd_argno2_ = (int)(argno2); \
    PetscValidDevice(pccd_dev1_, pccd_argno1_); \
    PetscValidDevice(pccd_dev2_, pccd_argno2_); \
    PetscCheckCompatibleDeviceTypes(pccd_dev1_->type, pccd_argno1_, pccd_dev2_->type, pccd_argno2_); \
  } while (0)

#define PetscValidStreamType(stype, argno) \
  do { \
    PetscStreamType pvst_stype_ = PetscStreamTypeCast(stype); \
    int             pvst_argno_ = (int)(argno); \
    PetscCheck(((int)pvst_stype_ >= 0) && ((int)pvst_stype_ <= (int)PETSC_STREAM_MAX), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamType '%d': Argument #%d", pvst_stype_, pvst_argno_); \
    PetscCheck((int)pvst_stype_ != (int)PETSC_STREAM_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Invalid PetscStreamType '%s': Argument #%d", PetscStreamTypes[pvst_stype_], pvst_argno_); \
  } while (0)

#define PetscValidDeviceContext(dctx, argno) \
  do { \
    PetscDeviceContext pvdc_dctx_  = dctx; \
    int                pvdc_argno_ = (int)(argno); \
    PetscValidHeaderSpecific(pvdc_dctx_, PETSC_DEVICE_CONTEXT_CLASSID, pvdc_argno_); \
    PetscValidStreamType(pvdc_dctx_->streamType, pvdc_argno_); \
    if (pvdc_dctx_->device) { \
      PetscValidDevice(pvdc_dctx_->device, pvdc_argno_); \
    } else { \
      PetscCheck(!pvdc_dctx_->setup, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                 "Invalid PetscDeviceContext: Argument #%d; " \
                 "PetscDeviceContext is setup but has no PetscDevice", \
                 pvdc_argno_); \
    } \
    PetscCheck(((PetscObject)pvdc_dctx_)->id >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid PetscDeviceContext: Argument #%d; id %" PetscInt64_FMT " < 1", pvdc_argno_, ((PetscObject)pvdc_dctx_)->id); \
    PetscCheck(pvdc_dctx_->numChildren <= pvdc_dctx_->maxNumChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid PetscDeviceContext: Argument #%d; number of children %" PetscInt_FMT " > max number of children %" PetscInt_FMT, pvdc_argno_, \
               pvdc_dctx_->numChildren, pvdc_dctx_->maxNumChildren); \
  } while (0)

#define PetscCheckCompatibleDeviceContexts(dctx1, argno1, dctx2, argno2) \
  do { \
    PetscDeviceContext pccdc_dctx1_ = (dctx1), pccdc_dctx2_ = (dctx2); \
    int                pccdc_argno1_ = (int)(argno1), pccdc_argno2_ = (int)(argno2); \
    PetscValidDeviceContext(pccdc_dctx1_, pccdc_argno1_); \
    PetscValidDeviceContext(pccdc_dctx2_, pccdc_argno2_); \
    if (pccdc_dctx1_->device && pccdc_dctx2_->device) PetscCheckCompatibleDevices(pccdc_dctx1_->device, pccdc_argno1_, pccdc_dctx2_->device, pccdc_argno2_); \
  } while (0)
#else /* PetscDefined(USE_DEBUG) */
#define PetscValidDeviceType(dtype, argno)
#define PetscCheckCompatibleDeviceTypes(dtype1, argno1, dtype2, argno2)
#define PetscValidDevice(dev, argno)
#define PetscCheckCompatibleDevices(dev1, argno1, dev2, argno2)
#define PetscValidStreamType(stype, argno)
#define PetscValidDeviceContext(dctx, argno)
#define PetscCheckCompatibleDeviceContexts(dctx1, argno1, dctx2, argno2)
#endif /* PetscDefined(USE_DEBUG) */

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T>
void PetscValidManagedType(T, int);
template <typename T, typename U>
void PetscCheckManagedTypeCompatibleDeviceContext(T, int, U, int);
#elif PetscDefined(USE_DEBUG) || PetscDefined(DEVICE_KEEP_ERROR_CHECKING_MACROS)
#define PetscValidManagedType(scal__, argno__) \
  do { \
    int pvmt_argno_ = (argno__); \
    PetscValidPointer((scal__), pvmt_argno_); \
    if ((scal__)->host) PetscValidPointer((scal__)->host, pvmt_argno_); \
    PetscCheck((scal__)->n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "Invalid PetscManagedType: Argument #%d; Have negative managed type length %" PetscInt_FMT, pvmt_argno_, (scal__)->n); \
    PetscValidDeviceType((scal__)->dtype, pvmt_argno_); \
    if ((scal__)->parent) PetscValidPointer((scal__)->parent, pvmt_argno_); \
  } while (0)

#define PetscCheckManagedTypeCompatibleDeviceContext(dctx__, argno_dctx__, scal__, argno_scal__) \
  do { \
    PetscDeviceType    pcmtcdt_dtype_      = PETSC_DEVICE_MAX; \
    PetscDeviceContext pcmtcdt_dctx_       = (dctx__); \
    int                pcmtcdt_argno_dctx_ = (argno_dctx__), pcmtcdt_argno_scal_ = (argno_scal__); \
    PetscValidDeviceContext(pcmtcdt_dctx_, pcmtcdt_argno_dctx_); \
    PetscValidManagedType((scal__), pcmtcdt_argno_scal_); \
    PetscCall(PetscDeviceContextGetDeviceType(pcmtcdt_dctx_, &pcmtcdt_dtype_)); \
    PetscCheckCompatibleDeviceTypes(pcmtcdt_dtype_, pcmtcdt_argno_dctx_, (scal__)->dtype, pcmtcdt_argno_scal_); \
  } while (0)
#else
#define PetscValidManagedType(scal, argno)
#define PetscCheckManagedTypeCompatibleDeviceContext(dctx, argno_d, scal, argno_s)
#endif
/* if someone is ready to rock with more than 128 GPUs on hand then we're in real trouble */
#define PETSC_DEVICE_MAX_DEVICES 128

/*
  the configure-time default device type, used as the initial the value of
  PETSC_DEVICE_DEFAULT() as well as what it is restored to during PetscFinalize()
*/
#if PetscDefined(HAVE_HIP)
#define PETSC_DEVICE_INITIAL_DEFAULT_TYPE PETSC_DEVICE_HIP
#elif PetscDefined(HAVE_CUDA)
#define PETSC_DEVICE_INITIAL_DEFAULT_TYPE PETSC_DEVICE_CUDA
#elif PetscDefined(HAVE_SYCL)
#define PETSC_DEVICE_INITIAL_DEFAULT_TYPE PETSC_DEVICE_SYCL
#else
#define PETSC_DEVICE_INITIAL_DEFAULT_TYPE PETSC_DEVICE_HOST
#endif

typedef struct _DeviceOps *DeviceOps;
struct _DeviceOps {
  /* the creation routine for the corresponding PetscDeviceContext, this is NOT intended
   * to be called by the PetscDevice itself */
  PetscErrorCode (*createcontext)(PetscDeviceContext);
  PetscErrorCode (*configure)(PetscDevice);
  PetscErrorCode (*view)(PetscDevice, PetscViewer);
};

struct _n_PetscDevice {
  struct _DeviceOps ops[1];
  PetscInt          refcnt;   /* reference count for the device */
  PetscInt          id;       /* unique id per created PetscDevice */
  PetscInt          deviceId; /* the id of the underlying device, i.e. the return of
                               * cudaGetDevice() for example */
  PetscDeviceType   type;     /* type of device */
  void             *data;     /* placeholder */
};

#define PetscManagedTypeOps_(PetscManagedType, PetscType, PetscTypeSuffix_L) \
  PetscErrorCode (*destroymanaged##PetscTypeSuffix_L)(PetscDeviceContext, PetscManagedType); \
  PetscErrorCode (*getmanagedvalues##PetscTypeSuffix_L)(PetscDeviceContext, PetscManagedType, PetscMemType, PetscMemoryAccessMode, PetscType **); \
  PetscErrorCode (*applyoperator##PetscTypeSuffix_L)(PetscDeviceContext, PetscManagedType, PetscOperatorType, PetscMemType, const PetscType *, PetscManagedType)

#define PetscManagedTypeOps(PetscTypeSuffix, PetscTypeSuffix_L) PetscManagedTypeOps_(PetscConcat(PetscManaged, PetscTypeSuffix), PetscConcat(Petsc, PetscTypeSuffix), PetscTypeSuffix_L)

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext, PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext, PetscBool *);
  PetscErrorCode (*waitforcontext)(PetscDeviceContext, PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
  PetscErrorCode (*getblashandle)(PetscDeviceContext, void *);
  PetscErrorCode (*getsolverhandle)(PetscDeviceContext, void *);
  PetscErrorCode (*getstreamhandle)(PetscDeviceContext, void *);
  PetscErrorCode (*begintimer)(PetscDeviceContext);
  PetscErrorCode (*endtimer)(PetscDeviceContext, PetscLogDouble *);
  PetscErrorCode (*memalloc)(PetscDeviceContext, PetscBool, PetscMemType, size_t, void **);
  PetscErrorCode (*memfree)(PetscDeviceContext, PetscMemType, void *);
  PetscErrorCode (*memcopy)(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, size_t, PetscDeviceCopyMode);
  PetscErrorCode (*memset)(PetscDeviceContext, PetscMemType, void *, PetscInt, size_t);
  PetscManagedTypeOps(Scalar, scalar);
  PetscManagedTypeOps(Real, real);
  PetscManagedTypeOps(Int, int);
};

#undef PetscManagedTypeOps
#undef PetscManagedTypeOps_

typedef struct {
  PetscBool allow_orphans;
} DeviceContextOptions;

struct _p_PetscDeviceContext {
  PETSCHEADER(struct _DeviceContextOps);
  PetscDevice          device;         /* the device this context stems from */
  void                *data;           /* solver contexts, event, stream */
  PetscObjectId       *childIDs;       /* array containing ids of contexts currently forked from this one */
  PetscInt             numChildren;    /* how many children does this context expect to destroy */
  PetscInt             maxNumChildren; /* how many children can this context have room for without realloc'ing */
  PetscStreamType      streamType;     /* how should this contexts stream behave around other streams? */
  PetscBool            setup;
  PetscBool            usersetdevice;
  PetscBool            contained;
  DeviceContextOptions options;
};

// ===================================================================================
//                            PetscDevice Internal Functions
// ===================================================================================
#if PetscDefined(HAVE_CXX)
PETSC_INTERN PetscErrorCode                PetscDeviceInitializeFromOptions_Internal(MPI_Comm);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceType, PetscInt);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType, PetscDevice *);

static inline PetscErrorCode PetscDeviceReference_Internal(PetscDevice device) {
  PetscFunctionBegin;
  ++(device->refcnt);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceDereference_Internal(PetscDevice device) {
  PetscFunctionBegin;
  --(device->refcnt);
  PetscAssert(device->refcnt >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT, "PetscDevice has negative reference count %" PetscInt_FMT, device->refcnt);
  PetscFunctionReturn(0);
}
#else /* PETSC_HAVE_CXX for PetscDevice Internal Functions */
#define PetscDeviceInitializeFromOptions_Internal(comm)       0
#define PetscDeviceInitializeDefaultDevice_Internal(type, id) 0
#define PetscDeviceGetDefaultForType_Internal(Type, device)   0
#define PetscDeviceReference_Internal(device)                 0
#define PetscDeviceDereference_Internal(device)               0
#endif /* PETSC_HAVE_CXX for PetscDevice Internal Functions */

static inline PetscErrorCode PetscDeviceCheckDeviceCount_Internal(PetscInt count) {
  PetscFunctionBegin;
  PetscAssert(count < PETSC_DEVICE_MAX_DEVICES, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected %" PetscInt_FMT " devices, which is larger than maximum supported number of devices %d", count, PETSC_DEVICE_MAX_DEVICES);
  PetscFunctionReturn(0);
}

/* More general form of PetscDeviceDefaultType_Internal(), as it calls the former using
 * the automatically selected default PetscDeviceType */
#define PetscDeviceGetDefault_Internal(device) PetscDeviceGetDefaultForType_Internal(PETSC_DEVICE_DEFAULT(), device)

static inline PETSC_CONSTEXPR_14 PetscBool PetscDeviceConfiguredFor_Internal(PetscDeviceType type) {
  switch (type) {
  case PETSC_DEVICE_HOST:
    return PETSC_TRUE;
    /* casts are needed in C++ */
  case PETSC_DEVICE_CUDA: return (PetscBool)PetscDefined(HAVE_CUDA);
  case PETSC_DEVICE_HIP: return (PetscBool)PetscDefined(HAVE_HIP);
  case PETSC_DEVICE_SYCL: return (PetscBool)PetscDefined(HAVE_SYCL);
  case PETSC_DEVICE_MAX:
    return PETSC_FALSE;
    /* Do not add default case! Will make compiler warn on new additions to PetscDeviceType! */
  }
  PetscUnreachable();
  return PETSC_FALSE;
}

// ===================================================================================
//                     PetscDeviceContext Internal Functions
// ===================================================================================
#if PetscDefined(HAVE_CXX)
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType);
PETSC_INTERN PetscErrorCode                PetscDeviceContextSetRootStreamType_Internal(PetscStreamType);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *);

static inline PetscErrorCode PetscDeviceContextGetHandle_Private(PetscDeviceContext dctx, void *handle, PetscErrorCode (*gethandle_op)(PetscDeviceContext, void *)) {
  PetscFunctionBegin;
  PetscValidPointer(handle, 2);
  PetscValidFunction(gethandle_op, 3);
  PetscCall((*gethandle_op)(dctx, handle));
  // getting a handle implies work is being done
  dctx->contained = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetBLASHandle_Internal(PetscDeviceContext dctx, void *handle) {
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getblashandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetSOLVERHandle_Internal(PetscDeviceContext dctx, void *handle) {
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getsolverhandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetStreamHandle_Internal(PetscDeviceContext dctx, void *handle) {
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetHandle_Private(dctx, handle, dctx->ops->getstreamhandle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextBeginTimer_Internal(PetscDeviceContext dctx) {
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscUseTypeMethod(dctx, begintimer);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextEndTimer_Internal(PetscDeviceContext dctx, PetscLogDouble *elapsed) {
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx, 1);
  PetscValidRealPointer(elapsed, 2);
  PetscUseTypeMethod(dctx, endtimer, elapsed);
  PetscFunctionReturn(0);
}
#else /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */
#define PetscDeviceContextSetRootDeviceType_Internal(type)       0
#define PetscDeviceContextSetRootStreamType_Internal(type)       0
#define PetscDeviceContextGetNullContext_Internal(dctx)          (*(dctx) = PETSC_NULLPTR, 0)
#define PetscDeviceContextGetBLASHandle_Internal(dctx, handle)   (*(handle) = PETSC_NULLPTR, 0)
#define PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle) (*(handle) = PETSC_NULLPTR, 0)
#define PetscDeviceContextGetStreamHandle_Internal(dctx, handle) (*(handle) = PETSC_NULLPTR, 0)
#define PetscDeviceContextBeginTimer_Internal(dctx)              0
#define PetscDeviceContextEndTimer_Internal(dctx, elapsed)       0
#endif /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */

#define PetscDeviceContextSetDefaultDevice_Internal(dctx) PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, PETSC_DEVICE_DEFAULT())

/* note, only does assertion checking in debug mode */
static inline PetscErrorCode PetscDeviceContextGetCurrentContextAssertType_Internal(PetscDeviceContext *dctx, PetscDeviceType type) {
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(dctx));
  if (PetscDefined(USE_DEBUG)) {
    PetscDevice     dev;
    PetscDeviceType dtype;

    PetscValidDeviceType(type, 2);
    PetscCall(PetscDeviceContextGetDevice(*dctx, &dev));
    PetscCall(PetscDeviceGetType(dev, &dtype));
    PetscCheckCompatibleDeviceTypes(dtype, 1, type, 2);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetOptionalNullContext_Internal(PetscDeviceContext *dctx) {
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  if (!*dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(dctx));
  PetscValidDeviceContext(*dctx, 1);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextAllReduceManagedType_Internal(PetscDeviceContext dctx, void *ptr, const PetscInt *nin, MPI_Datatype dtype, MPI_Op op, PetscObject obj) {
  MPI_Comm    comm;
  PetscMPIInt n, size;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscValidIntPointer(nin, 3);
  PetscValidHeader(obj, 6);
  PetscCall(PetscMPIIntCast(*nin, &n));
  PetscCall(PetscObjectGetComm(obj, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, ptr, n, dtype, op, comm));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextAllReduceManagedScalar_Internal(PetscDeviceContext dctx, PetscManagedScalar scal, const PetscInt *n, MPI_Op op, PetscObject obj) {
  PetscScalar *scalptr;

  PetscFunctionBegin;
  if (use_gpu_aware_mpi) {
    PetscMemType mtype;

    PetscCall(PetscManagedScalarGetPointerAndMemType(dctx, scal, PETSC_MEMORY_ACCESS_READ_WRITE, &scalptr, &mtype));
    // we are about to sync, so we can reset this
    if (PetscMemTypeHost(mtype)) scal->pure = PETSC_TRUE;
  } else {
    PetscCall(PetscManagedScalarGetValues(dctx, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &scalptr));
  }
  PetscCall(PetscDeviceContextAllReduceManagedType_Internal(dctx, scalptr, n, MPIU_SCALAR, op, obj));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextAllReduceManagedReal_Internal(PetscDeviceContext dctx, PetscManagedReal scal, const PetscInt *n, MPI_Op op, PetscObject obj) {
  PetscReal *scalptr;

  PetscFunctionBegin;
  if (use_gpu_aware_mpi) {
    PetscMemType mtype;

    PetscCall(PetscManagedRealGetPointerAndMemType(dctx, scal, PETSC_MEMORY_ACCESS_READ_WRITE, &scalptr, &mtype));
    // we are about to sync, so we can reset this
    if (PetscMemTypeHost(mtype)) scal->pure = PETSC_TRUE;
  } else {
    PetscCall(PetscManagedRealGetValues(dctx, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &scalptr));
  }
  PetscCall(PetscDeviceContextAllReduceManagedType_Internal(dctx, scalptr, n, MPIU_REAL, op, obj));
  PetscFunctionReturn(0);
}
#define PetscWrapHostTypeAndDctx(Type, ptr_name__, ptr_size__, scal_name__, dctx_name__, ...) \
  do { \
    PetscManaged##Type scal_name__; \
\
    PetscCall(PetscManageHost##Type(dctx_name__, ptr_name__, ptr_size__, &scal_name__)); \
    __VA_ARGS__; \
    PetscCall(PetscManagedHost##Type##Destroy(dctx_name__, scal_name__)); \
  } while (0)

#define PetscWrapHostType(Type, ptr_name__, ptr_size__, scal_name__, ...) \
  do { \
    PetscDeviceContext dctx; \
\
    PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx)); \
    PetscWrapHostTypeAndDctx(Type, ptr_name__, ptr_size__, scal_name__, dctx, __VA_ARGS__); \
  } while (0)

#define PetscWrapHostScalar(ptr_name, ptr_size, scal_name, ...) PetscWrapHostType(Scalar, ptr_name, ptr_size, scal_name, __VA_ARGS__)

#define PetscWrapHostReal(ptr_name, ptr_size, scal_name, ...) PetscWrapHostType(Real, ptr_name, ptr_size, scal_name, __VA_ARGS__)

#define PetscWrapHostInt(ptr_name, ptr_size, scal_name, ...) PetscWrapHostType(Int, ptr_name, ptr_size, scal_name, __VA_ARGS__)

PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext);
#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_SYCL)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext);
#endif
#endif /* PETSCDEVICEIMPL_H */
