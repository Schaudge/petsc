/*
    Defines profile/logging in PETSc.
*/
#ifndef PETSCLOG_H
#define PETSCLOG_H

#include <petscsystypes.h>
#include <petsctime.h>
#include <petscbt.h>
#include <petscoptions.h>

/* SUBMANSEC = Sys */

/* General logging of information; different from event logging */
PETSC_EXTERN PetscErrorCode PetscInfo_Private(const char[], PetscObject, const char[], ...) PETSC_ATTRIBUTE_FORMAT(3, 4);
#if defined(PETSC_USE_INFO)
  #define PetscInfo(A, ...) PetscInfo_Private(PETSC_FUNCTION_NAME, ((PetscObject)A), __VA_ARGS__)
#else
  #define PetscInfo(A, ...) PETSC_SUCCESS
#endif

#define PetscInfo1(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo2(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo3(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo4(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo5(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo6(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo7(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo8(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo9(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)

/*E
  PetscInfoCommFlag - Describes the method by which to filter information displayed by `PetscInfo()` by communicator size

  Values:
+ `PETSC_INFO_COMM_ALL` - Default uninitialized value. `PetscInfo()` will not filter based on
                          communicator size (i.e. will print for all communicators)
. `PETSC_INFO_COMM_NO_SELF` - `PetscInfo()` will NOT print for communicators with size = 1 (i.e. *_COMM_SELF)
- `PETSC_INFO_COMM_ONLY_SELF` - `PetscInfo()` will ONLY print for communicators with size = 1

  Level: intermediate

  Note:
  Used as an input for `PetscInfoSetFilterCommSelf()`

.seealso: `PetscInfo()`, `PetscInfoSetFromOptions()`, `PetscInfoSetFilterCommSelf()`
E*/
typedef enum {
  PETSC_INFO_COMM_ALL       = -1,
  PETSC_INFO_COMM_NO_SELF   = 0,
  PETSC_INFO_COMM_ONLY_SELF = 1
} PetscInfoCommFlag;

PETSC_EXTERN const char *const PetscInfoCommFlags[];
PETSC_EXTERN PetscErrorCode    PetscInfoDeactivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode    PetscInfoActivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode    PetscInfoEnabled(PetscClassId, PetscBool *);
PETSC_EXTERN PetscErrorCode    PetscInfoAllow(PetscBool);
PETSC_EXTERN PetscErrorCode    PetscInfoSetFile(const char[], const char[]);
PETSC_EXTERN PetscErrorCode    PetscInfoGetFile(char **, FILE **);
PETSC_EXTERN PetscErrorCode    PetscInfoSetClasses(PetscBool, PetscInt, const char *const *);
PETSC_EXTERN PetscErrorCode    PetscInfoGetClass(const char *, PetscBool *);
PETSC_EXTERN PetscErrorCode    PetscInfoGetInfo(PetscBool *, PetscBool *, PetscBool *, PetscBool *, PetscInfoCommFlag *);
PETSC_EXTERN PetscErrorCode    PetscInfoProcessClass(const char[], PetscInt, const PetscClassId[]);
PETSC_EXTERN PetscErrorCode    PetscInfoSetFilterCommSelf(PetscInfoCommFlag);
PETSC_EXTERN PetscErrorCode    PetscInfoSetFromOptions(PetscOptions);
PETSC_EXTERN PetscErrorCode    PetscInfoDestroy(void);
PETSC_EXTERN PetscBool         PetscLogPrintInfo; /* if true, indicates PetscInfo() is turned on */

PETSC_EXTERN PetscErrorCode PetscIntStackCreate(PetscIntStack *);
PETSC_EXTERN PetscErrorCode PetscIntStackDestroy(PetscIntStack);
PETSC_EXTERN PetscErrorCode PetscIntStackPush(PetscIntStack, int);
PETSC_EXTERN PetscErrorCode PetscIntStackPop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackTop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackEmpty(PetscIntStack, PetscBool *);

#include <petsclogdeprecated.h>

typedef struct _n_PetscLogRegistry *PetscLogRegistry;

typedef struct _n_PetscLogState *PetscLogState;
struct _n_PetscLogState {
  PetscLogRegistry registry;
  PetscBT          active;
  PetscIntStack    stage_stack;
  int              current_stage;
  int              bt_num_stages;
  int              bt_num_events;
  int              refct;
};

PETSC_EXTERN PetscErrorCode PetscLogHandlerCreate(MPI_Comm, PetscLogHandler *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetContext(PetscLogHandler, void *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetContext(PetscLogHandler, void *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetOperation(PetscLogHandler, PetscLogHandlerOpType, void (*)(void));
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetOperation(PetscLogHandler, PetscLogHandlerOpType, void (**)(void));
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetState(PetscLogHandler, PetscLogState);
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetState(PetscLogHandler, PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventBegin(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventEnd(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventSync(PetscLogHandler, PetscLogEvent, MPI_Comm);
//PETSC_EXTERN PetscErrorCode PetscLogHandlerEventIgnorePush(PetscLogHandler, PetscLogStage, PetscLogEvent);
//PETSC_EXTERN PetscErrorCode PetscLogHandlerEventIgnorePop(PetscLogHandler, PetscLogStage, PetscLogEvent);
//PETSC_EXTERN PetscErrorCode PetscLogHandlerEventsPause(PetscLogHandler);
//PETSC_EXTERN PetscErrorCode PetscLogHandlerEventsUnpause(PetscLogHandler);
PETSC_EXTERN PetscErrorCode PetscLogHandlerObjectCreate(PetscLogHandler, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerObjectDestroy(PetscLogHandler, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerStagePush(PetscLogHandler, PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogHandlerStagePop(PetscLogHandler, PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogHandlerView(PetscLogHandler, PetscViewer);

#define PetscLogStateEventCurrentlyActive(state, event) ((state) && PetscBTLookup((state)->active, (state)->current_stage) && PetscBTLookup((state)->active, (state)->current_stage + (event + 1) * (state)->bt_num_stages))

typedef struct _n_PetscLogHandlerEntry *PetscLogHandlerEntry;
typedef PetscErrorCode (*PetscLogEventFn)(PetscLogHandlerEntry, PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
typedef PetscErrorCode (*PetscLogEventSyncFn)(PetscLogHandlerEntry, PetscLogState, PetscLogEvent, MPI_Comm);
typedef PetscErrorCode (*PetscLogObjectFn)(PetscLogHandlerEntry, PetscLogState, PetscObject);
typedef struct _n_PetscLogHandlerImpl *PetscLogHandlerImpl;

typedef PetscErrorCode (*_PetscLogEventFn)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
typedef PetscErrorCode (*_PetscLogEventSyncFn)(PetscLogHandler, PetscLogEvent, MPI_Comm);
typedef PetscErrorCode (*_PetscLogObjectFn)(PetscLogHandler, PetscObject);

typedef struct _n_PetscLogHandlerHot {
  PetscLogHandler      handler;
  _PetscLogEventFn     EventBegin;
  _PetscLogEventFn     EventEnd;
  _PetscLogEventSyncFn EventSync;
  _PetscLogObjectFn    ObjectCreate;
  _PetscLogObjectFn    ObjectDestroy;
} PetscLogHandlerHot;

struct _n_PetscLogHandlerEntry {
  PetscLogHandlerImpl impl;
  PetscLogEventFn     event_begin;
  PetscLogEventFn     event_end;
  PetscLogEventSyncFn event_sync;
  PetscLogObjectFn    object_create;
  PetscLogObjectFn    object_destroy;
};

/* Handle multithreading */
#if defined(PETSC_HAVE_THREADSAFETY)
  #if defined(__cplusplus)
    #define PETSC_TLS thread_local
  #else
    #define PETSC_TLS _Thread_local
  #endif
  #define PETSC_EXTERN_TLS extern PETSC_TLS PETSC_VISIBILITY_PUBLIC
PETSC_EXTERN PetscErrorCode PetscAddLogDouble(PetscLogDouble *, PetscLogDouble *, PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscAddLogDoubleCnt(PetscLogDouble *, PetscLogDouble *, PetscLogDouble *, PetscLogDouble *, PetscLogDouble);
#else
  #define PETSC_EXTERN_TLS PETSC_EXTERN
  #define PETSC_TLS
  #define PetscAddLogDouble(a, b, c)          ((PetscErrorCode)((*(a) += (c), PETSC_SUCCESS) || ((*(b) += (c)), PETSC_SUCCESS)))
  #define PetscAddLogDoubleCnt(a, b, c, d, e) ((PetscErrorCode)(PetscAddLogDouble(a, c, 1) || PetscAddLogDouble(b, d, e)))
#endif


PETSC_DEPRECATED_FUNCTION("PetscLogObjectParent() is deprecated (since version 3.18)") static inline PetscErrorCode PetscLogObjectParent(PetscObject o, PetscObject p)
{
  (void)o;
  (void)p;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_FUNCTION("PetscLogObjectMemory() is deprecated (since version 3.18)") static inline PetscErrorCode PetscLogObjectMemory(PetscObject o, PetscLogDouble m)
{
  (void)o;
  (void)m;
  return PETSC_SUCCESS;
}

#if defined(PETSC_USE_LOG) /* --- Logging is turned on --------------------------------*/
PETSC_EXTERN PetscLogState petsc_log_state;

  #define PETSC_LOG_HANDLER_MAX 4
PETSC_EXTERN PetscLogHandlerEntry PetscLogHandlers[PETSC_LOG_HANDLER_MAX];
PETSC_EXTERN PetscLogHandlerHot   _PetscLogHandlers[PETSC_LOG_HANDLER_MAX];

PETSC_EXTERN PetscErrorCode PetscLogHandlerStart(PetscLogHandler);
PETSC_EXTERN PetscErrorCode PetscLogHandlerStop(PetscLogHandler);

PETSC_EXTERN PetscErrorCode PetscGetFlops(PetscLogDouble *);

  #if defined(PETSC_HAVE_MPE)
PETSC_EXTERN PetscErrorCode PetscLogMPEBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogMPEDump(const char[]);
  #endif

/* Initialization functions */
PETSC_EXTERN PetscErrorCode PetscLogDefaultBegin(void);
PETSC_DEPRECATED_FUNCTION("Use PetscLogDefaultBegin() (since version 3.20)") static inline PetscErrorCode PetscLogAllBegin(void)
{
  return PetscLogDefaultBegin();
}
PETSC_EXTERN PetscErrorCode PetscLogNestedBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogTraceBegin(FILE *);
PETSC_EXTERN PetscErrorCode PetscLogActions(PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogObjects(PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogSetThreshold(PetscLogDouble, PetscLogDouble *);

/* Output functions */
PETSC_EXTERN PetscErrorCode PetscLogView(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscLogViewFromOptions(void);
PETSC_EXTERN PetscErrorCode PetscLogDump(const char[]);

/* Status checking functions */
PETSC_EXTERN PetscErrorCode PetscLogIsActive(PetscBool *);

/* Stage functions */
PETSC_EXTERN PetscErrorCode PetscLogStageRegister(const char[], PetscLogStage *);
PETSC_EXTERN PetscErrorCode PetscLogStagePush(PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogStagePop(void);
PETSC_EXTERN PetscErrorCode PetscLogStageSetActive(PetscLogStage, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStageGetActive(PetscLogStage, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLogStageSetVisible(PetscLogStage, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStageGetVisible(PetscLogStage, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLogStageGetId(const char[], PetscLogStage *);
PETSC_EXTERN PetscErrorCode PetscLogStageGetName(PetscLogEvent, const char **);

/* Event functions */
PETSC_EXTERN PetscErrorCode PetscLogEventRegister(const char[], PetscClassId, PetscLogEvent *);
PETSC_EXTERN PetscErrorCode PetscLogEventSetCollective(PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogEventIncludeClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventExcludeClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventActivate(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventGetActive(PetscLogStage, PetscLogEvent, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivate(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivatePush(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivatePop(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventsPause(void);
PETSC_EXTERN PetscErrorCode PetscLogEventsUnpause(void);
PETSC_EXTERN PetscErrorCode PetscLogEventSetActiveAll(PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogEventActivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventGetId(const char[], PetscLogEvent *);
PETSC_EXTERN PetscErrorCode PetscLogEventGetName(PetscLogEvent, const char **);
PETSC_EXTERN PetscErrorCode PetscLogEventGetPerfInfo(PetscLogStage, PetscLogEvent, PetscEventPerfInfo *);
PETSC_EXTERN PetscErrorCode PetscLogEventSetDof(PetscLogEvent, PetscInt, PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscLogEventSetError(PetscLogEvent, PetscInt, PetscLogDouble);

PETSC_EXTERN PetscErrorCode PetscLogClassGetId(const char[], PetscClassId *);
PETSC_EXTERN PetscErrorCode PetscLogClassGetName(PetscClassId, const char **);

static inline PETSC_UNUSED PetscErrorCode PetscLogEventSync(PetscLogEvent e, MPI_Comm comm)
{
  if (PetscLogStateEventCurrentlyActive(petsc_log_state, e)) {
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      PetscLogHandlerEntry h = PetscLogHandlers[i];
      if (h && h->event_sync) {
        PetscErrorCode err = (*(h->event_sync))(h, petsc_log_state, e, comm);
        if (err != PETSC_SUCCESS) return err;
      }
    }
  }
  return PETSC_SUCCESS;
}

static inline PETSC_UNUSED PetscErrorCode PetscLogEventBegin_Internal(PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  if (PetscLogStateEventCurrentlyActive(petsc_log_state, e)) {
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      PetscLogHandlerEntry h = PetscLogHandlers[i];
      if (h && h->event_begin) {
        PetscErrorCode err = (*(h->event_begin))(h, petsc_log_state, e, 0, o1, o2, o3, o4);
        if (err != PETSC_SUCCESS) return err;
      }
    }
  }
  return PETSC_SUCCESS;
}
  #define PetscLogEventBegin(e, o1, o2, o3, o4) PetscLogEventBegin_Internal(e, (PetscObject)(o1), (PetscObject)(o2), (PetscObject)(o3), (PetscObject)(o4))

static inline PETSC_UNUSED PetscErrorCode PetscLogEventEnd_Internal(PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  if (PetscLogStateEventCurrentlyActive(petsc_log_state, e)) {
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      PetscLogHandlerEntry h = PetscLogHandlers[i];
      if (h && h->event_end) {
        PetscErrorCode err = (*(h->event_end))(h, petsc_log_state, e, 0, o1, o2, o3, o4);
        if (err != PETSC_SUCCESS) return err;
      }
    }
  }
  return PETSC_SUCCESS;
}
  #define PetscLogEventEnd(e, o1, o2, o3, o4) PetscLogEventEnd_Internal(e, (PetscObject)(o1), (PetscObject)(o2), (PetscObject)(o3), (PetscObject)(o4))

  /* Object functions */
  #define PetscLogObjectParents(p, n, d) PetscMacroReturnStandard(for (int _i = 0; _i < (n); ++_i) PetscCall(PetscLogObjectParent((PetscObject)(p), (PetscObject)(d)[_i]));)
static inline PETSC_UNUSED PetscErrorCode PetscLogObjectCreate_Internal(PetscObject o)
{
  if (petsc_log_state) {
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      PetscLogHandlerEntry h = PetscLogHandlers[i];
      if (h && h->object_create) {
        PetscErrorCode err = (*(h->object_create))(h, petsc_log_state, o);
        if (err != PETSC_SUCCESS) return err;
      }
    }
  }
  return PETSC_SUCCESS;
}
  #define PetscLogObjectCreate(o) PetscLogObjectCreate_Internal((PetscObject)(o))

static inline PETSC_UNUSED PetscErrorCode PetscLogObjectDestroy_Internal(PetscObject o)
{
  if (petsc_log_state) {
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      PetscLogHandlerEntry h = PetscLogHandlers[i];
      if (h && h->object_destroy) {
        PetscErrorCode err = (*(h->object_destroy))(h, petsc_log_state, o);
        if (err != PETSC_SUCCESS) return err;
      }
    }
  }
  return PETSC_SUCCESS;
}
  #define PetscLogObjectDestroy(o) PetscLogObjectDestroy_Internal((PetscObject)(o))

PETSC_EXTERN PetscBool PetscLogMemory;
PETSC_EXTERN PetscBool PetscLogSyncOn; /* true if logging synchronization is enabled */

/* Global flop counter */
PETSC_EXTERN PetscLogDouble petsc_TotalFlops;
PETSC_EXTERN PetscLogDouble petsc_irecv_ct;
PETSC_EXTERN PetscLogDouble petsc_isend_ct;
PETSC_EXTERN PetscLogDouble petsc_recv_ct;
PETSC_EXTERN PetscLogDouble petsc_send_ct;
PETSC_EXTERN PetscLogDouble petsc_irecv_len;
PETSC_EXTERN PetscLogDouble petsc_isend_len;
PETSC_EXTERN PetscLogDouble petsc_recv_len;
PETSC_EXTERN PetscLogDouble petsc_send_len;
PETSC_EXTERN PetscLogDouble petsc_allreduce_ct;
PETSC_EXTERN PetscLogDouble petsc_gather_ct;
PETSC_EXTERN PetscLogDouble petsc_scatter_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_any_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_all_ct;
PETSC_EXTERN PetscLogDouble petsc_sum_of_waits_ct;

/* Thread local storage */
PETSC_EXTERN_TLS PetscLogDouble petsc_TotalFlops_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_irecv_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_isend_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_recv_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_send_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_irecv_len_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_isend_len_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_recv_len_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_send_len_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_allreduce_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gather_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_scatter_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_wait_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_wait_any_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_wait_all_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_sum_of_waits_ct_th;

  /*
   Flop counting:  We count each arithmetic operation (e.g., addition, multiplication) separately.

   For the complex numbers version, note that
       1 complex addition = 2 flops
       1 complex multiplication = 6 flops,
   where we define 1 flop as that for a double precision scalar.  We roughly approximate
   flop counting for complex numbers by multiplying the total flops by 4; this corresponds
   to the assumption that we're counting mostly additions and multiplications -- and
   roughly the same number of each.  More accurate counting could be done by distinguishing
   among the various arithmetic operations.
 */

  #if defined(PETSC_USE_COMPLEX)
    #define PETSC_FLOPS_PER_OP 4.0
  #else
    #define PETSC_FLOPS_PER_OP 1.0
  #endif

/*@C
       PetscLogFlops - Log how many flops are performed in a calculation

   Input Parameter:
.   flops - the number of flops

   Level: intermediate

   Note:
     To limit the chance of integer overflow when multiplying by a constant, represent the constant as a double,
     not an integer. Use `PetscLogFlops`(4.0*n) not `PetscLogFlops`(4*n)

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`
@*/
static inline PetscErrorCode PetscLogFlops(PetscLogDouble n)
{
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot log negative flops");
  return PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th, PETSC_FLOPS_PER_OP * n);
}

  /*
     These are used internally in the PETSc routines to keep a count of MPI messages and
   their sizes.

     This does not work for MPI-Uni because our include/petsc/mpiuni/mpi.h file
   uses macros to defined the MPI operations.

     It does not work correctly from HP-UX because it processes the
   macros in a way that sometimes it double counts, hence
   PETSC_HAVE_BROKEN_RECURSIVE_MACRO

     It does not work with Windows because winmpich lacks MPI_Type_size()
*/
  #if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO)
/*
   Logging of MPI activities
*/
static inline PetscErrorCode PetscMPITypeSize(PetscInt count, MPI_Datatype type, PetscLogDouble *length, PetscLogDouble *length_th)
{
  PetscMPIInt typesize;

  if (type == MPI_DATATYPE_NULL) return PETSC_SUCCESS;
  PetscCallMPI(MPI_Type_size(type, &typesize));
  return PetscAddLogDouble(length, length_th, (PetscLogDouble)(count * typesize));
}

static inline PetscErrorCode PetscMPITypeSizeComm(MPI_Comm comm, const PetscMPIInt *counts, MPI_Datatype type, PetscLogDouble *length, PetscLogDouble *length_th)
{
  PetscMPIInt    typesize, size, p;
  PetscLogDouble l;

  if (type == MPI_DATATYPE_NULL) return PETSC_SUCCESS;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Type_size(type, &typesize));
  for (p = 0, l = 0.0; p < size; ++p) l += (PetscLogDouble)(counts[p] * typesize);
  return PetscAddLogDouble(length, length_th, l);
}

/*
    Returns 1 if the communicator is parallel else zero
*/
static inline int PetscMPIParallelComm(MPI_Comm comm)
{
  PetscMPIInt size;
  MPI_Comm_size(comm, &size);
  return size > 1;
}

    #define MPI_Irecv(buf, count, datatype, source, tag, comm, request) \
      (PetscAddLogDouble(&petsc_irecv_ct, &petsc_irecv_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_irecv_len), &(petsc_irecv_len_th)) || MPI_Irecv((buf), (count), (datatype), (source), (tag), (comm), (request)))

    #define MPI_Irecv_c(buf, count, datatype, source, tag, comm, request) \
      (PetscAddLogDouble(&petsc_irecv_ct, &petsc_irecv_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_irecv_len), &(petsc_irecv_len_th)) || MPI_Irecv_c((buf), (count), (datatype), (source), (tag), (comm), (request)))

    #define MPI_Isend(buf, count, datatype, dest, tag, comm, request) \
      (PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_isend_len), &(petsc_isend_len_th)) || MPI_Isend((buf), (count), (datatype), (dest), (tag), (comm), (request)))

    #define MPI_Isend_c(buf, count, datatype, dest, tag, comm, request) \
      (PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_isend_len), &(petsc_isend_len_th)) || MPI_Isend_c((buf), (count), (datatype), (dest), (tag), (comm), (request)))

    #define MPI_Startall_irecv(count, datatype, number, requests) \
      (PetscAddLogDouble(&petsc_irecv_ct, &petsc_irecv_ct_th, number) || PetscMPITypeSize((count), (datatype), &(petsc_irecv_len), &(petsc_irecv_len_th)) || ((number) && MPI_Startall((number), (requests))))

    #define MPI_Startall_isend(count, datatype, number, requests) \
      (PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, number) || PetscMPITypeSize((count), (datatype), &(petsc_isend_len), &(petsc_isend_len_th)) || ((number) && MPI_Startall((number), (requests))))

    #define MPI_Start_isend(count, datatype, requests) (PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_isend_len), (&petsc_isend_len_th)) || MPI_Start((requests)))

    #define MPI_Recv(buf, count, datatype, source, tag, comm, status) \
      (PetscAddLogDouble(&petsc_recv_ct, &petsc_recv_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_recv_len), (&petsc_recv_len_th)) || MPI_Recv((buf), (count), (datatype), (source), (tag), (comm), (status)))

    #define MPI_Recv_c(buf, count, datatype, source, tag, comm, status) \
      (PetscAddLogDouble(&petsc_recv_ct, &petsc_recv_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_recv_len), &(petsc_recv_len_th)) || MPI_Recv_c((buf), (count), (datatype), (source), (tag), (comm), (status)))

    #define MPI_Send(buf, count, datatype, dest, tag, comm) \
      (PetscAddLogDouble(&petsc_send_ct, &petsc_send_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Send((buf), (count), (datatype), (dest), (tag), (comm)))

    #define MPI_Send_c(buf, count, datatype, dest, tag, comm) \
      (PetscAddLogDouble(&petsc_send_ct, &petsc_send_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Send_c((buf), (count), (datatype), (dest), (tag), (comm)))

    #define MPI_Wait(request, status) (PetscAddLogDouble(&petsc_wait_ct, &petsc_wait_ct_th, 1) || PetscAddLogDouble(&petsc_sum_of_waits_ct, &petsc_sum_of_waits_ct_th, 1) || MPI_Wait((request), (status)))

    #define MPI_Waitany(a, b, c, d) (PetscAddLogDouble(&petsc_wait_any_ct, &petsc_wait_any_ct_th, 1) || PetscAddLogDouble(&petsc_sum_of_waits_ct, &petsc_sum_of_waits_ct_th, 1) || MPI_Waitany((a), (b), (c), (d)))

    #define MPI_Waitall(count, array_of_requests, array_of_statuses) \
      (PetscAddLogDouble(&petsc_wait_all_ct, &petsc_wait_all_ct_th, 1) || PetscAddLogDouble(&petsc_sum_of_waits_ct, &petsc_sum_of_waits_ct_th, count) || MPI_Waitall((count), (array_of_requests), (array_of_statuses)))

    #define MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm) (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || MPI_Allreduce((sendbuf), (recvbuf), (count), (datatype), (op), (comm)))

    #define MPI_Bcast(buffer, count, datatype, root, comm) (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || MPI_Bcast((buffer), (count), (datatype), (root), (comm)))

    #define MPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || MPI_Reduce_scatter_block((sendbuf), (recvbuf), (recvcount), (datatype), (op), (comm)))

    #define MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Alltoall((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (comm)))

    #define MPI_Alltoallv(sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || PetscMPITypeSizeComm((comm), (sendcnts), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Alltoallv((sendbuf), (sendcnts), (sdispls), (sendtype), (recvbuf), (recvcnts), (rdispls), (recvtype), (comm)))

    #define MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, PetscMPIParallelComm(comm)) || MPI_Allgather((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (comm)))

    #define MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcount, displs, recvtype, comm) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, PetscMPIParallelComm(comm)) || MPI_Allgatherv((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (displs), (recvtype), (comm)))

    #define MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, 1) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Gather((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm)))

    #define MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcount, displs, recvtype, root, comm) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, 1) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Gatherv((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (displs), (recvtype), (root), (comm)))

    #define MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm) \
      (PetscAddLogDouble(&petsc_scatter_ct, &petsc_scatter_ct_th, 1) || PetscMPITypeSize((recvcount), (recvtype), (&petsc_recv_len), &(petsc_recv_len_th)) || MPI_Scatter((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm)))

    #define MPI_Scatterv(sendbuf, sendcount, displs, sendtype, recvbuf, recvcount, recvtype, root, comm) \
      (PetscAddLogDouble(&petsc_scatter_ct, &petsc_scatter_ct_th, 1) || PetscMPITypeSize((recvcount), (recvtype), (&petsc_recv_len), &(petsc_recv_len_th)) || MPI_Scatterv((sendbuf), (sendcount), (displs), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm)))

    #define MPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Ialltoall((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (comm), (request)))

    #define MPI_Ialltoallv(sendbuf, sendcnts, sdispls, sendtype, recvbuf, recvcnts, rdispls, recvtype, comm, request) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || PetscMPITypeSizeComm((comm), (sendcnts), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Ialltoallv((sendbuf), (sendcnts), (sdispls), (sendtype), (recvbuf), (recvcnts), (rdispls), (recvtype), (comm), (request)))

    #define MPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, PetscMPIParallelComm(comm)) || MPI_Iallgather((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (comm), (request)))

    #define MPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcount, displs, recvtype, comm, request) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, PetscMPIParallelComm(comm)) || MPI_Iallgatherv((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (displs), (recvtype), (comm), (request)))

    #define MPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, 1) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Igather((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm), (request)))

    #define MPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcount, displs, recvtype, root, comm, request) \
      (PetscAddLogDouble(&petsc_gather_ct, &petsc_gather_ct_th, 1) || PetscMPITypeSize((sendcount), (sendtype), (&petsc_send_len), (&petsc_send_len_th)) || MPI_Igatherv((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (displs), (recvtype), (root), (comm), (request)))

    #define MPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request) \
      (PetscAddLogDouble(&petsc_scatter_ct, &petsc_scatter_ct_th, 1) || PetscMPITypeSize((recvcount), (recvtype), (&petsc_recv_len), (&petsc_recv_len_th)) || MPI_Iscatter((sendbuf), (sendcount), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm), (request)))

    #define MPI_Iscatterv(sendbuf, sendcount, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request) \
      (PetscAddLogDouble(&petsc_scatter_ct, &petsc_scatter_ct_th, 1) || PetscMPITypeSize((recvcount), (recvtype), (&petsc_recv_len), (&petsc_recv_len_th)) || MPI_Iscatterv((sendbuf), (sendcount), (displs), (sendtype), (recvbuf), (recvcount), (recvtype), (root), (comm), (request)))

    #define MPIX_Send_enqueue(buf, count, datatype, dest, tag, comm) \
      (PetscAddLogDouble(&petsc_send_ct, &petsc_send_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_send_len), (&petsc_send_len_th)) || MPIX_Send_enqueue((buf), (count), (datatype), (dest), (tag), (comm)))

    #define MPIX_Recv_enqueue(buf, count, datatype, source, tag, comm, status) \
      (PetscAddLogDouble(&petsc_recv_ct, &petsc_recv_ct_th, 1) || PetscMPITypeSize((count), (datatype), (&petsc_recv_len), (&petsc_recv_len_th)) || MPIX_Recv_enqueue((buf), (count), (datatype), (source), (tag), (comm), (status)))

    #define MPIX_Isend_enqueue(buf, count, datatype, dest, tag, comm, request) \
      (PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_isend_len), &(petsc_isend_len_th)) || MPIX_Isend_enqueue((buf), (count), (datatype), (dest), (tag), (comm), (request)))

    #define MPIX_Irecv_enqueue(buf, count, datatype, source, tag, comm, request) \
      (PetscAddLogDouble(&petsc_irecv_ct, &petsc_irecv_ct_th, 1) || PetscMPITypeSize((count), (datatype), &(petsc_irecv_len), &(petsc_irecv_len_th)) || MPIX_Irecv_enqueue((buf), (count), (datatype), (source), (tag), (comm), (request)))

    #define MPIX_Allreduce_enqueue(sendbuf, recvbuf, count, datatype, op, comm) \
      (PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, PetscMPIParallelComm(comm)) || MPIX_Allreduce_enqueue((sendbuf), (recvbuf), (count), (datatype), (op), (comm)))

    #define MPIX_Wait_enqueue(request, status) (PetscAddLogDouble(&petsc_wait_ct, &petsc_wait_ct_th, 1) || PetscAddLogDouble(&petsc_sum_of_waits_ct, &petsc_sum_of_waits_ct_th, 1) || MPIX_Wait_enqueue((request), (status)))

    #define MPIX_Waitall_enqueue(count, array_of_requests, array_of_statuses) \
      (PetscAddLogDouble(&petsc_wait_all_ct, &petsc_wait_all_ct_th, 1) || PetscAddLogDouble(&petsc_sum_of_waits_ct, &petsc_sum_of_waits_ct_th, count) || MPIX_Waitall_enqueue((count), (array_of_requests), (array_of_statuses)))
  #else

    #define MPI_Startall_irecv(count, datatype, number, requests) ((number) && MPI_Startall((number), (requests)))

    #define MPI_Startall_isend(count, datatype, number, requests) ((number) && MPI_Startall((number), (requests)))

    #define MPI_Start_isend(count, datatype, requests) (MPI_Start((requests)))

  #endif /* !MPIUNI_H && ! PETSC_HAVE_BROKEN_RECURSIVE_MACRO */

#else /* ---Logging is turned off --------------------------------------------*/

  #define PetscLogMemory PETSC_FALSE

  #define PetscLogHandlerStart(a) PETSC_SUCCESS
  #define PetscLogHandlerStop(a) PETSC_SUCCESS

  #define PetscLogFlops(n) ((void)(n), PETSC_SUCCESS)
  #define PetscGetFlops(a) (*(a) = 0.0, PETSC_SUCCESS)

  #define PetscLogStageRegister(a, b)   PETSC_SUCCESS
  #define PetscLogStagePush(a)          PETSC_SUCCESS
  #define PetscLogStagePop()            PETSC_SUCCESS
  #define PetscLogStageSetActive(a, b)  PETSC_SUCCESS
  #define PetscLogStageGetActive(a, b)  PETSC_SUCCESS
  #define PetscLogStageGetVisible(a, b) PETSC_SUCCESS
  #define PetscLogStageSetVisible(a, b) PETSC_SUCCESS
  #define PetscLogStageGetId(a, b)      (*(b) = 0, PETSC_SUCCESS)
  #define PetscLogStageGetName(a, b)    (*(b) = 0, PETSC_SUCCESS)

  #define PetscLogEventRegister(a, b, c)    PETSC_SUCCESS
  #define PetscLogEventSetCollective(a, b)  PETSC_SUCCESS
  #define PetscLogEventIncludeClass(a)      PETSC_SUCCESS
  #define PetscLogEventExcludeClass(a)      PETSC_SUCCESS
  #define PetscLogEventActivate(a)          PETSC_SUCCESS
  #define PetscLogEventDeactivate(a)        PETSC_SUCCESS
  #define PetscLogEventDeactivatePush(a)    PETSC_SUCCESS
  #define PetscLogEventDeactivatePop(a)     PETSC_SUCCESS
  #define PetscLogEventsPause()             PETSC_SUCCESS
  #define PetscLogEventsUnpause()           PETSC_SUCCESS
  #define PetscLogEventActivateClass(a)     PETSC_SUCCESS
  #define PetscLogEventDeactivateClass(a)   PETSC_SUCCESS
  #define PetscLogEventSetActiveAll(a, b)   PETSC_SUCCESS
  #define PetscLogEventGetId(a, b)          (*(b) = 0, PETSC_SUCCESS)
  #define PetscLogEventGetName(a, b)        (*(b) = 0, PETSC_SUCCESS)
  #define PetscLogEventGetPerfInfo(a, b, c) PETSC_SUCCESS
  #define PetscLogEventSetDof(a, b, c)      PETSC_SUCCESS
  #define PetscLogEventSetError(a, b, c)    PETSC_SUCCESS

  #define PetscLogClassGetId(a, b)   (*(b) = 0, PETSC_SUCCESS)
  #define PetscLogClassGetName(a, b) (*(b) = 0, PETSC_SUCCESS)

  #define PetscLogObjectParents(p, n, c) PETSC_SUCCESS
  #define PetscLogObjectCreate(h)        PETSC_SUCCESS
  #define PetscLogObjectDestroy(h)       PETSC_SUCCESS

  #if defined(PETSC_HAVE_MPE)
    #define PetscLogMPEBegin() PETSC_SUCCESS
    #define PetscLogMPEDump(c) PETSC_SUCCESS
  #endif
  #define PetscLogDefaultBegin()     PETSC_SUCCESS
  #define PetscLogAllBegin()         PETSC_SUCCESS
  #define PetscLogNestedBegin()      PETSC_SUCCESS
  #define PetscLogTraceBegin(file)   PETSC_SUCCESS
  #define PetscLogActions(a)         PETSC_SUCCESS
  #define PetscLogObjects(a)         PETSC_SUCCESS
  #define PetscLogSetThreshold(a, b) PETSC_SUCCESS
  #define PetscLogSet(lb, le)        PETSC_SUCCESS
  #define PetscLogIsActive(flag)     (*(flag) = PETSC_FALSE, PETSC_SUCCESS)

  #define PetscLogView(viewer)      PETSC_SUCCESS
  #define PetscLogViewFromOptions() PETSC_SUCCESS
  #define PetscLogDump(c)           PETSC_SUCCESS

  #define PetscLogEventSync(e, comm)                            PETSC_SUCCESS
  #define PetscLogEventBegin(e, o1, o2, o3, o4)                 PETSC_SUCCESS
  #define PetscLogEventEnd(e, o1, o2, o3, o4)                   PETSC_SUCCESS

  /* If PETSC_USE_LOG is NOT defined, these still need to be! */
  #define MPI_Startall_irecv(count, datatype, number, requests) ((number) && MPI_Startall(number, requests))
  #define MPI_Startall_isend(count, datatype, number, requests) ((number) && MPI_Startall(number, requests))
  #define MPI_Start_isend(count, datatype, requests)            MPI_Start(requests)

#endif /* PETSC_USE_LOG */

PETSC_EXTERN PetscErrorCode PetscLogObjectState(PetscObject, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2, 3);

#define PetscPreLoadBegin(flag, name) \
  do { \
    PetscBool     PetscPreLoading = flag; \
    int           PetscPreLoadMax, PetscPreLoadIt; \
    PetscLogStage _stageNum; \
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-preload", &PetscPreLoading, NULL)); \
    PetscPreLoadMax     = (int)(PetscPreLoading); \
    PetscPreLoadingUsed = PetscPreLoading ? PETSC_TRUE : PetscPreLoadingUsed; \
    for (PetscPreLoadIt = 0; PetscPreLoadIt <= PetscPreLoadMax; PetscPreLoadIt++) { \
      PetscPreLoadingOn = PetscPreLoading; \
      PetscCall(PetscBarrier(NULL)); \
      if (PetscPreLoadIt > 0) PetscCall(PetscLogStageGetId(name, &_stageNum)); \
      else PetscCall(PetscLogStageRegister(name, &_stageNum)); \
      PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(!PetscPreLoadMax || PetscPreLoadIt))); \
      PetscCall(PetscLogStagePush(_stageNum));

#define PetscPreLoadEnd() \
  PetscCall(PetscLogStagePop()); \
  PetscPreLoading = PETSC_FALSE; \
  } \
  } \
  while (0)

#define PetscPreLoadStage(name) \
  do { \
    PetscCall(PetscLogStagePop()); \
    if (PetscPreLoadIt > 0) PetscCall(PetscLogStageGetId(name, &_stageNum)); \
    else PetscCall(PetscLogStageRegister(name, &_stageNum)); \
    PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(!PetscPreLoadMax || PetscPreLoadIt))); \
    PetscCall(PetscLogStagePush(_stageNum)); \
  } while (0)

/* some vars for logging */
PETSC_EXTERN PetscBool PetscPreLoadingUsed; /* true if we are or have done preloading */
PETSC_EXTERN PetscBool PetscPreLoadingOn;   /* true if we are currently in a preloading calculation */

#if defined(PETSC_USE_LOG) && defined(PETSC_HAVE_DEVICE)

/* Global GPU counters */
PETSC_EXTERN PetscLogDouble petsc_ctog_ct;
PETSC_EXTERN PetscLogDouble petsc_gtoc_ct;
PETSC_EXTERN PetscLogDouble petsc_ctog_sz;
PETSC_EXTERN PetscLogDouble petsc_gtoc_sz;
PETSC_EXTERN PetscLogDouble petsc_ctog_ct_scalar;
PETSC_EXTERN PetscLogDouble petsc_gtoc_ct_scalar;
PETSC_EXTERN PetscLogDouble petsc_ctog_sz_scalar;
PETSC_EXTERN PetscLogDouble petsc_gtoc_sz_scalar;
PETSC_EXTERN PetscLogDouble petsc_gflops;
PETSC_EXTERN PetscLogDouble petsc_gtime;

/* Thread local storage */
PETSC_EXTERN_TLS PetscLogDouble petsc_ctog_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gtoc_ct_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_ctog_sz_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gtoc_sz_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_ctog_ct_scalar_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gtoc_ct_scalar_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_ctog_sz_scalar_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gtoc_sz_scalar_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gflops_th;
PETSC_EXTERN_TLS PetscLogDouble petsc_gtime_th;

PETSC_EXTERN PetscErrorCode PetscLogGpuTime(void);
PETSC_EXTERN PetscErrorCode PetscLogGpuTimeBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogGpuTimeEnd(void);

/*@C
       PetscLogGpuFlops - Log how many flops are performed in a calculation on the device

   Input Parameter:
.   flops - the number of flops

   Level: intermediate

   Notes:
     To limit the chance of integer overflow when multiplying by a constant, represent the constant as a double,
     not an integer. Use `PetscLogFlops`(4.0*n) not `PetscLogFlops`(4*n)

     The values are also added to the total flop count for the MPI rank that is set with `PetscLogFlops()`; hence the number of flops
     just on the CPU would be the value from set from `PetscLogFlops()` minus the value set from `PetscLogGpuFlops()`

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogFlops()`, `PetscLogGpuTimeBegin()`, `PetscLogGpuTimeEnd()`
@*/
static inline PetscErrorCode PetscLogGpuFlops(PetscLogDouble n)
{
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot log negative flops");
  PetscCall(PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th, PETSC_FLOPS_PER_OP * n));
  PetscCall(PetscAddLogDouble(&petsc_gflops, &petsc_gflops_th, PETSC_FLOPS_PER_OP * n));
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLogGpuTimeAdd(PetscLogDouble t)
{
  return PetscAddLogDouble(&petsc_gtime, &petsc_gtime_th, t);
}

static inline PetscErrorCode PetscLogCpuToGpu(PetscLogDouble size)
{
  return PetscAddLogDoubleCnt(&petsc_ctog_ct, &petsc_ctog_sz, &petsc_ctog_ct_th, &petsc_ctog_sz_th, size);
}

static inline PetscErrorCode PetscLogGpuToCpu(PetscLogDouble size)
{
  return PetscAddLogDoubleCnt(&petsc_gtoc_ct, &petsc_gtoc_sz, &petsc_gtoc_ct_th, &petsc_gtoc_sz_th, size);
}

static inline PetscErrorCode PetscLogCpuToGpuScalar(PetscLogDouble size)
{
  return PetscAddLogDoubleCnt(&petsc_ctog_ct_scalar, &petsc_ctog_sz_scalar, &petsc_ctog_ct_scalar_th, &petsc_ctog_sz_scalar_th, size);
}

static inline PetscErrorCode PetscLogGpuToCpuScalar(PetscLogDouble size)
{
  return PetscAddLogDoubleCnt(&petsc_gtoc_ct_scalar, &petsc_gtoc_sz_scalar, &petsc_gtoc_ct_scalar_th, &petsc_gtoc_sz_scalar_th, size);
}
#else

  #define PetscLogCpuToGpu(a)       PETSC_SUCCESS
  #define PetscLogGpuToCpu(a)       PETSC_SUCCESS
  #define PetscLogCpuToGpuScalar(a) PETSC_SUCCESS
  #define PetscLogGpuToCpuScalar(a) PETSC_SUCCESS
  #define PetscLogGpuFlops(a)       PETSC_SUCCESS
  #define PetscLogGpuTime()         PETSC_SUCCESS
  #define PetscLogGpuTimeAdd(a)     PETSC_SUCCESS
  #define PetscLogGpuTimeBegin()    PETSC_SUCCESS
  #define PetscLogGpuTimeEnd()      PETSC_SUCCESS

#endif /* PETSC_USE_LOG && PETSC_HAVE_DEVICE */

/* remove TLS defines */
#undef PETSC_EXTERN_TLS
#undef PETSC_TLS

#endif
