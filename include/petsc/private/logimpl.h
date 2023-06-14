#ifndef PETSC_LOGIMPL_H
#define PETSC_LOGIMPL_H

#include <petsc/private/petscimpl.h>

typedef struct {
  char        *name;       /* The name of this event */
  PetscClassId classid;    /* The class the event is associated with */
  PetscBool    collective; /* Flag this event as collective */
  PetscBool    visible;    /* The flag to print info in summary */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this event */
#endif
#if defined(PETSC_HAVE_MPE)
  int mpe_id_begin; /* MPE IDs that define the event */
  int mpe_id_end;
#endif
} PetscEventRegInfo;

typedef struct {
  char        *name;    /* The class name */
  PetscClassId classid; /* The integer identifying this class */
} PetscClassRegInfo;

typedef struct _PetscStageRegInfo {
  char              *name;     /* The stage name */
  PetscBool          visible;  /* The flag to print info in summary */
} PetscStageRegInfo;

typedef struct {
  PetscClassId   id;           /* The integer identifying this class */
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class; this is completely wrong and should possibly be removed */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects; this is completely wrong and should possibly be removed */
} PetscClassPerfInfo;

#define PETSC_LOG_RESIZABLE_ARRAY(entrytype,containertype) \
  typedef struct _n_##containertype *containertype; \
  struct _n_##containertype { \
    int max_entries; \
    int num_entries; \
    entrytype *array; \
  };
PETSC_LOG_RESIZABLE_ARRAY(PetscEventRegInfo,PetscEventRegLog)
PETSC_LOG_RESIZABLE_ARRAY(PetscClassRegInfo,PetscClassRegLog)
PETSC_LOG_RESIZABLE_ARRAY(PetscStageRegInfo,PetscStageRegLog)
PETSC_LOG_RESIZABLE_ARRAY(PetscClassPerfInfo,PetscClassPerfLog)
PETSC_LOG_RESIZABLE_ARRAY(PetscEventPerfInfo,PetscEventPerfLog)
#undef PETSC_LOG_RESIZABLE_ARRAY

#define PetscLogResizableArrayEnsureSize(ra,new_size,blank_entry) \
  PetscMacroReturnStandard( \
    if ((new_size) > ra->max_entries) { \
      int new_max_entries = 2; \
      int rem_size = PetscMax(0,(new_size) - 1); \
      char *new_array; \
      char **old_array = (char **) &((ra)->array); \
      while (rem_size >>= 1) new_max_entries *= 2; \
      PetscCall(PetscMalloc(new_max_entries * sizeof(*((ra)->array)), &new_array)); \
      PetscCall(PetscMemcpy(new_array, (ra)->array, sizeof(*((ra)->array)) * (ra)->num_entries)); \
      PetscCall(PetscFree((ra)->array)); \
      *old_array = new_array; \
      (ra)->max_entries = new_max_entries; \
    } \
    for (int i = (ra)->num_entries; i < (new_size); i++) (ra)->array[i] = (blank_entry); \
    (ra)->num_entries = (new_size); \
  )

#define PetscLogResizableArrayCreate(ra_p,max_init) \
  PetscMacroReturnStandard( \
    PetscCall(PetscNew(ra_p)); \
    (*(ra_p))->num_entries = 0; \
    (*(ra_p))->max_entries = (max_init); \
    PetscCall(PetscMalloc1((max_init), &((*(ra_p))->array))); \
  )

typedef struct _PetscStageInfo {
  char              *name;     /* The stage name */
  PetscBool          used;     /* The stage was pushed on this processor */
  PetscBool          active;
  PetscEventPerfInfo perfInfo; /* The stage performance information */
  PetscEventPerfLog  eventLog; /* The event information for this stage */
  PetscClassPerfLog  classLog; /* The class information for this stage */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this stage */
#endif
} PetscStageInfo;

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              max_entries;
  int              num_entries;
  PetscStageInfo  *array;
  PetscIntStack    stack;     /* The stack for activated stages */
  int              curStage;  /* The current stage (only used in macros so we don't call PetscIntStackTop) */
};

typedef struct _n_PetscLogRegistry *PetscLogRegistry;
struct _n_PetscLogRegistry {
  PetscEventRegLog events;
  PetscClassRegLog classes;
  PetscStageRegLog stages;
};

PETSC_INTERN PetscLogRegistry petsc_log_registry;
PETSC_INTERN PetscErrorCode PetscLogGetRegistry(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry,const char[],PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry,const char[],PetscClassId,PetscLogStage *);

PETSC_INTERN PetscErrorCode PetscLogGetState(PetscLogState *);
PETSC_INTERN PetscErrorCode PetscLogStateCreate(PetscInt, PetscInt, PetscLogState *);
PETSC_INTERN PetscErrorCode PetscLogStateDestroy(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateStagePush(PetscLogState,PetscLogStage);
PETSC_INTERN PetscErrorCode PetscLogStateStagePop(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState, PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogStateEnsureSize(PetscLogState, PetscInt, PetscInt);

PETSC_INTERN PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *);
PETSC_INTERN PetscErrorCode PetscStageLogGetCurrent(PetscStageLog, PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog, PetscLogStage, PetscEventPerfLog *);
PETSC_INTERN PetscErrorCode PetscLogSet(PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject));

enum {PETSC_LOG_HANDLER_DEFAULT, PETSC_LOG_HANDLER_NESTED};


/* A simple stack */
struct _n_PetscIntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

/* The structure for action logging */
#define CREATE      0
#define DESTROY     1
#define ACTIONBEGIN 2
#define ACTIONEND   3
typedef struct _Action {
  int            action;        /* The type of execution */
  PetscLogEvent  event;         /* The event number */
  PetscClassId   classid;       /* The event class id */
  PetscLogDouble time;          /* The time of occurrence */
  PetscLogDouble flops;         /* The cumulative flops */
  PetscLogDouble mem;           /* The current memory usage */
  PetscLogDouble maxmem;        /* The maximum memory usage */
  int            id1, id2, id3; /* The ids of associated objects */
} Action;

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

/* Action and object logging variables */
PETSC_EXTERN Action   *petsc_actions;
PETSC_EXTERN Object   *petsc_objects;
PETSC_EXTERN PetscBool petsc_logActions;
PETSC_EXTERN PetscBool petsc_logObjects;
PETSC_EXTERN int       petsc_numActions;
PETSC_EXTERN int       petsc_maxActions;
PETSC_EXTERN int       petsc_numObjects;
PETSC_EXTERN int       petsc_maxObjects;
PETSC_EXTERN int       petsc_numObjectsDestroyed;

PETSC_EXTERN FILE          *petsc_tracefile;
PETSC_EXTERN int            petsc_tracelevel;
PETSC_EXTERN const char    *petsc_traceblanks;
PETSC_EXTERN char           petsc_tracespace[128];
PETSC_EXTERN PetscLogDouble petsc_tracetime;

/* Thread-safety internals */

/* SpinLock for shared Log registry variables */
PETSC_INTERN PetscSpinlock PetscLogRegistrySpinLock;

/* SpinLock for shared Log variables */
PETSC_INTERN PetscSpinlock PetscLogSpinLock;

#if defined(PETSC_HAVE_THREADSAFETY)
  #if defined(__cplusplus)
    #define PETSC_TLS thread_local
  #else
    #define PETSC_TLS _Thread_local
  #endif
  #define PETSC_INTERN_TLS extern PETSC_TLS PETSC_VISIBILITY_INTERNAL

/* Access PETSc internal thread id */
PETSC_INTERN PetscInt PetscLogGetTid(void);

  /* Map from (threadid,stage,event) to perfInfo data struct */
  #include <petsc/private/hashmapijk.h>

PETSC_HASH_MAP(HMapEvent, PetscHashIJKKey, PetscEventPerfInfo *, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, NULL)

PETSC_INTERN PetscHMapEvent eventInfoMap_th;

#else
  #define PETSC_TLS
  #define PETSC_INTERN_TLS
#endif

#ifdef PETSC_USE_LOG

PETSC_EXTERN PetscErrorCode PetscIntStackCreate(PetscIntStack *);
PETSC_EXTERN PetscErrorCode PetscIntStackDestroy(PetscIntStack);
PETSC_EXTERN PetscErrorCode PetscIntStackPush(PetscIntStack, int);
PETSC_EXTERN PetscErrorCode PetscIntStackPop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackTop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackEmpty(PetscIntStack, PetscBool *);

PETSC_INTERN PetscErrorCode PetscStageRegLogCreate(PetscStageRegLog *);
PETSC_INTERN PetscErrorCode PetscStageRegLogDestroy(PetscStageRegLog);
PETSC_INTERN PetscErrorCode PetscStageRegLogEnsureSize(PetscStageRegLog, int);
PETSC_INTERN PetscErrorCode PetscStageRegLogInsert(PetscStageRegLog, const char[], int *);

/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog);
/* General functions */
PETSC_EXTERN PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog, int);
PETSC_EXTERN PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *);
PETSC_EXTERN PetscErrorCode PetscEventPerfInfoCopy(const PetscEventPerfInfo *, PetscEventPerfInfo *);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog, const char[], PetscClassId, PetscLogEvent *);
/* Query functions */
PETSC_EXTERN PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool *);
/* Activaton functions */
PETSC_EXTERN PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);

/* Logging functions */
PETSC_EXTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *);
PETSC_EXTERN PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog);
PETSC_EXTERN PetscErrorCode PetscClassPerfLogCreate(PetscClassPerfLog *);
PETSC_EXTERN PetscErrorCode PetscClassPerfLogDestroy(PetscClassPerfLog);
PETSC_EXTERN PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *);
/* General functions */
PETSC_EXTERN PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog, int);
PETSC_EXTERN PetscErrorCode PetscClassPerfInfoClear(PetscClassPerfInfo *);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog, const char[], PetscClassId);
/* Query functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog, PetscClassId, int *);
/* Logging functions */
PETSC_EXTERN PetscErrorCode PetscLogObjCreateDefault(PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscStageLogCreate(PetscStageLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogDestroy(PetscStageLog);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscStageLogRegister(PetscStageLog, const char[], int *);
/* Runtime functions */
PETSC_EXTERN PetscErrorCode PetscStageLogPush(PetscStageLog, int);
PETSC_EXTERN PetscErrorCode PetscStageLogPop(PetscStageLog);
PETSC_EXTERN PetscErrorCode PetscStageLogSetActive(PetscStageLog, int, PetscBool);
PETSC_EXTERN PetscErrorCode PetscStageLogGetActive(PetscStageLog, int, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStageLogSetVisible(PetscStageLog, int, PetscBool);
PETSC_EXTERN PetscErrorCode PetscStageLogGetVisible(PetscStageLog, int, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetStage(PetscStageLog, const char[], PetscLogStage *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);

PETSC_EXTERN PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog, const char[], PetscLogEvent *);

PETSC_EXTERN PetscErrorCode PetscLogGetEventLog(PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscLogGetClassLog(PetscClassRegLog *);

PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogNestedEnd(void);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);

PETSC_INTERN PetscErrorCode PetscLogGetCurrentEvent_Internal(PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscLogEventPause_Internal(PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscLogEventResume_Internal(PetscLogEvent);

  #if defined(PETSC_HAVE_DEVICE)
PETSC_EXTERN PetscBool PetscLogGpuTimeFlag;
  #endif
#else /* PETSC_USE_LOG */
  #define PetscLogGetCurrentEvent_Internal(event) ((*(event) = PETSC_DECIDE), PETSC_SUCCESS)
  #define PetscLogEventPause_Internal(event)      PETSC_SUCCESS
  #define PetscLogEventResume_Internal(event)     PETSC_SUCCESS
#endif /* PETSC_USE_LOG */
static inline PetscErrorCode PetscLogPauseCurrentEvent_Internal(PetscLogEvent *event)
{
  PetscFunctionBegin;
  PetscValidIntPointer(event, 1);
  PetscCall(PetscLogGetCurrentEvent_Internal(event));
  PetscCall(PetscLogEventPause_Internal(*event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_LOG_VIEW_FROM_OPTIONS_MAX 4
#endif /* PETSC_LOGIMPL_H */
