/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Bas van 't Hof, Koos Huijssen, Christiaan M. Klaij                    *
 *************************************************************************************
 *    content: Support for nested PetscTimers                                        *
 *************************************************************************************/
#include <petsclog.h> /*I "petsclog.h" I*/
#include <petsc/private/logimpl.h>
#include <petsctime.h>
#include <petscviewer.h>
#include "../src/sys/logging/xmlviewer.h"
#include <../src/sys/logging/plog.h>

#if defined(PETSC_USE_LOG)

/*
 * Support for nested PetscTimers
 *
 * PetscTimers keep track of a lot of useful information: Wall clock times,
 * message passing statistics, flop counts.  Information about the nested structure
 * of the timers is lost. Example:
 *
 * 7:30   Start: awake
 * 7:30      Start: morning routine
 * 7:40         Start: eat
 * 7:49         Done:  eat
 * 7:43      Done:  morning routine
 * 8:15      Start: work
 * 12:15        Start: eat
 * 12:45        Done:  eat
 * 16:00     Done:  work
 * 16:30     Start: evening routine
 * 18:30        Start: eat
 * 19:15        Done:  eat
 * 22:00     Done:  evening routine
 * 22:00  Done:  awake
 *
 * Petsc timers provide the following timer results:
 *
 *    awake:              1 call    14:30 hours
 *    morning routine:    1 call     0:13 hours
 *    eat:                3 calls    1:24 hours
 *    work:               1 call     7:45 hours
 *    evening routine     1 call     5:30 hours
 *
 * Nested timers can be used to get the following table:
 *
 *   [1 call]: awake                14:30 hours
 *   [1 call]:    morning routine         0:13 hours         ( 2 % of awake)
 *   [1 call]:       eat                       0:09 hours         (69 % of morning routine)
 *                   rest (morning routine)    0:04 hours         (31 % of morning routine)
 *   [1 call]:    work                    7:45 hours         (53 % of awake)
 *   [1 call]:       eat                       0:30 hours         ( 6 % of work)
 *                   rest (work)               7:15 hours         (94 % of work)
 *   [1 call]:    evening routine         5:30 hours         (38 % of awake)
 *   [1 call]:       eat                       0:45 hours         (14 % of evening routine)
 *                   rest (evening routine)    4:45 hours         (86 % of morning routine)
 *
 * In default logging, events and stages both accumulate performance statistics, the
 * only difference is that in PetscLogView_Default() the stages are used to aggregate
 * event timers separately.  In nested logging, events and stages are treated almost identically.
 *
 */

/*
 * Data structures for keeping track of nested timers:
 *
 *   nestedEvents: information about the timers that have actually been activated
 *   dftParentActive: if a timer is started now, it is part of (nested inside) the dftParentActive
 *
 * The Default-timers are used to time the nested timers. Every nested timer corresponds to
 * (one or more) default timers, where one of the default timers has the same event-id as the
 * nested one.
 *
 * Because of the risk of confusion between nested timer ids and default timer ids, we
 * introduce a typedef for nested events (NestedEventId) and use the existing type PetscLogEvent
 * only for default events. Also, all nested event variables are prepended with 'nst', and
 * default timers with 'dft'.
 */

  #define DFT_ID_AWAKE    -1
  #define MAINSTAGE_EVENT -2

typedef PetscLogEvent NestedEventId;
typedef struct {
  NestedEventId  nstEvent;         /* event-code for this nested event, argument 'event' in PetscLogEventStartNested */
  int            nParents;         /* number of 'dftParents': the default timer which was the dftParentActive when this nested timer was activated */
  PetscLogEvent *dftParentsSorted; /* The default timers which were the dftParentActive when this nested event was started */
  PetscLogEvent *dftEvents;        /* The default timers which represent the different 'instances' of this nested event */

  PetscLogEvent *dftParents;      /* The default timers which were the dftParentActive when this nested event was started */
  PetscLogEvent *dftEventsSorted; /* The default timers which represent the different 'instances' of this nested event */
} PetscNestedEvent;

static PetscLogEvent     dftParentActive             = DFT_ID_AWAKE;
static int               nNestedEvents               = 0;
static int               nNestedEventsAllocated      = 0;
static PetscNestedEvent *nestedEvents                = NULL;
static PetscLogDouble    thresholdTime               = 0.01; /* initial value was 0.1 */
static size_t            num_stages_allocated        = 0;
static size_t            num_events_allocated        = 0;
static NestedEventId    *nested_stage_to_root_stage  = NULL;
static NestedEventId    *nested_event_to_root_event  = NULL;

static PetscErrorCode NestedStageToRootStage(NestedEventId id, NestedEventId *root_id)
{
  PetscStageLog stageLog;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  if (!nested_stage_to_root_stage) {
    PetscCall(PetscMalloc1(stageLog->numStages, &nested_stage_to_root_stage));
    for (int i = 0; i < stageLog->numStages; i++) nested_stage_to_root_stage[i] = i;
    num_stages_allocated = stageLog->numStages;
  }
  if (id < num_stages_allocated) {
    *root_id = nested_stage_to_root_stage[id];
  } else {
    *root_id = id;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NestedStageSetRootStage(NestedEventId id, NestedEventId root_id)
{
  PetscFunctionBegin;
  if (id >= num_stages_allocated) {
    size_t         new_num_stages_allocated = (id + 1) * 2;
    NestedEventId *new_table;

    PetscCall(PetscMalloc1(new_num_stages_allocated, &new_table));
    PetscCall(PetscArraycpy(new_table, nested_stage_to_root_stage, num_stages_allocated));
    PetscCall(PetscFree(nested_stage_to_root_stage));
    for (int i = num_stages_allocated; i < new_num_stages_allocated; i++) new_table[i] = i;
    nested_stage_to_root_stage = new_table;
    num_stages_allocated        = new_num_stages_allocated;
  }
  nested_stage_to_root_stage[id] = root_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NestedEventToRootEvent(NestedEventId id, NestedEventId *root_id)
{
  PetscStageLog eventLog;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&eventLog));
  if (!nested_event_to_root_event) {
    PetscCall(PetscMalloc1(eventLog->eventLog->numEvents, &nested_event_to_root_event));
    for (int i = 0; i < eventLog->eventLog->numEvents; i++) nested_event_to_root_event[i] = i;
    num_events_allocated = eventLog->eventLog->numEvents;
  }
  if (id < num_events_allocated) {
    *root_id = nested_event_to_root_event[id];
  } else {
    *root_id = id;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NestedEventSetRootEvent(NestedEventId id, NestedEventId root_id)
{
  PetscFunctionBegin;
  if (id >= num_events_allocated) {
    size_t         new_num_events_allocated = (id + 1) * 2;
    NestedEventId *new_table;

    PetscCall(PetscMalloc1(new_num_events_allocated, &new_table));
    PetscCall(PetscArraycpy(new_table, nested_event_to_root_event, num_events_allocated));
    PetscCall(PetscFree(nested_event_to_root_event));
    for (int i = num_events_allocated; i < new_num_events_allocated; i++) new_table[i] = i;
    nested_event_to_root_event = new_table;
    num_events_allocated        = new_num_events_allocated;
  }
  nested_event_to_root_event[id] = root_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define THRESHOLD (thresholdTime / 100.0 + 1e-12)

static PetscErrorCode       PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
static PetscErrorCode       PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
static PetscErrorCode       PetscLogStageBeginHandler_Nested(PetscStageLog);
static PetscErrorCode       PetscLogStageEndHandler_Nested(PetscStageLog);
PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);
static PetscClassId         LogNestedEvent = -1;

/*@C
  PetscLogNestedBegin - Turns on nested logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.

  Logically Collective over `PETSC_COMM_WORLD`

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogNestedBegin(void)
{
  PetscFunctionBegin;
  PetscCheck(!nestedEvents, PETSC_COMM_SELF, PETSC_ERR_COR, "nestedEvents already allocated");

  nNestedEventsAllocated = 10;
  PetscCall(PetscMalloc1(nNestedEventsAllocated, &nestedEvents));
  dftParentActive = DFT_ID_AWAKE;
  nNestedEvents   = 0;

  PetscCall(PetscLogSet(PetscLogEventBeginNested, PetscLogEventEndNested));
  PetscCall(PetscLogStageSet(PetscLogStageBeginHandler_Nested, PetscLogStageEndHandler_Nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Delete the data structures for the nested timers */
PetscErrorCode PetscLogNestedEnd(void)
{
  int i;

  PetscFunctionBegin;
  if (!nestedEvents) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < nNestedEvents; i++) PetscCall(PetscFree4(nestedEvents[i].dftParentsSorted, nestedEvents[i].dftEventsSorted, nestedEvents[i].dftParents, nestedEvents[i].dftEvents));
  PetscCall(PetscFree(nestedEvents));
  nestedEvents           = NULL;
  nNestedEvents          = 0;
  nNestedEventsAllocated = 0;
  num_stages_allocated   = 0;
  num_events_allocated   = 0;
  PetscCall(PetscFree(nested_stage_to_root_stage));
  PetscCall(PetscFree(nested_event_to_root_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 UTILITIES: FIND STUFF IN SORTED ARRAYS

    dftIndex - index to be found
    dftArray - sorted array of PetscLogEvent-ids
    narray - dimension of dftArray
    entry - entry in the array where dftIndex may be found;

     if dftArray[entry] != dftIndex, then dftIndex is not part of dftArray
     In that case, the dftIndex can be inserted at this entry.
*/
static PetscErrorCode PetscLogEventFindDefaultTimer(PetscLogEvent dftIndex, const PetscLogEvent *dftArray, int narray, int *entry)
{
  PetscFunctionBegin;
  if (narray == 0 || dftIndex <= dftArray[0]) {
    *entry = 0;
  } else if (dftIndex > dftArray[narray - 1]) {
    *entry = narray;
  } else {
    int ihigh = narray - 1, ilow = 0;
    while (ihigh > ilow) {
      const int imiddle = (ihigh + ilow) / 2;
      if (dftArray[imiddle] > dftIndex) {
        ihigh = imiddle;
      } else if (dftArray[imiddle] < dftIndex) {
        ilow = imiddle + 1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Utility: find the nested event with given identification

    nstEvent - Nested event to be found
    entry - entry in the nestedEvents where nstEvent may be found;

    if nestedEvents[entry].nstEvent != nstEvent, then index is not part of iarray
*/
static PetscErrorCode PetscLogEventFindNestedTimer(NestedEventId nstEvent, int *entry)
{
  PetscFunctionBegin;
  if (nNestedEvents == 0 || nstEvent <= nestedEvents[0].nstEvent) {
    *entry = 0;
  } else if (nstEvent > nestedEvents[nNestedEvents - 1].nstEvent) {
    *entry = nNestedEvents;
  } else {
    int ihigh = nNestedEvents - 1, ilow = 0;
    while (ihigh > ilow) {
      const int imiddle = (ihigh + ilow) / 2;
      if (nestedEvents[imiddle].nstEvent > nstEvent) {
        ihigh = imiddle;
      } else if (nestedEvents[imiddle].nstEvent < nstEvent) {
        ilow = imiddle + 1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStagePush_Internal(PetscLogStage);
PETSC_INTERN PetscErrorCode PetscLogStagePop_Internal();

/******************************************************************************************/
/* Start a nested event or stage */
static PetscErrorCode PetscLogEventBeginNested_Internal(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, PetscBool is_event)
{
  int           entry, pentry, tentry, i;
  PetscLogEvent dftEvent;

  PetscFunctionBegin;
  PetscCall(PetscLogEventFindNestedTimer(nstEvent, &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != nstEvent) {
    /* Nested event doesn't exist yet: create it */

    if (nNestedEvents == nNestedEventsAllocated) {
      /* Enlarge and re-allocate nestedEvents if needed */
      PetscNestedEvent *tmp = nestedEvents;
      PetscCall(PetscMalloc1(2 * nNestedEvents, &nestedEvents));
      nNestedEventsAllocated *= 2;
      PetscCall(PetscArraycpy(nestedEvents, tmp, nNestedEvents));
      PetscCall(PetscFree(tmp));
    }

    /* Clear space in nestedEvents for new nested event */
    nNestedEvents++;
    for (i = nNestedEvents - 1; i > entry; i--) nestedEvents[i] = nestedEvents[i - 1];

    /* Create event in nestedEvents */
    nestedEvents[entry].nstEvent = nstEvent;
    nestedEvents[entry].nParents = 1;
    PetscCall(PetscMalloc4(1, &nestedEvents[entry].dftParentsSorted, 1, &nestedEvents[entry].dftEventsSorted, 1, &nestedEvents[entry].dftParents, 1, &nestedEvents[entry].dftEvents));

    /* Fill in new event */
    pentry   = 0;
    dftEvent = (PetscLogEvent)nstEvent;

    nestedEvents[entry].nstEvent                 = nstEvent;
    nestedEvents[entry].dftParents[pentry]       = dftParentActive;
    nestedEvents[entry].dftEvents[pentry]        = dftEvent;
    nestedEvents[entry].dftParentsSorted[pentry] = dftParentActive;
    nestedEvents[entry].dftEventsSorted[pentry]  = dftEvent;

  } else {
    /* Nested event exists: find current dftParentActive among parents */
    PetscLogEvent *dftParentsSorted = nestedEvents[entry].dftParentsSorted;
    PetscLogEvent *dftEvents        = nestedEvents[entry].dftEvents;
    int            nParents         = nestedEvents[entry].nParents;

    PetscCall(PetscLogEventFindDefaultTimer(dftParentActive, dftParentsSorted, nParents, &pentry));

    if (pentry >= nParents || dftParentActive != dftParentsSorted[pentry]) {
      /* dftParentActive not in the list: add it to the list */
      PetscStageLog  stage_log;
      int            i, current_stage;
      PetscLogEvent *dftParents      = nestedEvents[entry].dftParents;
      PetscLogEvent *dftEventsSorted = nestedEvents[entry].dftEventsSorted;
      char           name[100];

      /* Register a new default timer */
      if (!is_event) PetscCall(PetscLogStagePop_Internal());
      PetscCall(PetscLogGetStageLog(&stage_log));
      PetscCall(PetscStageLogGetCurrent(stage_log, &current_stage));
      PetscCall(PetscSNPrintf(name, PETSC_STATIC_ARRAY_LENGTH(name), "__Nested %d: %d -> %d", current_stage, (int)dftParentActive, (int)nstEvent));
      if (is_event) {
        PetscCall(PetscLogEventRegister(name, LogNestedEvent, &dftEvent));
        PetscCall(NestedEventSetRootEvent(dftEvent, nstEvent));
      } else {
        PetscCall(PetscLogStageRegister(name, &dftEvent));
        // stop the timer for the default stage and start it for the nested stage
        PetscCall(PetscInfo(NULL, "Swapping stage %d for new stage %d\n", (int)-(nstEvent + 2), dftEvent));
        PetscCall(PetscLogStagePush_Internal(dftEvent));
        PetscCall(NestedStageSetRootStage(dftEvent, -(nstEvent + 2)));
        dftEvent = -(dftEvent + 2);
      }
      PetscCall(PetscLogEventFindDefaultTimer(dftEvent, dftEventsSorted, nParents, &tentry));

      /* Reallocate parents and dftEvents to make space for new parent */
      PetscCall(PetscMalloc4(1 + nParents, &nestedEvents[entry].dftParentsSorted, 1 + nParents, &nestedEvents[entry].dftEventsSorted, 1 + nParents, &nestedEvents[entry].dftParents, 1 + nParents, &nestedEvents[entry].dftEvents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftParentsSorted, dftParentsSorted, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftEventsSorted, dftEventsSorted, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftParents, dftParents, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftEvents, dftEvents, nParents));
      PetscCall(PetscFree4(dftParentsSorted, dftEventsSorted, dftParents, dftEvents));

      dftParents       = nestedEvents[entry].dftParents;
      dftEvents        = nestedEvents[entry].dftEvents;
      dftParentsSorted = nestedEvents[entry].dftParentsSorted;
      dftEventsSorted  = nestedEvents[entry].dftEventsSorted;

      nestedEvents[entry].nParents++;
      nParents++;

      for (i = nParents - 1; i > pentry; i--) {
        dftParentsSorted[i] = dftParentsSorted[i - 1];
        dftEvents[i]        = dftEvents[i - 1];
      }
      for (i = nParents - 1; i > tentry; i--) {
        dftParents[i]      = dftParents[i - 1];
        dftEventsSorted[i] = dftEventsSorted[i - 1];
      }

      /* Fill in the new default timer */
      dftParentsSorted[pentry] = dftParentActive;
      dftEvents[pentry]        = dftEvent;
      dftParents[tentry]       = dftParentActive;
      dftEventsSorted[tentry]  = dftEvent;
    } else {
      /* dftParentActive was found: find the corresponding default 'dftEvent'-timer */
      dftEvent = nestedEvents[entry].dftEvents[pentry];
    }
  }

  /* Start the default 'dftEvent'-timer and update the dftParentActive */
  if (is_event) PetscCall(PetscLogEventBeginDefault(dftEvent, t, o1, o2, o3, o4));

  dftParentActive = dftEvent;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/******************************************************************************************/
/* Start a nested event */
static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBeginNested_Internal(nstEvent, t, o1, o2, o3, o4, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/******************************************************************************************/
/* Start a nested stage */
static PetscErrorCode PetscLogStageBeginHandler_Nested(PetscStageLog stage_log)
{
  PetscInt stage_id = stage_log->stack->stack[stage_log->stack->top];

  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Pushing stage %d\n", (int)stage_id));
  PetscCall(PetscLogEventBeginNested_Internal(-(stage_id + 2), 0, NULL, NULL, NULL, NULL, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested_Internal(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, PetscBool is_event)
{
  int            entry, pentry, nParents;
  PetscLogEvent *dftEventsSorted;

  PetscFunctionBegin;
  /* Find the nested event */
  PetscCall(PetscLogEventFindNestedTimer(nstEvent, &entry));
  PetscCheck(entry < nNestedEvents, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event %d larger than number of events %d", entry, nNestedEvents);
  PetscCheck(nstEvent < 0 || nestedEvents[entry].nstEvent == nstEvent, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event %d had unbalanced begin/end pairs does not match %d", entry, nstEvent);
  dftEventsSorted = nestedEvents[entry].dftEventsSorted;
  nParents        = nestedEvents[entry].nParents;

  /* Find the current default timer among the 'dftEvents' of this event */
  PetscCall(PetscLogEventFindDefaultTimer(dftParentActive, dftEventsSorted, nParents, &pentry));

  PetscCheck(pentry < nParents, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Entry %d is larger than number of parents %d", pentry, nParents);
  PetscCheck(dftEventsSorted[pentry] == dftParentActive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Active parent is %d, but we seem to be closing %d", dftParentActive, dftEventsSorted[pentry]);

  /* Stop the default timer and update the dftParentActive */
  if (is_event) PetscCall(PetscLogEventEndDefault(dftParentActive, t, o1, o2, o3, o4));
  dftParentActive = nestedEvents[entry].dftParents[pentry];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventEndNested_Internal(nstEvent, t, o1, o2, o3, o4, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStageEndHandler_Nested(PetscStageLog stage_log)
{
  PetscInt stage_id = stage_log->stack->stack[stage_log->stack->top];
  PetscInt root_stage;

  PetscFunctionBegin;
  PetscCall(NestedStageToRootStage(stage_id, &root_stage));
  PetscCall(PetscInfo(NULL, "Popping stage %d\n", (int)root_stage));
  PetscCall(PetscLogEventEndNested_Internal(-(root_stage + 2), 0, NULL, NULL, NULL, NULL, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscLogSetThreshold - Set the threshold time for logging the events; this is a percentage out of 100, so 1. means any event
          that takes 1 or more percent of the time.

  Logically Collective over `PETSC_COMM_WORLD`

  Input Parameter:
.   newThresh - the threshold to use

  Output Parameter:
.   oldThresh - the previously set threshold value

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
      PetscLogSetThreshold(0.1,&oldthresh);
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`,
          `PetscLogNestedBegin()`
@*/
PetscErrorCode PetscLogSetThreshold(PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscFunctionBegin;
  if (oldThresh) *oldThresh = thresholdTime;
  if (newThresh == (PetscLogDouble)PETSC_DECIDE) newThresh = 0.01;
  if (newThresh == (PetscLogDouble)PETSC_DEFAULT) newThresh = 0.01;
  thresholdTime = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintExeSpecs(PetscViewer viewer)
{
  char        arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128];
  char        version[256], buildoptions[128] = "";
  PetscMPIInt size;
  size_t      len;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));

  PetscCall(PetscViewerXMLStartSection(viewer, "runspecification", "Run Specification"));
  PetscCall(PetscViewerXMLPutString(viewer, "executable", "Executable", pname));
  PetscCall(PetscViewerXMLPutString(viewer, "architecture", "Architecture", arch));
  PetscCall(PetscViewerXMLPutString(viewer, "hostname", "Host", hostname));
  PetscCall(PetscViewerXMLPutInt(viewer, "nprocesses", "Number of processes", size));
  PetscCall(PetscViewerXMLPutString(viewer, "user", "Run by user", username));
  PetscCall(PetscViewerXMLPutString(viewer, "date", "Started at", date));
  PetscCall(PetscViewerXMLPutString(viewer, "petscrelease", "Petsc Release", version));

  if (PetscDefined(USE_DEBUG)) PetscCall(PetscStrlcat(buildoptions, "Debug ", sizeof(buildoptions)));
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscStrlcat(buildoptions, "Complex ", sizeof(buildoptions)));
  if (PetscDefined(USE_REAL_SINGLE)) {
    PetscCall(PetscStrlcat(buildoptions, "Single ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FLOAT128)) {
    PetscCall(PetscStrlcat(buildoptions, "Quadruple ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FP16)) {
    PetscCall(PetscStrlcat(buildoptions, "Half ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_64BIT_INDICES)) PetscCall(PetscStrlcat(buildoptions, "Int64 ", sizeof(buildoptions)));
  #if defined(__cplusplus)
  PetscCall(PetscStrlcat(buildoptions, "C++ ", sizeof(buildoptions)));
  #endif
  PetscCall(PetscStrlen(buildoptions, &len));
  if (len) PetscCall(PetscViewerXMLPutString(viewer, "petscbuildoptions", "Petsc build options", buildoptions));
  PetscCall(PetscViewerXMLEndSection(viewer, "runspecification"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLGlobalPerformanceElement(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble local_val, const PetscBool print_average, const PetscBool print_total)
{
  PetscLogDouble min, tot, ratio, avg;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  PetscLogDouble valrank[2], max[2];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  valrank[0] = local_val;
  valrank[1] = (PetscLogDouble)rank;
  PetscCall(MPIU_Allreduce(&local_val, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(valrank, &max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCall(MPIU_Allreduce(&local_val, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max[0] / min;
  else ratio = 0.0;

  PetscCall(PetscViewerXMLStartSection(viewer, name, desc));
  PetscCall(PetscViewerXMLPutDouble(viewer, "max", NULL, max[0], "%e"));
  PetscCall(PetscViewerXMLPutInt(viewer, "maxrank", "rank at which max was found", (PetscMPIInt)max[1]));
  PetscCall(PetscViewerXMLPutDouble(viewer, "ratio", NULL, ratio, "%f"));
  if (print_average) PetscCall(PetscViewerXMLPutDouble(viewer, "average", NULL, avg, "%e"));
  if (print_total) PetscCall(PetscViewerXMLPutDouble(viewer, "total", NULL, tot, "%e"));
  PetscCall(PetscViewerXMLEndSection(viewer, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintGlobalPerformance(PetscViewer viewer, PetscLogDouble locTotalTime)
{
  PetscLogDouble  flops, mem, red, mess;
  const PetscBool print_total_yes = PETSC_TRUE, print_total_no = PETSC_FALSE, print_average_no = PETSC_FALSE, print_average_yes = PETSC_TRUE;

  PetscFunctionBegin;
  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  PetscCall(PetscViewerXMLStartSection(viewer, "globalperformance", "Global performance"));

  /*   Time */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "time", "Time (sec)", locTotalTime, print_average_yes, print_total_no));

  /*   Objects */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "objects", "Objects", (PetscLogDouble)petsc_numObjects, print_average_yes, print_total_no));

  /*   Flop */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflop", "MFlop", petsc_TotalFlops / 1.0E6, print_average_yes, print_total_yes));

  /*   Flop/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops / locTotalTime;
  else flops = 0.0;
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflops", "MFlop/sec", flops / 1.0E6, print_average_yes, print_total_yes));

  /*   Memory */
  PetscCall(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "memory", "Memory (MiB)", mem / 1024.0 / 1024.0, print_average_yes, print_total_yes));
  /*   Messages */
  mess = 0.5 * (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagetransfers", "MPI Message Transfers", mess, print_average_yes, print_total_yes));

  /*   Message Volume */
  mess = 0.5 * (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagevolume", "MPI Message Volume (MiB)", mess / 1024.0 / 1024.0, print_average_yes, print_total_yes));

  /*   Reductions */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "reductions", "MPI Reductions", red, print_average_no, print_total_no));
  PetscCall(PetscViewerXMLEndSection(viewer, "globalperformance"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  const char *name;
  PetscInt id;
  PetscInt parent;
  PetscInt num_descendants;
} PetscNestedEventNode;

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;
  PetscFunctionBegin;

  node = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt num_this_descendants = 0;
      PetscNestedEventNode tmp = tree[start_loc];
      tree[start_loc] = tree[node];
      tree[node] = tmp;
      PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes - start_loc - 1, tree[start_loc].id, &tree[start_loc + 1], &num_this_descendants));
      tree[start_loc].num_descendants = num_this_descendants;
      *num_descendants += 1 + num_this_descendants;
      start_loc += 1 + num_this_descendants;
      node = start_loc;
    } else {
      node++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesFillPerf(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscEventPerfInfo perf[], PetscStageInfo *current_stage, PetscStageLog stage_log, PetscLogGlobalNames global_stages, PetscLogGlobalNames global_events)
{
  PetscInt node;

  PetscFunctionBegin;
  node = 0;
  while (node < num_nodes) {
    PetscAssert(tree[node].parent == parent, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed tree traversal");
    if (tree[node].id < global_stages->count) {
      // this is a stage
      PetscInt stage_id = global_stages->global_to_local[tree[node].id];
      if (stage_id >= 0) { // stage was called locally, skip subtree
        PetscStageInfo *new_current_stage = &stage_log->stageInfo[stage_id];
        PetscCall(PetscLogNestedEventNodesFillPerf(tree[node].num_descendants, tree[node].id, &tree[node + 1], &perf[node + 1], new_current_stage, stage_log, global_stages, global_events));
        if (current_stage) {
          // add back in subtracted values so that stages have timings that are inclusive, like events
          current_stage->perfInfo.time  += new_current_stage->perfInfo.time;
          current_stage->perfInfo.flops += new_current_stage->perfInfo.flops;
          current_stage->perfInfo.numMessages += new_current_stage->perfInfo.numMessages;
          current_stage->perfInfo.messageLength += new_current_stage->perfInfo.messageLength;
          current_stage->perfInfo.numReductions += new_current_stage->perfInfo.numReductions;
        }
        perf[node] = new_current_stage->perfInfo;
      }
    } else {
      // this is an event
      PetscInt event_id = global_events->global_to_local[tree[node].id - global_stages->count];

      if (current_stage && event_id >= 0 && event_id < current_stage->eventLog->numEvents) {
        perf[node] = current_stage->eventLog->eventInfo[event_id];
        PetscCall(PetscLogNestedEventNodesFillPerf(tree[node].num_descendants, tree[node].id, &tree[node + 1], &perf[node + 1], current_stage, stage_log, global_stages, global_events));
      }
    }
    node += 1 + tree[node].num_descendants;
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCreatePerfTree(MPI_Comm comm, PetscStageLog stage_log, PetscLogGlobalNames global_stages, PetscLogGlobalNames global_events, PetscNestedEventNode **tree_p, PetscEventPerfInfo **perf_p)
{
  PetscMPIInt size;
  PetscInt num_nodes = global_stages->count + global_events->count;
  PetscEventPerfInfo *perf;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    tree[node].parent = -1;
  }
  for (PetscInt stage = 0; stage < global_stages->count; stage++) {
    PetscInt node = stage;
    PetscInt stage_id = global_stages->global_to_local[stage];

    tree[node].name = global_stages->names[stage];

    if (stage_id >= 0) {
      PetscNestedEvent *nested_event;
      PetscLogEvent    *dftParents;
      PetscLogEvent     parent_id;
      PetscLogEvent    *dftEventsSorted;
      int               entry;
      int               nParents, tentry;
      NestedEventId     root_id;

      PetscCall(NestedStageToRootStage(stage_id, &root_id));
      PetscCall(PetscLogEventFindNestedTimer(-(root_id+2), &entry));
      if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != -(root_id + 2)) continue;
      nested_event = &nestedEvents[entry];
      dftParents      = nested_event->dftParents;
      dftEventsSorted = nested_event->dftEventsSorted;
      nParents         = nested_event->nParents;
      PetscCall(PetscLogEventFindDefaultTimer(-(stage_id+2), dftEventsSorted, nParents, &tentry));
      PetscAssert(dftEventsSorted[tentry] == -(stage_id+2), PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
      parent_id = dftParents[tentry];
      if (parent_id >= 0) {
        // parent is an event
        tree[node].parent = global_events->local_to_global[parent_id] + global_stages->count;
      } else if (parent_id <= MAINSTAGE_EVENT) {
        // parent is a stage
        tree[node].parent = global_stages->local_to_global[-(parent_id+2)];
      }
    }
  }
  for (PetscInt event = 0; event < global_events->count; event++) {
    PetscInt node = event + global_stages->count;
    PetscInt event_id = global_events->global_to_local[event];

    tree[node].name = global_events->names[event];
    if (event_id >= 0) {
      PetscNestedEvent *nested_event;
      PetscLogEvent    *dftParents;
      PetscLogEvent     parent_id;
      PetscLogEvent    *dftEventsSorted;
      int               entry;
      int               nParents, tentry;
      NestedEventId     root_id;

      PetscCall(NestedEventToRootEvent(event_id, &root_id));
      PetscCall(PetscLogEventFindNestedTimer(root_id, &entry));
      if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != root_id) continue;
      nested_event = &nestedEvents[entry];
      dftParents      = nested_event->dftParents;
      dftEventsSorted = nested_event->dftEventsSorted;
      nParents         = nested_event->nParents;
      PetscCall(PetscLogEventFindDefaultTimer(event_id, dftEventsSorted, nParents, &tentry));
      PetscAssert(dftEventsSorted[tentry] == event_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
      parent_id = dftParents[tentry];
      if (parent_id >= 0) {
        // parent is an event
        tree[node].parent = global_events->local_to_global[parent_id] + global_stages->count;
      } else if (parent_id <= MAINSTAGE_EVENT) {
        // parent is a stage
        tree[node].parent = global_stages->local_to_global[-(parent_id+2)];
      }
    }
  }

  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) { // get missing parents from other processes
    PetscInt *parents;

    PetscCall(PetscMalloc1(num_nodes, &parents));
    for (PetscInt node = 0; node < num_nodes; node++) parents[node] = tree[node].parent;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, parents, num_nodes, MPIU_INT, MPI_MAX, comm));
    for (PetscInt node = 0; node < num_nodes; node++) tree[node].parent = parents[node];
    PetscCall(PetscFree(parents));
  }

  PetscInt num_descendants = 0;

  PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes, -1, tree, &num_descendants));
  PetscAssert(num_descendants == num_nodes, comm, PETSC_ERR_PLIB, "Failed tree ordering invariant");

  PetscCall(PetscCalloc1(num_nodes, &perf));
  PetscCall(PetscLogNestedEventNodesFillPerf(num_nodes, -1, tree, perf, NULL, stage_log, global_stages, global_events));

  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(0);
}

typedef struct {
  PetscLogEvent  dftEvent;
  NestedEventId  nstEvent;
  PetscLogEvent  dftParent;
  NestedEventId  nstParent;
  PetscBool      own;
  int            depth;
  NestedEventId *nstPath;
} PetscNestedEventTree;

typedef struct {
  MPI_Comm            comm;
  PetscStageLog       stage_log;
  PetscLogGlobalNames global_stages;
  PetscLogGlobalNames global_events;
  PetscNestedEventNode *tree;
  PetscEventPerfInfo   *perf;
} PetscNestedEventTreeNew;

static PetscErrorCode NestedEventGetLogStageInfo(const PetscNestedEventTree *tree, PetscStageInfo **info)
{
  PetscFunctionBegin;
  for (PetscInt i = tree->depth - 1; i >= 0; i--) {
    int id = tree->nstPath[i];
    if (id < DFT_ID_AWAKE) {
      int           stage = -(id + 2);
      PetscStageLog stageLog;

      PetscCall(PetscLogGetStageLog(&stageLog));
      *info = &stageLog->stageInfo[stage];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscUnreachable();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compare timers to sort them in the tree */
static int compareTreeItems(const void *item1_, const void *item2_)
{
  int                   i;
  PetscNestedEventTree *item1 = (PetscNestedEventTree *)item1_;
  PetscNestedEventTree *item2 = (PetscNestedEventTree *)item2_;

  for (i = 0; i < PetscMin(item1->depth, item2->depth); i++) {
    if (item1->nstPath[i] < item2->nstPath[i]) return -1;
    if (item1->nstPath[i] > item2->nstPath[i]) return +1;
  }
  if (item1->depth < item2->depth) return -1;
  if (item1->depth > item2->depth) return 1;
  return 0;
}

/*
 * Do MPI communication to get the complete, nested calling tree for all processes: there may be
 * calls that happen in some processes, but not in others.
 *
 * The output, tree[nTimers] is an array of PetscNestedEventTree-structs.
 * The tree is sorted so that the timers can be printed in the order of appearance.
 *
 * For tree-items which appear in the trees of multiple processes (which will be most items), the
 * following rule is followed:
 * + if information from my own process is available, then that is the information stored in tree.
 *   otherwise it is some other process's information.
 */
static PetscErrorCode PetscLogNestedTreeCreate(PetscViewer viewer, PetscNestedEventTree **p_tree, int *p_nTimers)
{
  PetscNestedEventTree *tree = NULL, *newTree;
  int                  *treeIndices, *treeIndices_allocated;
  int                   nTimers, totalNTimers, i, j, iTimer0, maxDefaultTimer, minDefaultTimer;
  int                   yesno;
  PetscBool             done;
  int                   maxdepth;
  int                   depth;
  int                   illegalEvent;
  int                   iextra;
  NestedEventId        *nstPath, *nstMyPath;
  MPI_Comm              comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  /* Calculate memory needed to store everybody's information and allocate tree */
  nTimers = 0;
  for (i = 0; i < nNestedEvents; i++) nTimers += nestedEvents[i].nParents;

  PetscCall(PetscMalloc1(nTimers, &tree));

  /* Fill tree with readily available information */
  iTimer0         = 0;
  maxDefaultTimer = 0;
  minDefaultTimer = 0;
  for (i = 0; i < nNestedEvents; i++) {
    int            nParents         = nestedEvents[i].nParents;
    NestedEventId  nstEvent         = nestedEvents[i].nstEvent;
    PetscLogEvent *dftParentsSorted = nestedEvents[i].dftParentsSorted;
    PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    for (j = 0; j < nParents; j++) {
      maxDefaultTimer = PetscMax(dftEvents[j], maxDefaultTimer);
      minDefaultTimer = PetscMin(dftEvents[j], minDefaultTimer);

      tree[iTimer0 + j].dftEvent  = dftEvents[j];
      tree[iTimer0 + j].nstEvent  = nstEvent;
      tree[iTimer0 + j].dftParent = dftParentsSorted[j];
      tree[iTimer0 + j].own       = PETSC_TRUE;

      tree[iTimer0 + j].nstParent = 0;
      tree[iTimer0 + j].depth     = 0;
      tree[iTimer0 + j].nstPath   = NULL;
    }
    iTimer0 += nParents;
  }

  /* Calculate the global maximum for the default timer index, so array treeIndices can
   * be allocated only once */
  PetscCall(MPIU_Allreduce(&maxDefaultTimer, &j, 1, MPI_INT, MPI_MAX, comm));
  maxDefaultTimer = j;
  PetscCall(MPIU_Allreduce(&minDefaultTimer, &j, 1, MPI_INT, MPI_MIN, comm));
  minDefaultTimer = j;

  /* Find default timer's place in the tree */
  PetscCall(PetscCalloc1(maxDefaultTimer + 1 - PetscMin(minDefaultTimer, 0), &treeIndices_allocated));
  treeIndices    = &treeIndices_allocated[-PetscMin(0, minDefaultTimer)];
  treeIndices[0] = 0;
  for (i = 0; i < nTimers; i++) {
    PetscLogEvent dftEvent = tree[i].dftEvent;
    treeIndices[dftEvent]  = i;
  }

  /* Find each dftParent's nested identification */
  for (i = 0; i < nTimers; i++) {
    PetscLogEvent dftParent = tree[i].dftParent;
    if (dftParent != DFT_ID_AWAKE) {
      int j             = treeIndices[dftParent];
      tree[i].nstParent = tree[j].nstEvent;
    }
  }

  /* Find depths for each timer path */
  done     = PETSC_FALSE;
  maxdepth = 1;
  while (!done) {
    done = PETSC_TRUE;
    for (i = 0; i < nTimers; i++) {
      if (tree[i].dftParent == DFT_ID_AWAKE) {
        tree[i].depth = 1;
        maxdepth      = PetscMax(1, maxdepth);
      } else {
        int j = treeIndices[tree[i].dftParent];
        depth = 1 + tree[j].depth;
        if (depth > tree[i].depth) {
          done          = PETSC_FALSE;
          tree[i].depth = depth;
          maxdepth      = PetscMax(depth, maxdepth);
        }
      }
    }
  }

  /* Allocate the paths in the entire tree */
  for (i = 0; i < nTimers; i++) {
    depth = tree[i].depth;
    PetscCall(PetscCalloc1(depth, &tree[i].nstPath));
  }

  /* Calculate the paths for all timers */
  for (depth = 1; depth <= maxdepth; depth++) {
    for (i = 0; i < nTimers; i++) {
      if (tree[i].depth == depth) {
        if (depth > 1) {
          int j = treeIndices[tree[i].dftParent];
          PetscCall(PetscArraycpy(tree[i].nstPath, tree[j].nstPath, depth - 1));
        }
        tree[i].nstPath[depth - 1] = tree[i].nstEvent;
      }
    }
  }
  PetscCall(PetscFree(treeIndices_allocated));

  /* Sort the tree on basis of the paths */
  qsort(tree, nTimers, sizeof(PetscNestedEventTree), compareTreeItems);

  /* Allocate an array to store paths */
  depth = maxdepth;
  PetscCall(MPIU_Allreduce(&depth, &maxdepth, 1, MPI_INT, MPI_MAX, comm));
  PetscCall(PetscMalloc1(maxdepth + 1, &nstPath));
  PetscCall(PetscMalloc1(maxdepth + 1, &nstMyPath));

  /* Find an illegal nested event index (1+largest nested event index) */
  illegalEvent = 1 + nestedEvents[nNestedEvents - 1].nstEvent;
  i            = illegalEvent;
  PetscCall(MPIU_Allreduce(&i, &illegalEvent, 1, MPI_INT, MPI_MAX, comm));

  /* First, detect timers which are not available in this process, but are available in others
   *        Allocate a new tree, that can contain all timers
   * Then,  fill the new tree with all (own and not-own) timers */
  newTree = NULL;
  for (yesno = 0; yesno <= 1; yesno++) {
    depth  = 1;
    i      = 0;
    iextra = 0;
    while (depth > 0) {
      int       j;
      PetscBool same;

      /* Construct the next path in this process's tree:
       * if necessary, supplement with invalid path entries */
      depth++;
      PetscCheck(depth <= maxdepth + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Depth %d > maxdepth+1 %d", depth, maxdepth + 1);
      if (i < nTimers) {
        for (j = 0; j < tree[i].depth; j++) nstMyPath[j] = tree[i].nstPath[j];
        for (j = tree[i].depth; j < depth; j++) nstMyPath[j] = illegalEvent;
      } else {
        for (j = 0; j < depth; j++) nstMyPath[j] = illegalEvent;
      }

      /* Communicate with other processes to obtain the next path and its depth */
      PetscCall(MPIU_Allreduce(nstMyPath, nstPath, depth, MPI_INT, MPI_MIN, comm));
      for (j = depth - 1; (int)j >= 0; j--) {
        if (nstPath[j] == illegalEvent) depth = j;
      }

      if (depth > 0) {
        /* If the path exists */

        /* check whether the next path is the same as this process's next path */
        same = PETSC_TRUE;
        for (j = 0; same && j < depth; j++) same = (same && nstMyPath[j] == nstPath[j]) ? PETSC_TRUE : PETSC_FALSE;

        if (same) {
          /* Register 'own path' */
          if (newTree) newTree[i + iextra] = tree[i];
          i++;
        } else {
          /* Register 'not an own path' */
          if (newTree) {
            newTree[i + iextra].nstEvent = nstPath[depth - 1];
            newTree[i + iextra].own      = PETSC_FALSE;
            newTree[i + iextra].depth    = depth;
            PetscCall(PetscMalloc1(depth, &newTree[i + iextra].nstPath));
            for (j = 0; j < depth; j++) newTree[i + iextra].nstPath[j] = nstPath[j];

            newTree[i + iextra].dftEvent  = 0;
            newTree[i + iextra].dftParent = 0;
            newTree[i + iextra].nstParent = 0;
          }
          iextra++;
        }
      }
    }

    /* Determine the size of the complete tree (with own and not-own timers) and allocate the new tree */
    totalNTimers = nTimers + iextra;
    if (!newTree) PetscCall(PetscMalloc1(totalNTimers, &newTree));
  }
  PetscCall(PetscFree(nstPath));
  PetscCall(PetscFree(nstMyPath));
  PetscCall(PetscFree(tree));
  tree    = newTree;
  newTree = NULL;

  /* Set return value and return */
  *p_tree    = tree;
  *p_nTimers = totalNTimers;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Delete the nested timer tree
 */
static PetscErrorCode PetscLogNestedTreeDestroy(PetscNestedEventTree *tree, int nTimers)
{
  int i;

  PetscFunctionBegin;
  for (i = 0; i < nTimers; i++) PetscCall(PetscFree(tree[i].nstPath));
  PetscCall(PetscFree(tree));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLNestedLinePerfResults(PetscViewer viewer, const char *name, PetscLogDouble value, PetscLogDouble minthreshold, PetscLogDouble maxthreshold, PetscLogDouble minmaxtreshold)
{
  MPI_Comm       comm; /* MPI communicator in reduction */
  PetscMPIInt    rank; /* rank of this process */
  PetscLogDouble val_in[2], max[2], min[2];
  PetscLogDouble minvalue, maxvalue, tot;
  PetscMPIInt    size;
  PetscMPIInt    minLoc, maxLoc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  val_in[0] = value;
  val_in[1] = (PetscLogDouble)rank;
  PetscCall(MPIU_Allreduce(val_in, max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCall(MPIU_Allreduce(val_in, min, 1, MPIU_2PETSCLOGDOUBLE, MPI_MINLOC, comm));
  maxvalue = max[0];
  maxLoc   = (PetscMPIInt)max[1];
  minvalue = min[0];
  minLoc   = (PetscMPIInt)min[1];
  PetscCall(MPIU_Allreduce(&value, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

  if (maxvalue < maxthreshold && minvalue >= minthreshold) {
    /* One call per parent or NO value: don't print */
  } else {
    PetscCall(PetscViewerXMLStartSection(viewer, name, NULL));
    if (maxvalue > minvalue * minmaxtreshold) {
      PetscCall(PetscViewerXMLPutDouble(viewer, "avgvalue", NULL, tot / size, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "minvalue", NULL, minvalue, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "maxvalue", NULL, maxvalue, "%g"));
      PetscCall(PetscViewerXMLPutInt(viewer, "minloc", NULL, minLoc));
      PetscCall(PetscViewerXMLPutInt(viewer, "maxloc", NULL, maxLoc));
    } else {
      PetscCall(PetscViewerXMLPutDouble(viewer, "value", NULL, tot / size, "%g"));
    }
    PetscCall(PetscViewerXMLEndSection(viewer, name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintLine(PetscViewer viewer, PetscEventPerfInfo perfInfo, PetscLogDouble countsPerCall, int parentCount, int depth, const char *name, PetscLogDouble totalTime, PetscBool *isPrinted)
{
  PetscLogDouble time = perfInfo.time;
  PetscLogDouble timeMx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(MPIU_Allreduce(&time, &timeMx, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  *isPrinted = ((timeMx / totalTime) >= THRESHOLD) ? PETSC_TRUE : PETSC_FALSE;
  if (*isPrinted) {
    PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
    PetscCall(PetscViewerXMLPutString(viewer, "name", NULL, name));
    PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "time", time / totalTime * 100.0, 0, 0, 1.02));
    PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", parentCount > 0 ? countsPerCall : 0, 0.99, 1.01, 1.02));
    PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mflops", time >= timeMx * 0.001 ? 1e-6 * perfInfo.flops / time : 0, 0, 0.01, 1.05));
    PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mbps", time >= timeMx * 0.001 ? perfInfo.messageLength / (1024 * 1024 * time) : 0, 0, 0.01, 1.05));
    PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", time >= timeMx * 0.001 ? perfInfo.numReductions / time : 0, 0, 0.01, 1.05));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Count the number of times the parent event was called */

static int countParents(const PetscNestedEventTree *tree, PetscEventPerfInfo *eventPerfInfo, int i)
{
  if (tree[i].depth <= 1) {
    return 1; /* Main event: only once */
  } else if (!tree[i].own) {
    return 1; /* This event didn't happen in this process, but did in another */
  } else {
    int iParent;
    for (iParent = i - 1; iParent >= 0; iParent--) {
      if (tree[iParent].depth == tree[i].depth - 1) break;
    }
    if (tree[iParent].depth != tree[i].depth - 1) {
      /* *****  Internal error: cannot find parent */
      return -2;
    } else {
      PetscLogEvent dftEvent = tree[iParent].dftEvent;
      return eventPerfInfo[dftEvent].count;
    }
  }
}

typedef struct {
  int            id;
  PetscLogDouble val;
} PetscSortItem;

static int compareSortItems(const void *item1_, const void *item2_)
{
  PetscSortItem *item1 = (PetscSortItem *)item1_;
  PetscSortItem *item2 = (PetscSortItem *)item2_;
  if (item1->val > item2->val) return -1;
  if (item1->val < item2->val) return +1;
  return 0;
}

/*
 * Find the number of child events.
 */
static PetscErrorCode PetscLogNestedTreeGetChildrenCount(const PetscNestedEventTree *tree, int nTimers, int iStart, int depth, int *nChildren)
{
  int n = 0;

  PetscFunctionBegin;
  for (int i = iStart + 1; i < nTimers; i++) {
    if (tree[i].depth <= depth) break;
    if (tree[i].depth == depth + 1) n++;
  }
  *nChildren = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Initialize child event sort items with ID and times.
 */
static PetscErrorCode PetscLogNestedTreeSetChildrenSortItems(const PetscViewer viewer, const PetscNestedEventTree *tree, int nTimers, int iStart, int depth, int nChildren, PetscSortItem **children)
{
  MPI_Comm        comm;
  PetscLogDouble *times, *maxTimes;
  PetscStageLog   stageLog;
  PetscStageInfo *stageInfo;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscLogGetStageLog(&stageLog));

  if (nChildren > 0) {
    /* Create an array for the id-s and maxTimes of the children,
     *  leaving 2 spaces for self-time and other-time */

    PetscCall(PetscMalloc1(nChildren + 2, children));
    nChildren = 0;
    for (int i = iStart + 1; i < nTimers; i++) {
      if (tree[i].depth <= depth) break;
      if (tree[i].depth == depth + 1) {
        PetscInt event_id;

        (*children)[nChildren].id = i;
        PetscCall(NestedEventGetLogStageInfo(&tree[i], &stageInfo));
        event_id = tree[i].dftEvent;
        if (event_id >= 0) {
          (*children)[nChildren].val = stageInfo->eventLog->eventInfo[tree[i].dftEvent].time;
        } else {
          // event node
          (*children)[nChildren].val = stageInfo->perfInfo.time;
        }
        nChildren++;
      }
    }

    /* Calculate the children's maximum times, to see whether children will be ignored or printed */
    PetscCall(PetscMalloc1(nChildren, &times));
    for (int i = 0; i < nChildren; i++) times[i] = (*children)[i].val;

    PetscCall(PetscMalloc1(nChildren, &maxTimes));
    PetscCall(MPIU_Allreduce(times, maxTimes, nChildren, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
    PetscCall(PetscFree(times));

    for (int i = 0; i < nChildren; i++) (*children)[i].val = maxTimes[i];
    PetscCall(PetscFree(maxTimes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Set 'self' and 'other' performance info.
 */
static PetscErrorCode PetscLogNestedTreeSetSelfOtherPerfInfo(const PetscNestedEventTree *tree, int iStart, PetscLogDouble totalTime, const PetscSortItem *children, int nChildren, PetscEventPerfInfo *myPerfInfo, PetscEventPerfInfo *selfPerfInfo, PetscEventPerfInfo *otherPerfInfo, int *parentCount, PetscLogDouble *countsPerCall)
{
  PetscStageLog       stageLog;
  PetscStageInfo     *stageInfo;
  PetscEventPerfInfo *eventPerfInfo;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  //eventPerfInfo = stageLog->stageInfo[stage].eventLog->eventInfo;
  if (!tree[iStart].own) {
    /* Set values for a timer that was not activated in this process
   * (but was, in other processes of this run) */
    PetscCall(PetscMemzero(myPerfInfo, sizeof(*myPerfInfo)));

    *selfPerfInfo  = *myPerfInfo;
    *otherPerfInfo = *myPerfInfo;

    *parentCount   = 1;
    *countsPerCall = 0;
  } else {
    /* Set the values for a timer that was activated in this process */
    PetscLogEvent dftEvent = tree[iStart].dftEvent;

    PetscCall(NestedEventGetLogStageInfo(&tree[iStart], &stageInfo));
    eventPerfInfo = stageInfo->eventLog->eventInfo;

    *parentCount = countParents(tree, eventPerfInfo, iStart);
    if (dftEvent >= 0) {
      *myPerfInfo = eventPerfInfo[dftEvent];
    } else {
      *myPerfInfo = stageInfo->perfInfo;
    }
    *countsPerCall = (PetscLogDouble)myPerfInfo->count / (PetscLogDouble)*parentCount;

    *selfPerfInfo                = *myPerfInfo;
    otherPerfInfo->time          = 0;
    otherPerfInfo->flops         = 0;
    otherPerfInfo->numMessages   = 0;
    otherPerfInfo->messageLength = 0;
    otherPerfInfo->numReductions = 0;

    for (int i = 0; i < nChildren; i++) {
      /* For all child counters: subtract the child values from self-timers */
      PetscEventPerfInfo childPerfInfo;

      PetscLogEvent dftChild = tree[children[i].id].dftEvent;
      if (dftChild >= 0) {
        childPerfInfo = eventPerfInfo[dftChild];
      } else {
        childPerfInfo = stageLog->stageInfo[-(dftChild + 2)].perfInfo;
      }

      selfPerfInfo->time -= childPerfInfo.time;
      selfPerfInfo->flops -= childPerfInfo.flops;
      selfPerfInfo->numMessages -= childPerfInfo.numMessages;
      selfPerfInfo->messageLength -= childPerfInfo.messageLength;
      selfPerfInfo->numReductions -= childPerfInfo.numReductions;

      if ((children[i].val / totalTime) < THRESHOLD) {
        /* Add them to 'other' if the time is ignored in the output */
        otherPerfInfo->time += childPerfInfo.time;
        otherPerfInfo->flops += childPerfInfo.flops;
        otherPerfInfo->numMessages += childPerfInfo.numMessages;
        otherPerfInfo->messageLength += childPerfInfo.messageLength;
        otherPerfInfo->numReductions += childPerfInfo.numReductions;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Set max times across ranks for 'self' and 'other'.
 */
static PetscErrorCode PetscLogNestedTreeSetMaxTimes(MPI_Comm comm, int nChildren, const PetscEventPerfInfo selfPerfInfo, const PetscEventPerfInfo otherPerfInfo, PetscSortItem *children)
{
  PetscLogDouble times[2], maxTimes[2];

  PetscFunctionBegin;
  times[0] = selfPerfInfo.time;
  times[1] = otherPerfInfo.time;

  PetscCall(MPIU_Allreduce(times, maxTimes, 2, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  children[nChildren + 0].id  = -1;
  children[nChildren + 0].val = maxTimes[0];
  children[nChildren + 1].id  = -2;
  children[nChildren + 1].val = maxTimes[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrint(PetscViewer viewer, PetscNestedEventTree *tree, int nTimers, int iStart, PetscLogDouble totalTime)
{
  int                depth = tree[iStart].depth;
  const char        *name;
  int                parentCount = 1, nChildren;
  PetscSortItem     *children;
  PetscStageLog      stageLog;
  PetscEventRegInfo *eventRegInfo;
  PetscEventPerfInfo myPerfInfo = {0}, selfPerfInfo = {0}, otherPerfInfo = {0};
  PetscLogDouble     countsPerCall = 0;
  PetscBool          wasPrinted;
  PetscBool          childWasPrinted;
  MPI_Comm           comm;

  PetscFunctionBegin;
  /* Look up the name of the event and its PerfInfo */
  PetscCall(PetscLogGetStageLog(&stageLog));
  eventRegInfo = stageLog->eventLog->eventInfo;
  name         = eventRegInfo[(PetscLogEvent)tree[iStart].nstEvent].name;
  PetscCheck(name != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event name is NULL");
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  PetscCall(PetscLogNestedTreeGetChildrenCount(tree, nTimers, iStart, depth, &nChildren));
  PetscCall(PetscLogNestedTreeSetChildrenSortItems(viewer, tree, nTimers, iStart, depth, nChildren, &children));
  PetscCall(PetscLogNestedTreeSetSelfOtherPerfInfo(tree, iStart, totalTime, children, nChildren, &myPerfInfo, &selfPerfInfo, &otherPerfInfo, &parentCount, &countsPerCall));

  /* Main output for this timer */
  PetscCall(PetscLogNestedTreePrintLine(viewer, myPerfInfo, countsPerCall, parentCount, depth, name, totalTime, &wasPrinted));

  /* Now print the lines for the children */
  if (nChildren > 0) {
    int i;

    /* Calculate max-times for 'self' and 'other' */
    PetscCall(PetscLogNestedTreeSetMaxTimes(comm, nChildren, selfPerfInfo, otherPerfInfo, children));

    /* Now sort the children (including 'self' and 'other') on total time */
    qsort(children, nChildren + 2, sizeof(PetscSortItem), compareSortItems);

    /* Print (or ignore) the children in ascending order of total time */
    PetscCall(PetscViewerXMLStartSection(viewer, "events", NULL));
    for (i = 0; i < nChildren + 2; i++) {
      if ((children[i].val / totalTime) < THRESHOLD) {
        /* ignored: no output */
      } else if (children[i].id == -1) {
        PetscCall(PetscLogNestedTreePrintLine(viewer, selfPerfInfo, 1, parentCount, depth + 1, "self", totalTime, &childWasPrinted));
        if (childWasPrinted) PetscCall(PetscViewerXMLEndSection(viewer, "event"));
      } else if (children[i].id == -2) {
        size_t len;
        char  *otherName;

        PetscCall(PetscStrlen(name, &len));
        PetscCall(PetscMalloc1(len + 16, &otherName));
        PetscCall(PetscSNPrintf(otherName, len + 16, "%s: other-timed", name ? name : "(none)"));
        PetscCall(PetscLogNestedTreePrintLine(viewer, otherPerfInfo, 1, 1, depth + 1, otherName, totalTime, &childWasPrinted));
        PetscCall(PetscFree(otherName));
        if (childWasPrinted) PetscCall(PetscViewerXMLEndSection(viewer, "event"));
      } else {
        /* Print the child with a recursive call to this function */
        PetscCall(PetscLogNestedTreePrint(viewer, tree, nTimers, children[i].id, totalTime));
      }
    }
    PetscCall(PetscViewerXMLEndSection(viewer, "events"));
    PetscCall(PetscFree(children));
  }

  if (wasPrinted) PetscCall(PetscViewerXMLEndSection(viewer, "event"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintLineNew(PetscViewer viewer, const PetscEventPerfInfo *perfInfo, PetscLogDouble countsPerCall, int parentCount, const char *name, PetscLogDouble totalTime)
{
  PetscLogDouble time = perfInfo->time;
  PetscLogDouble timeMx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(MPIU_Allreduce(&time, &timeMx, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(PetscViewerXMLPutString(viewer, "name", NULL, name));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "time", time / totalTime * 100.0, 0, 0, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", parentCount > 0 ? countsPerCall : 1.0, 0.99, 1.01, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mflops", time >= timeMx * 0.001 ? 1e-6 * perfInfo->flops / time : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mbps", time >= timeMx * 0.001 ? perfInfo->messageLength / (1024 * 1024 * time) : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", time >= timeMx * 0.001 ? perfInfo->numReductions / time : 0, 0, 0.01, 1.05));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNestedNameGetBase(const char name[], const char *base[])
{
  size_t n;
  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &n));
  while (n > 0 && name[n-1] != ';') n--;
  *base = &name[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

// prints and leaves perf holding self times
static PetscErrorCode PetscLogNestedTreePrintNew(PetscViewer viewer, double total_time, PetscEventPerfInfo *parent_info, PetscInt parent, PetscInt num_nodes, const PetscNestedEventNode tree[], PetscEventPerfInfo perf[])
{
  PetscInt num_children = 0;
  PetscInt *perm;
  PetscReal *times;
  PetscEventPerfInfo other;

  PetscFunctionBegin;
  for (PetscInt node = 0; node < num_nodes; node += 1 + tree[node].num_descendants) {
    PetscAssert(tree[node].parent == parent, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed tree invariant");
    num_children++;
  }
  if (num_children == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMemzero(&other, sizeof(other)));
  PetscCall(PetscMalloc2(num_children + 2, &times, num_children + 2, &perm));
  for (PetscInt i = 0, node = 0; node < num_nodes; i++, node += 1 + tree[node].num_descendants) {
    PetscLogDouble child_time = perf[node].time;

    perm[i] = node;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &child_time, 1, MPI_DOUBLE, MPI_MAX, PetscObjectComm((PetscObject)viewer)));
    times[i] = -child_time;

    parent_info->time          -= perf[node].time;
    parent_info->flops         -= perf[node].flops;
    parent_info->numMessages   -= perf[node].numMessages;
    parent_info->messageLength -= perf[node].messageLength;
    parent_info->numReductions -= perf[node].numReductions;
    if (child_time / total_time < THRESHOLD) {
      other.time          += perf[node].time;
      other.flops         += perf[node].flops;
      other.numMessages   += perf[node].numMessages;
      other.messageLength += perf[node].messageLength;
      other.numReductions += perf[node].numReductions;
      other.count         += perf[node].count;
    }
  }
  perm[num_children] = -1;
  times[num_children] = -parent_info->time;
  perm[num_children + 1] = -2;
  times[num_children + 1] = -other.time;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &times[num_children], 2, MPI_DOUBLE, MPI_MIN, PetscObjectComm((PetscObject)viewer)));
  // sort descending by time
  PetscCall(PetscSortRealWithArrayInt(num_children + 2, times, perm));

  PetscCall(PetscViewerXMLStartSection(viewer, "events", NULL));
  for (PetscInt i = 0; i < num_children + 2; i++) {
    PetscInt node = perm[i];
    PetscLogDouble child_time = -times[i];

    if (child_time / total_time >= THRESHOLD || (node < 0 && child_time > 0.0)) {
      PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
      if (node == -1) {
        PetscCall(PetscLogNestedTreePrintLineNew(viewer, parent_info, 0, 0, "self", total_time));
      } else if (node == -2) {
        PetscCall(PetscLogNestedTreePrintLineNew(viewer, &other, ((double) other.count) / ((double) parent_info->count), parent_info->count, "other", total_time));
      } else {
        const char *base_name;
        PetscCall(PetscNestedNameGetBase(tree[node].name, &base_name));
        PetscCall(PetscLogNestedTreePrintLineNew(viewer, &perf[node], ((double) perf[node].count) / ((double) parent_info->count), parent_info->count, base_name, total_time));
        PetscCall(PetscLogNestedTreePrintNew(viewer, total_time, &perf[node], tree[node].id, tree[node].num_descendants, &tree[node+1], &perf[node+1]));
      }
      PetscCall(PetscViewerXMLEndSection(viewer, "event"));
    }
  }
  PetscCall(PetscViewerXMLEndSection(viewer, "events"));

  PetscCall(PetscFree2(times, perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintTopNew(PetscViewer viewer, PetscNestedEventTreeNew *tree, PetscLogDouble *total_time)
{
  PetscNestedEventNode *main_stage;
  PetscNestedEventNode *tree_rem;
  PetscEventPerfInfo   *main_stage_perf;
  PetscEventPerfInfo   *perf_rem;
  PetscLogDouble        time;
  PetscInt              num_nodes;

  PetscFunctionBegin;
  main_stage = &tree->tree[0];
  tree_rem = &tree->tree[1];
  main_stage_perf = &tree->perf[0];
  perf_rem = &tree->perf[1];
  num_nodes = main_stage->num_descendants;
  time = main_stage_perf->time;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, tree->comm));
  *total_time = time;
  /* Print (or ignore) the children in ascending order of total time */
  PetscCall(PetscViewerXMLStartSection(viewer, "timertree", "Timings tree"));
  PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, time, "%f"));
  PetscCall(PetscViewerXMLPutDouble(viewer, "timethreshold", NULL, thresholdTime, "%f"));
  PetscCall(PetscLogNestedTreePrintNew(viewer, main_stage_perf->time, main_stage_perf, main_stage->id, num_nodes, tree_rem, perf_rem));
  PetscCall(PetscViewerXMLEndSection(viewer, "timertree"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintTop(PetscViewer viewer, PetscNestedEventTree *tree, int nTimers, PetscLogDouble totalTime)
{
  int            i, nChildren;
  PetscSortItem *children;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  PetscCall(PetscLogNestedTreeGetChildrenCount(tree, nTimers, -1, 0, &nChildren));
  PetscCall(PetscLogNestedTreeSetChildrenSortItems(viewer, tree, nTimers, -1, 0, nChildren, &children));

  if (nChildren > 0) {
    /* Now sort the children on total time */
    qsort(children, nChildren, sizeof(PetscSortItem), compareSortItems);
    /* Print (or ignore) the children in ascending order of total time */
    PetscCall(PetscViewerXMLStartSection(viewer, "timertree", "Timings tree"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "timethreshold", NULL, thresholdTime, "%f"));

    for (i = 0; i < nChildren; i++) {
      if ((children[i].val / totalTime) < THRESHOLD) {
        /* ignored: no output */
      } else {
        /* Print the child with a recursive call to this function */
        PetscCall(PetscLogNestedTreePrint(viewer, tree, nTimers, children[i].id, totalTime));
      }
    }
    PetscCall(PetscViewerXMLEndSection(viewer, "timertree"));
    PetscCall(PetscFree(children));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  const char    *name;
  PetscLogDouble time;
  PetscLogDouble flops;
  PetscLogDouble numMessages;
  PetscLogDouble messageLength;
  PetscLogDouble numReductions;
} PetscSelfTimer;

static PetscErrorCode PetscCalcSelfTimeNew(PetscViewer viewer, PetscNestedEventTreeNew *tree, PetscSelfTimer **p_self)
{
  PetscInt global_count = tree->global_stages->count + tree->global_events->count;
  PetscInt event_offset = tree->global_stages->count;
  PetscSelfTimer     *perf_by_id;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(global_count, &perf_by_id));

  for (PetscInt i = 0; i < global_count; i++) {
    PetscInt loc = tree->tree[i].id;

    perf_by_id[loc].name = tree->tree[i].name;
    perf_by_id[loc].time = tree->perf[i].time;
    perf_by_id[loc].flops = tree->perf[i].flops;
    perf_by_id[loc].numMessages = tree->perf[i].numMessages;
    perf_by_id[loc].numReductions = tree->perf[i].numReductions;
  }

  for (PetscInt stage = 0; stage < tree->global_stages->count; stage++) {
    PetscInt stage_id = tree->global_stages->global_to_local[stage];
    if (stage_id >= 0) {
      PetscInt root_stage;
      PetscCall(NestedStageToRootStage(stage_id, &root_stage));
      if (root_stage != stage_id) {
        PetscInt global_root = tree->global_stages->local_to_global[root_stage];
        perf_by_id[global_root].time += perf_by_id[stage].time;
        perf_by_id[global_root].flops += perf_by_id[stage].flops;
        perf_by_id[global_root].numMessages += perf_by_id[stage].numMessages;
        perf_by_id[global_root].messageLength += perf_by_id[stage].messageLength;
        perf_by_id[global_root].numReductions += perf_by_id[stage].numReductions;
        PetscCall(PetscMemzero(&perf_by_id[stage], sizeof(*perf_by_id)));
      }
    }
  }

  for (PetscInt event = 0; event < tree->global_events->count; event++) {
    PetscInt event_id = tree->global_events->global_to_local[event];
    if (event_id >= 0) {
      PetscInt root_stage;
      PetscCall(NestedEventToRootEvent(event_id, &root_stage));
      if (root_stage != event_id) {
        PetscInt global_root = tree->global_events->local_to_global[root_stage] + event_offset;
        PetscInt global_leaf = event + event_offset;

        perf_by_id[global_root].time += perf_by_id[global_leaf].time;
        perf_by_id[global_root].flops += perf_by_id[global_leaf].flops;
        perf_by_id[global_root].numMessages += perf_by_id[global_leaf].numMessages;
        perf_by_id[global_root].messageLength += perf_by_id[global_leaf].messageLength;
        perf_by_id[global_root].numReductions += perf_by_id[global_leaf].numReductions;
        PetscCall(PetscMemzero(&perf_by_id[global_leaf], sizeof(*perf_by_id)));
      }
    }
  }

  *p_self = perf_by_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscCalcSelfTime(PetscViewer viewer, PetscSelfTimer **p_self, int *p_nstMax)
{
  PetscStageLog       stageLog;
  PetscEventRegInfo  *eventRegInfo;
  PetscEventPerfInfo *eventPerfInfo;
  PetscSelfTimer     *selftimes;
  PetscSelfTimer     *totaltimes;
  NestedEventId      *nstEvents;
  int                 i, j, maxDefaultTimer;
  NestedEventId       nst;
  PetscLogEvent       dft;
  int                 nstMax, nstMax_local;
  MPI_Comm            comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscLogGetStageLog(&stageLog));
  eventRegInfo = stageLog->eventLog->eventInfo;

  /* For each default timer, calculate the (one) nested timer that it corresponds to. */
  maxDefaultTimer = 0;
  for (i = 0; i < nNestedEvents; i++) {
    int            nParents  = nestedEvents[i].nParents;
    PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    for (j = 0; j < nParents; j++) maxDefaultTimer = PetscMax(dftEvents[j], maxDefaultTimer);
  }
  PetscCall(PetscMalloc1(maxDefaultTimer + 1, &nstEvents));
  for (dft = 0; dft < maxDefaultTimer; dft++) nstEvents[dft] = 0;
  for (i = 0; i < nNestedEvents; i++) {
    int            nParents  = nestedEvents[i].nParents;
    NestedEventId  nstEvent  = nestedEvents[i].nstEvent;
    PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    for (j = 0; j < nParents; j++) nstEvents[dftEvents[j]] = nstEvent;
  }

  /* Calculate largest nested event-ID */
  nstMax_local = 0;
  for (i = 0; i < nNestedEvents; i++) nstMax_local = PetscMax(nestedEvents[i].nstEvent, nstMax_local);
  PetscCall(MPIU_Allreduce(&nstMax_local, &nstMax, 1, MPI_INT, MPI_MAX, comm));

  /* Initialize all total-times with zero */
  PetscCall(PetscMalloc1(nstMax + 1, &selftimes));
  PetscCall(PetscMalloc1(nstMax + 1, &totaltimes));
  for (nst = 0; nst <= nstMax; nst++) {
    totaltimes[nst].time          = 0;
    totaltimes[nst].flops         = 0;
    totaltimes[nst].numMessages   = 0;
    totaltimes[nst].messageLength = 0;
    totaltimes[nst].numReductions = 0;
    totaltimes[nst].name          = NULL;
  }

  /* Calculate total-times */
  for (i = 0; i < nNestedEvents; i++) {
    const int            nParents  = nestedEvents[i].nParents;
    const NestedEventId  nstEvent  = nestedEvents[i].nstEvent;
    const PetscLogEvent *dftEvents = nestedEvents[i].dftEvents;
    for (j = 0; j < nParents; j++) {
      const PetscLogEvent dftEvent = dftEvents[j];
      totaltimes[nstEvent].time += eventPerfInfo[dftEvent].time;
      totaltimes[nstEvent].flops += eventPerfInfo[dftEvent].flops;
      totaltimes[nstEvent].numMessages += eventPerfInfo[dftEvent].numMessages;
      totaltimes[nstEvent].messageLength += eventPerfInfo[dftEvent].messageLength;
      totaltimes[nstEvent].numReductions += eventPerfInfo[dftEvent].numReductions;
    }
    totaltimes[nstEvent].name = eventRegInfo[(PetscLogEvent)nstEvent].name;
  }

  /* Initialize: self-times := totaltimes */
  for (nst = 0; nst <= nstMax; nst++) selftimes[nst] = totaltimes[nst];

  /* Subtract timed subprocesses from self-times */
  for (i = 0; i < nNestedEvents; i++) {
    const int            nParents         = nestedEvents[i].nParents;
    const PetscLogEvent *dftEvents        = nestedEvents[i].dftEvents;
    const NestedEventId *dftParentsSorted = nestedEvents[i].dftParentsSorted;
    for (j = 0; j < nParents; j++) {
      if (dftParentsSorted[j] != DFT_ID_AWAKE) {
        const PetscLogEvent dftEvent  = dftEvents[j];
        const NestedEventId nstParent = nstEvents[dftParentsSorted[j]];
        selftimes[nstParent].time -= eventPerfInfo[dftEvent].time;
        selftimes[nstParent].flops -= eventPerfInfo[dftEvent].flops;
        selftimes[nstParent].numMessages -= eventPerfInfo[dftEvent].numMessages;
        selftimes[nstParent].messageLength -= eventPerfInfo[dftEvent].messageLength;
        selftimes[nstParent].numReductions -= eventPerfInfo[dftEvent].numReductions;
      }
    }
  }

  PetscCall(PetscFree(nstEvents));
  PetscCall(PetscFree(totaltimes));

  /* Set outputs */
  *p_self   = selftimes;
  *p_nstMax = nstMax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintSelfTime(PetscViewer viewer, const PetscSelfTimer *selftimes, int num_times, PetscLogDouble totalTime)
{
  int                i;
  NestedEventId      nst;
  PetscSortItem     *sortSelfTimes;
  PetscLogDouble    *times, *maxTimes;
  const int          dum_count = 1, dum_parentcount = 1;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  PetscCall(PetscMalloc1(num_times, &times));
  PetscCall(PetscMalloc1(num_times, &maxTimes));
  for (nst = 0; nst < num_times; nst++) times[nst] = selftimes[nst].time;
  PetscCall(MPIU_Allreduce(times, maxTimes, num_times, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(PetscFree(times));

  PetscCall(PetscMalloc1(num_times + 1, &sortSelfTimes));

  /* Sort the self-timers on basis of the largest time needed */
  for (nst = 0; nst < num_times; nst++) {
    sortSelfTimes[nst].id  = nst;
    sortSelfTimes[nst].val = maxTimes[nst];
  }
  PetscCall(PetscFree(maxTimes));
  qsort(sortSelfTimes, num_times, sizeof(PetscSortItem), compareSortItems);

  PetscCall(PetscViewerXMLStartSection(viewer, "selftimertable", "Self-timings"));
  PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f"));

  for (i = 0; i < num_times; i++) {
    if ((sortSelfTimes[i].val / totalTime) >= THRESHOLD) {
      NestedEventId      nstEvent = sortSelfTimes[i].id;
      const char        *name;
      PetscEventPerfInfo selfPerfInfo;

      selfPerfInfo.time          = selftimes[nstEvent].time;
      selfPerfInfo.flops         = selftimes[nstEvent].flops;
      selfPerfInfo.numMessages   = selftimes[nstEvent].numMessages;
      selfPerfInfo.messageLength = selftimes[nstEvent].messageLength;
      selfPerfInfo.numReductions = selftimes[nstEvent].numReductions;

      PetscCall(PetscNestedNameGetBase(selftimes[nstEvent].name, &name));
      PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
      PetscCall(PetscLogNestedTreePrintLineNew(viewer, &selfPerfInfo, dum_count, dum_parentcount, name, totalTime));
      PetscCall(PetscViewerXMLEndSection(viewer, "event"));
    }
  }
  PetscCall(PetscViewerXMLEndSection(viewer, "selftimertable"));
  PetscCall(PetscFree(sortSelfTimes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogComputeEventNestedName(PetscStageLog, NestedEventId, char **, char **);

static PetscErrorCode PetscStageLogComputeStageNestedName(PetscStageLog stage_log, NestedEventId stage_id, char **nested_event_names, char **nested_stage_names)
{
  PetscNestedEvent *nested_event;
  PetscLogEvent    *dftParents;
  PetscLogEvent    parent_id;
  PetscLogEvent *dftEventsSorted;
  int            entry;
  int            nParents, tentry;
  char           buf[BUFSIZ];
  NestedEventId root_id;

  PetscFunctionBegin;
  if (nested_stage_names[stage_id]) PetscFunctionReturn(PETSC_SUCCESS);
  if (-(stage_id+2) == MAINSTAGE_EVENT) {
    PetscCall(PetscStrallocpy(stage_log->stageInfo[stage_id].name, &nested_stage_names[stage_id]));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(NestedStageToRootStage(stage_id, &root_id));
  PetscCall(PetscLogEventFindNestedTimer(-(root_id+2), &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != -(root_id + 2)) PetscFunctionReturn(PETSC_SUCCESS);
  nested_event = &nestedEvents[entry];
  dftParents      = nested_event->dftParents;
  dftEventsSorted = nested_event->dftEventsSorted;
  nParents         = nested_event->nParents;
  PetscCall(PetscLogEventFindDefaultTimer(-(stage_id+2), dftEventsSorted, nParents, &tentry));
  PetscAssert(dftEventsSorted[tentry] == -(stage_id+2), PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
  parent_id = dftParents[tentry];
  if (parent_id >= 0) {
    PetscCall(PetscStageLogComputeEventNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
    PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_event_names[parent_id], stage_log->stageInfo[root_id].name));
  } else {
    if (parent_id <= MAINSTAGE_EVENT) {
      parent_id = -(parent_id+2);
      PetscCall(PetscStageLogComputeStageNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_stage_names[parent_id], stage_log->stageInfo[root_id].name));
    } else {
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s", stage_log->stageInfo[root_id].name));
    }
  }
  PetscCall(PetscStrallocpy(buf, &nested_stage_names[stage_id]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogComputeEventNestedName(PetscStageLog stage_log, NestedEventId event_id, char **nested_event_names, char **nested_stage_names)
{
  PetscNestedEvent *nested_event;
  PetscLogEvent    *dftParents;
  PetscLogEvent    parent_id;
  PetscLogEvent *dftEventsSorted;
  int            entry;
  int            nParents, tentry;
  char           buf[BUFSIZ];
  NestedEventId root_id;

  PetscFunctionBegin;
  if (nested_event_names[event_id]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(NestedEventToRootEvent(event_id, &root_id));
  PetscCall(PetscLogEventFindNestedTimer(root_id, &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != root_id) PetscFunctionReturn(PETSC_SUCCESS);
  nested_event = &nestedEvents[entry];
  dftParents      = nested_event->dftParents;
  dftEventsSorted = nested_event->dftEventsSorted;
  nParents         = nested_event->nParents;
  PetscCall(PetscLogEventFindDefaultTimer(event_id, dftEventsSorted, nParents, &tentry));
  PetscAssert(dftEventsSorted[tentry] == event_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
  parent_id = dftParents[tentry];
  if (parent_id >= 0) {
    PetscCall(PetscStageLogComputeEventNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
    PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_event_names[parent_id], stage_log->eventLog->eventInfo[root_id].name));
  } else {
    if (parent_id <= MAINSTAGE_EVENT) {
      parent_id = -(parent_id+2);
      PetscCall(PetscStageLogComputeStageNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_stage_names[parent_id], stage_log->eventLog->eventInfo[root_id].name));
    } else {
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s", stage_log->eventLog->eventInfo[root_id].name));
    }
  }
  PetscCall(PetscStrallocpy(buf, &nested_event_names[event_id]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogNestNames(PetscStageLog stage_log)
{
  PetscInt num_stages = stage_log->numStages;
  PetscInt num_events = stage_log->eventLog->numEvents;
  char **nested_stage_names;
  char **nested_event_names;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(num_stages, &nested_stage_names));
  PetscCall(PetscCalloc1(num_events, &nested_event_names));

  for (NestedEventId i = 0; i < num_events; i++) PetscCall(PetscStageLogComputeEventNestedName(stage_log, i, nested_event_names, nested_stage_names));

  for (NestedEventId i = 0; i < num_stages; i++) PetscCall(PetscStageLogComputeStageNestedName(stage_log, i, nested_event_names, nested_stage_names));

  for (NestedEventId i = 0; i < num_events; i++) {
    if (nested_event_names[i]) {
      PetscCall(PetscFree(stage_log->eventLog->eventInfo[i].name));
      stage_log->eventLog->eventInfo[i].name = nested_event_names[i];
    }
  }

  for (NestedEventId i = 0; i < num_stages; i++) {
    if (nested_stage_names[i]) {
      PetscCall(PetscFree(stage_log->stageInfo[i].name));
      stage_log->stageInfo[i].name = nested_stage_names[i];
    }
  }

  PetscCall(PetscFree(nested_stage_names));
  PetscCall(PetscFree(nested_event_names));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogView_Nested(PetscViewer viewer)
{
  PetscLogDouble        locTotalTime, globTotalTime;
  PetscStageLog         stage_log_orig;
  MPI_Comm              comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscViewerInitASCII_XML(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "<!-- PETSc Performance Summary: -->\n"));
  PetscCall(PetscViewerXMLStartSection(viewer, "petscroot", NULL));

  /* Get the total elapsed time, local and global maximum */

  /* Print global information about this run */
  PetscCall(PetscPrintExeSpecs(viewer));
  PetscCall(PetscLogGetStageLog(&stage_log_orig));
  locTotalTime = stage_log_orig->stageInfo[0].perfInfo.time; // Main stage time
  PetscCall(PetscPrintGlobalPerformance(viewer, locTotalTime));
  /* Collect nested timer tree info from all processes */
  {
    PetscLogGlobalNames global_stages, global_events;
    PetscStageLog stage_log_nested;
    PetscNestedEventNode *tree;
    PetscNestedEventTreeNew tree_traversal;
    PetscEventPerfInfo *perf;
    PetscSelfTimer     *self_timers;

    PetscCall(PetscStageLogDuplicate(stage_log_orig, &stage_log_nested));
    PetscCall(PetscStageLogNestNames(stage_log_nested));
    PetscCall(PetscStageLogCreateGlobalStageNames(comm, stage_log_nested, &global_stages));
    PetscCall(PetscStageLogCreateGlobalEventNames(comm, stage_log_nested, &global_events));
    PetscCall(PetscLogNestedCreatePerfTree(comm, stage_log_nested, global_stages, global_events, &tree, &perf));
    tree_traversal.comm = comm;
    tree_traversal.global_events = global_events;
    tree_traversal.global_stages = global_stages;
    tree_traversal.perf = perf;
    tree_traversal.tree = tree;
    tree_traversal.stage_log = stage_log_nested;
    PetscCall(PetscLogNestedTreePrintTopNew(viewer, &tree_traversal, &globTotalTime));
    PetscCall(PetscCalcSelfTimeNew(viewer, &tree_traversal, &self_timers));
    PetscCall(PetscPrintSelfTime(viewer, self_timers, global_stages->count + global_events->count, globTotalTime));
    PetscCall(PetscFree(self_timers));

    PetscCall(PetscFree(perf));
    PetscCall(PetscFree(tree));
    PetscCall(PetscLogGlobalNamesDestroy(&global_events));
    PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
    PetscCall(PetscStageLogDestroy(stage_log_nested));
  }
  //PetscCall(PetscLogNestedTreeCreate(viewer, &tree, &nTimers));
  //PetscCall(PetscLogNestedTreePrintTop(viewer, tree, nTimers, globTotalTime));
  //PetscCall(PetscLogNestedTreeDestroy(tree, nTimers));

  /* Calculate self-time for all (not-nested) events */
  //PetscCall(PetscCalcSelfTime(viewer, &selftimers, &nstMax));
  //PetscCall(PetscPrintSelfTime(viewer, selftimers, nstMax, globTotalTime));
  //PetscCall(PetscFree(selftimers));

  PetscCall(PetscViewerXMLEndSection(viewer, "petscroot"));
  PetscCall(PetscViewerFinalASCII_XML(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Get the name of a nested event.
 */
static PetscErrorCode PetscGetNestedEventName(const PetscNestedEventTree *tree, int id, char **name)
{
  int           event;
  PetscStageLog stageLog;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  event = tree[id].nstEvent;
  if (event >= 0) {
    *name = stageLog->eventLog->eventInfo[(PetscLogEvent)event].name;
  } else {
    *name = stageLog->stageInfo[-(event + 2)].name;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Get the total time elapsed.
 */
static PetscErrorCode PetscGetTotalTime(const PetscViewer viewer, PetscLogDouble *totalTime)
{
  PetscLogDouble locTotalTime;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCall(MPIU_Allreduce(&locTotalTime, totalTime, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Write a line to the flame graph output and then recurse into child events.
 */
static PetscErrorCode PetscLogNestedTreePrintFlamegraph(PetscViewer viewer, PetscNestedEventTree *tree, int nTimers, int iStart, PetscLogDouble totalTime, PetscIntStack eventStack)
{
  int                depth = tree[iStart].depth, parentCount = 1, i, nChildren;
  char              *name       = NULL;
  PetscEventPerfInfo myPerfInfo = {0}, selfPerfInfo = {0}, otherPerfInfo = {0};
  PetscLogDouble     countsPerCall = 0, locTime, globTime;
  PetscSortItem     *children;
  PetscStageLog      stageLog;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  /* Determine information about the child events as well as 'self' and 'other' */
  PetscCall(PetscLogNestedTreeGetChildrenCount(tree, nTimers, iStart, depth, &nChildren));
  PetscCall(PetscLogNestedTreeSetChildrenSortItems(viewer, tree, nTimers, iStart, depth, nChildren, &children));
  PetscCall(PetscLogNestedTreeSetSelfOtherPerfInfo(tree, iStart, totalTime, children, nChildren, &myPerfInfo, &selfPerfInfo, &otherPerfInfo, &parentCount, &countsPerCall));

  /* Write line to the file. The time shown is 'self' + 'other' because each entry in the output
   * is the total time spent in the event minus the amount spent in child events. */
  locTime = selfPerfInfo.time + otherPerfInfo.time;
  PetscCall(MPIU_Allreduce(&locTime, &globTime, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  if (globTime / totalTime > THRESHOLD && tree[iStart].own) {
    /* Iterate over parent events in the stack and write them */
    for (i = 0; i <= eventStack->top; i++) {
      PetscCall(PetscGetNestedEventName(tree, eventStack->stack[i], &name));
      PetscCall(PetscViewerASCIIPrintf(viewer, "%s;", name));
    }
    PetscCall(PetscGetNestedEventName(tree, iStart, &name));
    /* The output is given as an integer in microseconds because otherwise the file cannot be read
     * by apps such as speedscope (https://speedscope.app/). */
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s %" PetscInt64_FMT "\n", name, (PetscInt64)(globTime * 1e6)));
  }

  /* Add the current event to the parent stack and write the child events */
  PetscCall(PetscIntStackPush(eventStack, iStart));
  for (i = 0; i < nChildren; i++) PetscCall(PetscLogNestedTreePrintFlamegraph(viewer, tree, nTimers, children[i].id, totalTime, eventStack));
  if (nChildren > 0) PetscCall(PetscFree(children));

  /* Pop the top item from the stack and immediately discard it */
  {
    int tmp;
    PetscCall(PetscIntStackPop(eventStack, &tmp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Print nested logging information to a file suitable for reading into a Flame Graph.
 *
 * The format consists of a semicolon-separated list of events and the event duration in microseconds (which must be an integer).
 * An example output would look like:
 *   MatAssemblyBegin 1
 *   MatAssemblyEnd 10
 *   MatView 302
 *   KSPSetUp 98
 *   KSPSetUp;VecSet 5
 *   KSPSolve 150
 *
 * This option may be requested from the command line by passing in the flag `-log_view :<somefile>.txt:ascii_flamegraph`.
 */
PetscErrorCode PetscLogView_Flamegraph(PetscViewer viewer)
{
  int                   nTimers = 0, i, nChildren;
  PetscIntStack         eventStack;
  PetscLogDouble        totalTime;
  PetscNestedEventTree *tree = NULL;
  PetscSortItem        *children;

  PetscFunctionBegin;
  PetscCall(PetscGetTotalTime(viewer, &totalTime));
  PetscCall(PetscLogNestedTreeCreate(viewer, &tree, &nTimers));
  /* We use an integer stack to keep track of parent event IDs */
  PetscCall(PetscIntStackCreate(&eventStack));

  /* Initialize the child events and write them recursively */
  PetscCall(PetscLogNestedTreeGetChildrenCount(tree, nTimers, -1, 0, &nChildren));
  PetscCall(PetscLogNestedTreeSetChildrenSortItems(viewer, tree, nTimers, -1, 0, nChildren, &children));
  for (i = 0; i < nChildren; i++) PetscCall(PetscLogNestedTreePrintFlamegraph(viewer, tree, nTimers, children[i].id, totalTime, eventStack));
  if (nChildren > 0) PetscCall(PetscFree(children));

  PetscCall(PetscLogNestedTreeDestroy(tree, nTimers));
  PetscCall(PetscIntStackDestroy(eventStack));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
