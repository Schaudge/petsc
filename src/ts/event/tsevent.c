#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

/*
  TSEventInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventInitialize(TSEvent event,TS ts,PetscReal t,Vec U)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(0);
  PetscValidPointer(event,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  event->ptime_left = t;
  event->iterctr    = 0;
  ierr = (*event->eventdetector)(ts,t,U,event->fvalue_prev,event->ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  if (!*event) PetscFunctionReturn(0);
  if (--(*event)->refct > 0) {*event = 0; PetscFunctionReturn(0);}

  ierr = PetscFree4((*event)->fvalue,(*event)->fvalue_prev,(*event)->fvalue_right,(*event)->zerocrossing);CHKERRQ(ierr);
  ierr = PetscFree4((*event)->side,(*event)->direction,(*event)->terminate,(*event)->events_zero);CHKERRQ(ierr);
  for (i=0; i < (*event)->recsize; i++) {
    ierr = PetscFree((*event)->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree4((*event)->recorder.eventidx,(*event)->recorder.nevents,(*event)->recorder.stepnum,(*event)->recorder.time);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*event)->monitor);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSSetPostEventTimeStepScale - Set the time-step to be used immediately following a detected event by scaling from the time step before the event was detected

  Logically Collective

  Input Arguments:
+ ts - time integration context
- dtscale - post event interval scaling factor

  Options Database Keys:
. -ts_event_post_dt_scale <dtscale> -  scaling of time-step after event

  Notes:
  The post event time-step should be selected based on the dynamics following the event.
  If the dynamics are stiff, a conservative (small) value should be used.
  If not, then a larger value can be used. TSSetPostEventTimeStep() can be used to set a particular given value for the post event timestep

  Level: advanced

  .seealso: TS, TSEvent, TSSetEventHandler(), TSSetPostEventTimeStep()
@*/
PetscErrorCode TSSetPostEventTimeStepScale(TS ts,PetscReal dtscale)
{
  PetscFunctionBegin;
  if (dtscale <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Post event timestep scale must be positive");
  ts->event->postevent_dtscale = dtscale;
  PetscFunctionReturn(0);
}

/*@
  TSSetPostEventTimeStep - Set the time-step to be used immediately following a detected event

  Logically Collective

  Input Arguments:
+ ts - time integration context
- dt - post event time step

  Options Database Keys:
. -ts_event_post_dt <dt> -  time-step after event interval

  Notes:
  The post event time-step should be selected based on the dynamics following the event.
  If the dynamics are stiff, a conservative (small) step should be used.
  If not, then a larger time-step can be used. TSSetPostEventTimeStepScale() can be used to set a value for the post event timestep based
  on the timestep before the event was detected

  Level: advanced

  .seealso: TS, TSEvent, TSSetEventHandler(), TSSetPostEventTimeStepScale()
@*/
PetscErrorCode TSSetPostEventTimeStep(TS ts,PetscReal dt)
{
  PetscFunctionBegin;
  if (dt <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Post event timestep must be positive");
  ts->event->postevent_dt = dt;
  PetscFunctionReturn(0);
}

/*@
   TSSetEventTolerance - Set tolerance for event zero crossings, this is relative to the current time-step size

   Logically Collective

   Input Arguments:
+  ts - time integration context
-  tol - scalar tolerance

   Options Database Keys:
.  -ts_event_tol <tol> tolerance for event zero crossing

   Notes:
   This tolerance is the relative size compare to the initial timestep of the interval that captures the sign change of the event
   Must call TSSetEventHandler() before setting the tolerances.

  The default is 1.e-3

   Level: beginner

.seealso: TS, TSEvent, TSSetEventHandler()
@*/
PetscErrorCode TSSetEventTolerance(TS ts,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set the events first by calling TSSetEventHandler()");
  ts->event->tol = tol;
  PetscFunctionReturn(0);
}

/*@C
   TSSetEventHandler - Sets a function used for detecting events and for handling it (making chanages in the problem being solved)

   Logically Collective on TS

   Input Parameters:
+  ts - the TS context obtained from TSCreate()
.  nevents - number of local events
.  direction - direction of zero crossing to be detected. -1 => Zero crossing in negative direction,
               +1 => Zero crossing in positive direction, 0 => both ways (one for each event)
.  terminate - flag to indicate whether time stepping should be terminated after
               event is detected (one for each event)
.  eventdetector - function that defines the zero crossing
.  eventhandler - [optional] this is called after the an event has been detected
-  ctx       - [optional] user-defined context for private data for the event detector and and handler (use NULL if no
               context is desired)

   Options Database Keys:
+   -ts_event_tol tol - Event tolerance for zero crossing check
.   -ts_event_monitor - Prints events detected and choices made by event detector
.   -ts_event_recorder_initial_size - Initial size of event recorder
.   -ts_event_post_dt dt - Time step to use after event detected, see TSSetEventPostTimeStep()
-   -ts_event_post_dt_scale scale - Scaling of time step before event to use after event, see TSSetEventPostTimeStepScale()

   Calling sequence of eventdetector:
   PetscErrorCode eventdetector(TS ts,PetscReal t,Vec U,PetscScalar fvalue[],void* ctx)

   Input Parameters:
+  ts  - the TS context
.  t   - current time
.  U   - current iterate
-  ctx - [optional] context passed with TSSetEventHandler()

   Output parameters:
.  fvalue    - function value of events at time t (one for each event)

   Calling sequence of eventhandler:
   PetscErrorCode eventhandler(TS ts,PetscInt nevents_zero,PetscInt events_zero[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)

   Input Parameters:
+  ts - the TS context
.  nevents_zero - number of local events
.  events_zero  - indices of local events
.  t            - current time
.  U            - current solution
.  forwardsolve - Flag to indicate whether TS is doing a forward solve (1) or adjoint solve (0)
-  ctx          - the context passed with TSSetEventHandler()

   Level: intermediate

.seealso: TSCreate(), TSSetTimeStep(), TSSetConvergedReason(), TSSetEventTolerance(), TSSetEventPostTimeStep(), TSSetEventPostTimeStepScale()
@*/
PetscErrorCode TSSetEventHandler(TS ts,PetscInt nevents,PetscInt direction[],PetscBool terminate[],PetscErrorCode (*eventdetector)(TS,PetscReal,Vec,PetscScalar[],void*),PetscErrorCode (*eventhandler)(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*),void *ctx)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscInt       i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (nevents) {
    PetscValidIntPointer(direction,2);
    PetscValidIntPointer(terminate,3);
  }

  ierr = PetscNewLog(ts,&event);CHKERRQ(ierr);
  ierr = PetscMalloc4(nevents,&event->fvalue,nevents,&event->fvalue_prev,nevents,&event->fvalue_right,nevents,&event->zerocrossing);CHKERRQ(ierr);
  ierr = PetscMalloc4(nevents,&event->side,nevents,&event->direction,nevents,&event->terminate,nevents,&event->events_zero);CHKERRQ(ierr);
  for (i=0; i < nevents; i++) {
    event->direction[i]    = direction[i];
    event->terminate[i]    = terminate[i];
    event->zerocrossing[i] = PETSC_FALSE;
    event->side[i]         = 0;
  }
  event->nevents           = nevents;
  event->eventhandler      = eventhandler;
  event->eventdetector     = eventdetector;
  event->ctx               = ctx;
  event->postevent_dt      = PETSC_DECIDE;
  event->postevent_dtscale = 1.0;
  event->prev_dt           = ts->time_step;

  event->recsize = 8;  /* Initial size of the recorder */
  ierr = PetscOptionsBegin(((PetscObject)ts)->comm,((PetscObject)ts)->prefix,"TS Event options","TS");CHKERRQ(ierr);
  {
    event->tol = 1.e-3;
    ierr = PetscOptionsReal("-ts_event_tol","Scalar event tolerance for zero crossing check","TSSetEventTolerance",event->tol,&event->tol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-ts_event_monitor","Print choices made by event handler","",&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_event_recorder_initial_size","Initial size of event recorder","",event->recsize,&event->recsize,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_event_post_dt","Time step after event detected","TSSetEventPostTimeStep",event->postevent_dt,&event->postevent_dt,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_event_post_dt_scale","Time step after event detected","TSSetEventPostTimeStepScale",event->postevent_dtscale,&event->postevent_dtscale,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc4(event->recsize,&event->recorder.time,event->recsize,&event->recorder.stepnum,event->recsize,&event->recorder.nevents,event->recsize,&event->recorder.eventidx);CHKERRQ(ierr);
  for (i=0; i < event->recsize; i++) {
    ierr = PetscMalloc1(event->nevents,&event->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  if (flg) {ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,"stdout",&event->monitor);CHKERRQ(ierr);}

  ierr = TSEventDestroy(&ts->event);CHKERRQ(ierr);
  ts->event = event;
  ts->event->refct = 1;
  PetscFunctionReturn(0);
}

/*
  TSEventRecorderResize - Resizes (2X) the event recorder arrays whenever the recording limit (event->recsize) is reached.
*/
static PetscErrorCode TSEventRecorderResize(TSEvent event)
{
  PetscErrorCode ierr;
  PetscReal      *time;
  PetscInt       *stepnum;
  PetscInt       *nevents;
  PetscInt       **eventidx;
  PetscInt       i,fact=2;

  PetscFunctionBegin;

  /* Create large arrays */
  ierr = PetscMalloc4(fact*event->recsize,&time,fact*event->recsize,&stepnum,fact*event->recsize,&nevents,fact*event->recsize,&eventidx);CHKERRQ(ierr);
  for (i=0; i < fact*event->recsize; i++) {
    ierr = PetscMalloc1(event->nevents,&eventidx[i]);CHKERRQ(ierr);
  }

  /* Copy over data */
  ierr = PetscArraycpy(time,event->recorder.time,event->recsize);CHKERRQ(ierr);
  ierr = PetscArraycpy(stepnum,event->recorder.stepnum,event->recsize);CHKERRQ(ierr);
  ierr = PetscArraycpy(nevents,event->recorder.nevents,event->recsize);CHKERRQ(ierr);
  for (i=0; i < event->recsize; i++) {
    ierr = PetscArraycpy(eventidx[i],event->recorder.eventidx[i],event->recorder.nevents[i]);CHKERRQ(ierr);
  }

  /* Destroy old arrays */
  for (i=0; i < event->recsize; i++) {
    ierr = PetscFree(event->recorder.eventidx[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(event->recorder.eventidx,event->recorder.nevents,event->recorder.stepnum,event->recorder.time);CHKERRQ(ierr);

  /* Set pointers */
  event->recorder.time     = time;
  event->recorder.stepnum  = stepnum;
  event->recorder.nevents  = nevents;
  event->recorder.eventidx = eventidx;

  event->recsize *= fact;
  PetscFunctionReturn(0);
}

/*
   Helper routine to manage user handler and recording
*/
static PetscErrorCode TSEventHandle(TS ts,PetscReal t,Vec U)
{
  PetscErrorCode ierr;
  TSEvent        event = ts->event;
  PetscBool      terminate = PETSC_FALSE;
  PetscBool      restart = PETSC_FALSE;
  PetscInt       i,ctr,stepnum;
  PetscBool      inflag[2],outflag[2];
  PetscBool      forwardsolve = PETSC_TRUE; /* Flag indicating that TS is doing a forward solve */

  PetscFunctionBegin;
  if (event->eventhandler) {
    PetscObjectState state_prev,state_post;
    ierr = PetscObjectStateGet((PetscObject)U,&state_prev);CHKERRQ(ierr);
    ierr = (*event->eventhandler)(ts,event->nevents_zero,event->events_zero,t,U,forwardsolve,event->ctx);CHKERRQ(ierr);
    ierr = PetscObjectStateGet((PetscObject)U,&state_post);CHKERRQ(ierr);
    if (state_prev != state_post) restart = PETSC_TRUE;
  }

  /* Handle termination events and step restart */
  for (i=0; i<event->nevents_zero; i++) if (event->terminate[event->events_zero[i]]) terminate = PETSC_TRUE;
  inflag[0] = restart; inflag[1] = terminate;
  ierr = MPIU_Allreduce(inflag,outflag,2,MPIU_BOOL,MPI_LOR,((PetscObject)ts)->comm);CHKERRQ(ierr);
  restart = outflag[0]; terminate = outflag[1];
  if (restart) {ierr = TSRestartStep(ts);CHKERRQ(ierr);}
  if (terminate) {ierr = TSSetConvergedReason(ts,TS_CONVERGED_EVENT);CHKERRQ(ierr);}

  /* Reset event residual functions as states might get changed by the postevent callback */
  if (event->eventhandler) {
    ierr = VecLockReadPush(U);CHKERRQ(ierr);
    ierr = (*event->eventdetector)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);
    ierr = VecLockReadPop(U);CHKERRQ(ierr);
  }

  /* Cache current time and event residual functions */
  event->ptime_left = t;
  for (i=0; i<event->nevents; i++) event->fvalue_prev[i] = event->fvalue[i];

  /* Record the event in the event recorder */
  ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
  ctr = event->recorder.ctr;
  if (ctr == event->recsize) {
    ierr = TSEventRecorderResize(event);CHKERRQ(ierr);
  }
  event->recorder.time[ctr]    = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (i=0; i<event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(0);
}

/* Uses Anderson-Bjorck variant of regula falsi method */
PETSC_STATIC_INLINE PetscReal TSEventComputeStepSize(PetscReal tleft,PetscReal t,PetscReal tright,PetscScalar fleft,PetscScalar f,PetscScalar fright,PetscInt side,PetscReal dt)
{
  PetscReal new_dt, scal = 1.0;
  if (PetscRealPart(fleft)*PetscRealPart(f) < 0) {
    if (side == 1) {
      scal = (PetscRealPart(fright) - PetscRealPart(f))/PetscRealPart(fright);
      if (scal < PETSC_SMALL) scal = 0.5;
    }
    new_dt = (scal*PetscRealPart(fleft)*t - PetscRealPart(f)*tleft)/(scal*PetscRealPart(fleft) - PetscRealPart(f)) - tleft;
  } else {
    if (side == -1) {
      scal = (PetscRealPart(fleft) - PetscRealPart(f))/PetscRealPart(fleft);
      if (scal < PETSC_SMALL) scal = 0.5;
    }
    new_dt = (PetscRealPart(f)*tright - scal*PetscRealPart(fright)*t)/(PetscRealPart(f) - scal*PetscRealPart(fright)) - t;
  }
  return PetscMin(dt,new_dt);
}

/*
    TSEventDetector - Manages detecting events

     The behavior depends on event->status

     TSEVENT_NONE - no events currently detected, so checks if the current time-step includes any events
     TSEVENT_PROCESSING - an interval was detected, now it is running the time-stepper to reduce the size of the interval and locate the event

   Will not detect multiple events that are closer together than the time-step; if the user could provide at each time a count of the number of events up to the that time then
   the detector could utilize this information to decrease the time-step until it detects all the events.

*/
PetscErrorCode TSEventDetector(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event = ts->event;
  PetscReal      t;
  Vec            U;
  PetscInt       i,anyeventsfound;
  PetscReal      dt,dt_min;
  PetscInt       rollback=0,in[2],out[2];
  PetscInt       fvalue_sign,fvalueprev_sign;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (event) PetscFunctionReturn(0);

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);

  event->nevents_zero = 0;
  event->nevents_zero = 0;

  /* detect any crossing that occur exactly on the current time step */
  event->status = TSEVENT_NONE;
  for (i=0; i < event->nevents; i++) {
    if (event->fvalue[i] == 0.0) {
      event->events_zero[event->nevents_zero++] = i;
      if (event->monitor) {
        ierr = PetscViewerASCIIPrintf(event->monitor,"    TSEvent: Event %D zero crossing at time %18.16e located in %D iterations exactly on timestep\n",i,(double)t,event->iterctr);CHKERRQ(ierr);
      }
      event->zerocrossing[i] = PETSC_FALSE;
      event->side[i]         = 0;
    } else if (event->zerocrossing[i]) {
      event->status = TSEVENT_PROCESSING;
    }
  }

  /* if the current time interval is smaller than tolerance find all crossings in that interval; cannot use dt because that is for taking the next step */
  if ((event->status == TSEVENT_PROCESSING) && ((ts->ptime - ts->ptime_prev) < event->tol*event->prev_dt)) {
    event->status = TSEVENT_NONE;
    for (i=0; i < event->nevents; i++) {
      if (event->zerocrossing[i]) {
        event->events_zero[event->nevents_zero++] = i;
        if (event->monitor) {
          ierr = PetscViewerASCIIPrintf(event->monitor,"    TSEvent: Event %D zero crossing at time %18.16e located in %D iterations, interval length %g\n",i,(double)t,event->iterctr,(double)(ts->ptime - ts->ptime_prev));CHKERRQ(ierr);
        }
        event->zerocrossing[i] = PETSC_FALSE;
        event->side[i]         = 0;
      }
    }
  }

  /*  remove any the zero crossing detected */
  in[0] = event->status; in[1] = event->nevents_zero;
  ierr = MPIU_Allreduce(in,out,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  event->status = (TSEventStatus)out[0]; anyeventsfound = out[1];
  if (anyeventsfound) {
    ierr = TSEventHandle(ts,t,U);CHKERRQ(ierr);
    dt   = event->postevent_dt == PETSC_DECIDE ? event->prev_dt : PetscMin(event->postevent_dt,event->prev_dt);
    dt   = event->postevent_dtscale*dt;
    if (event->monitor) {
      ierr = PetscViewerASCIIPrintf(event->monitor,"    TSEvent: Setting dt to %18.16e after events located\n",(double)dt);CHKERRQ(ierr);
    }
    if (event->status == TSEVENT_NONE) {
      ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
      event->prev_dt    = dt;   /* last regular timestep before event detected */
      event->ptime_left = t;    /* last current time before event detected and after event detected it is left end of interval containing zero crossings */
      event->iterctr    = 0;
      PetscFunctionReturn(0);
    }
  }

  /* evalute the user sign crossing functions at current time */
  ierr = VecLockReadPush(U);CHKERRQ(ierr);
  ierr = (*event->eventdetector)(ts,t,U,event->fvalue,event->ctx);CHKERRQ(ierr);
  ierr = VecLockReadPop(U);CHKERRQ(ierr);

  /* check for any events that have sign crossings between current time and previous time */
  for (i=0; i < event->nevents; i++) {
    fvalue_sign     = PetscSign(PetscRealPart(event->fvalue[i]));
    fvalueprev_sign = PetscSign(PetscRealPart(event->fvalue_prev[i]));
    if (fvalueprev_sign != 0 && (fvalue_sign != fvalueprev_sign)) {
      if ((event->direction[i] < 0 && fvalue_sign < 0) || (event->direction[i] > 0 && fvalue_sign > 0) || !event->direction[i]) { 
        rollback = 1;
        dt = TSEventComputeStepSize(event->ptime_left,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);
        if (event->monitor) {
          ierr = PetscViewerASCIIPrintf(event->monitor,"    TSEvent: iter %D - Event %D interval detected [%18.16e - %18.16e] direction %D\n",event->iterctr,i,(double)event->ptime_left,(double)t,event->direction[i]);CHKERRQ(ierr);
        }
        event->fvalue_right[i] = event->fvalue[i];
        event->side[i]         = 1;
        event->zerocrossing[i] = PETSC_TRUE;
      }
    }
  }

  ierr = MPIU_Allreduce(MPI_IN_PLACE,&rollback,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  if (rollback) {
    ierr = TSRollBack(ts);CHKERRQ(ierr);
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_ITERATING);CHKERRQ(ierr);
    event->status = TSEVENT_PROCESSING;
    event->ptime_right = t;
  } else if (event->status == TSEVENT_PROCESSING) {
     if (event->monitor) {
       ierr = PetscViewerASCIIPrintf(event->monitor,"    TSEvent: iter %D - Stepping forward as no event detected in interval [%18.16e - %18.16e]\n",event->iterctr,(double)event->ptime_left,(double)t);CHKERRQ(ierr);
    }
    for (i=0; i < event->nevents; i++) {
      if (event->zerocrossing[i]) {
        dt = TSEventComputeStepSize(event->ptime_left,t,event->ptime_right,event->fvalue_prev[i],event->fvalue[i],event->fvalue_right[i],event->side[i],dt);
        event->side[i] = -1;
      }
    }
  }
  ierr = MPIU_Allreduce(&dt,&dt_min,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt_min);CHKERRQ(ierr);

  for (i=0; i < event->nevents; i++) {
    event->fvalue_prev[i] = event->fvalue[i];
  }

  event->iterctr++;
  PetscFunctionReturn(0);
}

PetscErrorCode TSAdjointEventHandler(TS ts)
{
  PetscErrorCode ierr;
  TSEvent        event;
  PetscInt       step;
  PetscReal      t;
  Vec            U;
  PetscInt       ctr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->event) PetscFunctionReturn(0);
  event = ts->event;
  ierr  = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ctr   = event->recorder.ctr-1;
  if (ctr >= 0 && (step == event->recorder.stepnum[ctr])) {
    ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    if (event->eventhandler) {
      ierr = (*event->eventhandler)(ts,event->recorder.nevents[ctr],event->recorder.eventidx[ctr],t,U,PETSC_FALSE,event->ctx);CHKERRQ(ierr);
      event->recorder.ctr--;
    }
  }
  ierr = PetscBarrier((PetscObject)ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
