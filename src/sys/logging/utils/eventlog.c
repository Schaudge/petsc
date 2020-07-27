//#define _POSIX_C_SOURCE 199309L
#include <stdbool.h>
#include <semaphore.h>
#include "nvmlPower.hpp"
#include <sys/types.h>
#include <sys/syscall.h>

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
static bool pollThreadStatus = false;
static unsigned int deviceCount = 0;
static char deviceNameStr[64];
double consJoule;
static double IdlePower;
static int cuID;

static nvmlReturn_t nvmlResult;
static nvmlDevice_t *nvmlDeviceID;
static nvmlPciInfo_t nvmPCIInfo;
static nvmlEnableState_t pmmode;
static nvmlComputeMode_t computeMode;

static pthread_t powerPollThread;

static pthread_mutex_t lock;
static sem_t semb, semc;

/*
Poll the GPU using nvml APIs.
*/
void *powerPollingFunc(void *ptr)
{
	unsigned int powerLevel = 0;
	double joule = 0, meanWatt = 0;
	double begin, end;
	double time_spent;
	int nvml_err;
	int dev_id;
        bool start = true;

        begin = MPI_Wtime();
	dev_id = cuID;
	pthread_mutex_lock(&lock);   

	while (pollThreadStatus)
	{
		pthread_mutex_unlock(&lock);
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
		// Get the power management mode of the GPU.
		nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID[dev_id], &pmmode);

		// The following function may be utilized to handle errors as needed.
		if ((nvml_err = getNVMLError(nvmlResult)))
		{
			fprintf(stderr,"NVML error #%d\n",nvml_err);
			pthread_exit(0);
		}

		// Check if power management mode is enabled.
		if (pmmode == NVML_FEATURE_ENABLED)
		{
			// Get the power usage in milliWatts.
			nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID[dev_id], &powerLevel);
		}

		meanWatt = (powerLevel)/1000.0;
                if (start) {
                	IdlePower = meanWatt;
                        start = false;
                }

		end = MPI_Wtime();
		time_spent = (double)(end - begin);
		begin = end;
		joule = meanWatt * time_spent;

		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
		pthread_mutex_lock(&lock);

		consJoule += joule;
	}
	pthread_mutex_unlock(&lock);

	pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void nvmlAPIRun()
{
	int i;

	// Initialize nvml.
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	// Count the number of GPUs available.
	nvmlResult = nvmlDeviceGetCount(&deviceCount);
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	nvmlDeviceID = (nvmlDevice_t *) malloc(deviceCount*sizeof(nvmlDevice_t));	

	for (i = 0; i < deviceCount; i++)
	{
		// Get the device ID.
		nvmlResult = nvmlDeviceGetHandleByIndex(i, nvmlDeviceID+i);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the name of the device.
		nvmlResult = nvmlDeviceGetName(nvmlDeviceID[i], deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get PCI information of the device.
		nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID[i], &nvmPCIInfo);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		fprintf(stderr,"Found device %s with ID %d \n", deviceNameStr,i); 
		// Get the compute mode of the device which indicates CUDA capabilities.
		nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID[i], &computeMode);
		if (NVML_ERROR_NOT_SUPPORTED == nvmlResult)
		{
			printf("This is not a CUDA-capable device.\n");
		}
		else if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}
		
	}

	// This statement assumes that the first indexed GPU will be used.
	// If there are multiple GPUs that can be used by the system, this needs to be done with care.
	// Test thoroughly and ensure the correct device ID is being used.
	cudaGetDevice(&cuID);
	fprintf(stderr,"Using CUDA device with ID %d\n",cuID);
	nvmlResult = nvmlDeviceGetHandleByIndex(cuID, nvmlDeviceID+cuID);

	pollThreadStatus = true;

	const char *message = "Test";

        if (pthread_mutex_init(&lock, NULL) != 0)
        {
        	fprintf(stderr, "\n mutex init failed\n");
	}
        if((sem_init(&semb,0,0))!=0)
        	fprintf(stderr, "error in semb creation\n");
        if((sem_init(&semc,0,0))!=0)
        	fprintf(stderr, "error in semc creation\n");

	int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void*) message);
	if (iret)
	{
		fprintf(stderr,"Error - pthread_create() return code: %d\n",iret);
		exit(0);
	}
}

/*
End power measurement. This ends the polling thread.
*/
void nvmlAPIEnd()
{
	pthread_mutex_lock(&lock);
	pollThreadStatus = false;
	pthread_mutex_unlock(&lock);
	pthread_join(powerPollThread, NULL);

	nvmlResult = nvmlShutdown();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int getNVMLError(nvmlReturn_t resultToCheck)
{
	if (resultToCheck == NVML_ERROR_UNINITIALIZED)
		return 1;
	if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
		return 2;
	if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
		return 3;
	if (resultToCheck == NVML_ERROR_NO_PERMISSION)
		return 4;
	if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
		return 5;
	if (resultToCheck == NVML_ERROR_NOT_FOUND)
		return 6;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
		return 7;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
		return 8;
	if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
		return 9;
	if (resultToCheck == NVML_ERROR_TIMEOUT)
		return 10;
	if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
		return 11;
	if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
		return 12;
	if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
		return 13;
	if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
		return 14;
	if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
		return 15;
	if (resultToCheck == NVML_ERROR_UNKNOWN)
		return 16;

	return 0;
}

void nvmlAPIstart()
{
	pthread_mutex_lock(&lock);
	consJoule = 0;
	pthread_mutex_unlock(&lock);
	sem_post(&semc);
	sem_wait(&semb);
}

double nvmlAPIstop()
{
	pthread_mutex_lock(&lock);
	pthread_mutex_unlock(&lock);
        sem_wait(&semb);
	return consJoule;
}

double nvmlAPIcumul()
{
	double valret;
	pthread_mutex_lock(&lock);
        valret = consJoule;;
        pthread_mutex_unlock(&lock);
	return valret;
}

double nvmlAPIidle()
{
	return IdlePower;
}

/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h>  /*I    "petscsys.h"   I*/

PetscBool PetscLogSyncOn = PETSC_FALSE;
PetscBool PetscLogMemory = PETSC_FALSE;
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) 
PetscBool PetscLogGpuTraffic = PETSC_FALSE;
#endif

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*@C
  PetscEventRegLogCreate - This creates a PetscEventRegLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventRegLog

  Level: developer

.seealso: PetscEventRegLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *eventLog)
{
  PetscEventRegLog l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(&l);CHKERRQ(ierr);
  l->numEvents = 0;
  l->maxEvents = 100;
  ierr         = PetscMalloc1(l->maxEvents,&l->eventInfo);CHKERRQ(ierr);
  *eventLog    = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventRegLogDestroy - This destroys a PetscEventRegLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventRegLog

  Level: developer

.seealso: PetscEventRegLogCreate()
@*/
PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog eventLog)
{
  int            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    ierr = PetscFree(eventLog->eventInfo[e].name);CHKERRQ(ierr);
  }
  ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
  ierr = PetscFree(eventLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogCreate - This creates a PetscEventPerfLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventPerfLog

  Level: developer

.seealso: PetscEventPerfLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *eventLog)
{
  PetscEventPerfLog l;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(&l);CHKERRQ(ierr);
  l->numEvents = 0;
  l->maxEvents = 100;
  ierr         = PetscCalloc1(l->maxEvents,&l->eventInfo);CHKERRQ(ierr);
  *eventLog    = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDestroy - This destroys a PetscEventPerfLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventPerfLog

  Level: developer

.seealso: PetscEventPerfLogCreate()
@*/
PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog eventLog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
  ierr = PetscFree(eventLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscEventPerfInfoClear - This clears a PetscEventPerfInfo object.

  Not collective

  Input Parameter:
. eventInfo - The PetscEventPerfInfo

  Level: developer

.seealso: PetscEventPerfLogCreate()
@*/
PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *eventInfo)
{
  PetscFunctionBegin;
  eventInfo->id            = -1;
  eventInfo->active        = PETSC_TRUE;
  eventInfo->visible       = PETSC_TRUE;
  eventInfo->depth         = 0;
  eventInfo->count         = 0;
  eventInfo->flops         = 0.0;
  eventInfo->flops2        = 0.0;
  eventInfo->flopsTmp      = 0.0;
  eventInfo->time          = 0.0;
  eventInfo->time2         = 0.0;
  eventInfo->timeTmp       = 0.0;
  eventInfo->syncTime      = 0.0;
  eventInfo->dof[0]        = -1.0;
  eventInfo->dof[1]        = -1.0;
  eventInfo->dof[2]        = -1.0;
  eventInfo->dof[3]        = -1.0;
  eventInfo->dof[4]        = -1.0;
  eventInfo->dof[5]        = -1.0;
  eventInfo->dof[6]        = -1.0;
  eventInfo->dof[7]        = -1.0;
  eventInfo->errors[0]     = -1.0;
  eventInfo->errors[1]     = -1.0;
  eventInfo->errors[2]     = -1.0;
  eventInfo->errors[3]     = -1.0;
  eventInfo->errors[4]     = -1.0;
  eventInfo->errors[5]     = -1.0;
  eventInfo->errors[6]     = -1.0;
  eventInfo->errors[7]     = -1.0;
  eventInfo->numMessages   = 0.0;
  eventInfo->messageLength = 0.0;
  eventInfo->numReductions = 0.0;
  #if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) 
  eventInfo->CpuToGpuCount = 0.0;
  eventInfo->GpuToCpuCount = 0.0;
  eventInfo->CpuToGpuSize  = 0.0;
  eventInfo->GpuToCpuSize  = 0.0;
  eventInfo->GpuFlops      = 0.0;
  eventInfo->GpuTime       = 0.0;
  eventInfo->energyCons    = 0.0;
  #endif
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfInfoCopy - Copy the activity and visibility data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input PetscEventPerfInfo

  Output Parameter:
. outInfo   - The output PetscEventPerfInfo

  Level: developer

.seealso: PetscEventPerfInfoClear()
@*/
PetscErrorCode PetscEventPerfInfoCopy(PetscEventPerfInfo *eventInfo,PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->id      = eventInfo->id;
  outInfo->active  = eventInfo->active;
  outInfo->visible = eventInfo->visible;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogEnsureSize - This ensures that a PetscEventPerfLog is at least of a certain size.

  Not collective

  Input Parameters:
+ eventLog - The PetscEventPerfLog
- size     - The size

  Level: developer

.seealso: PetscEventPerfLogCreate()
@*/
PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog eventLog,int size)
{
  PetscEventPerfInfo *eventInfo;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while (size > eventLog->maxEvents) {
    ierr = PetscCalloc1(eventLog->maxEvents*2,&eventInfo);CHKERRQ(ierr);
    ierr = PetscArraycpy(eventInfo,eventLog->eventInfo,eventLog->maxEvents);CHKERRQ(ierr);
    ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
    eventLog->eventInfo  = eventInfo;
    eventLog->maxEvents *= 2;
  }
  while (eventLog->numEvents < size) {
    ierr = PetscEventPerfInfoClear(&eventLog->eventInfo[eventLog->numEvents++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPE)
#include <mpe.h>
PETSC_INTERN PetscErrorCode PetscLogMPEGetRGBColor(const char*[]);
PetscErrorCode PetscLogEventBeginMPE(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_begin,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndMPE(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_end,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscEventRegLogRegister - Registers an event for logging operations in an application code.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventLog
. ename    - The name associated with the event
- classid   - The classid associated to the class for this event

  Output Parameter:
. event    - The event

  Example of Usage:
.vb
      int USER_EVENT;
      PetscLogDouble user_event_flops;
      PetscLogEventRegister("User event name",0,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Notes:

  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  Level: developer

.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogFlops(),
          PetscEventLogActivate(), PetscEventLogDeactivate()
@*/
PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog eventLog,const char ename[],PetscClassId classid,PetscLogEvent *event)
{
  PetscEventRegInfo *eventInfo;
  char              *str;
  int               e;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(ename,2);
  PetscValidIntPointer(event,4);
  /* Should check classid I think */
  e = eventLog->numEvents++;
  if (eventLog->numEvents > eventLog->maxEvents) {
    ierr = PetscCalloc1(eventLog->maxEvents*2,&eventInfo);CHKERRQ(ierr);
    ierr = PetscArraycpy(eventInfo,eventLog->eventInfo,eventLog->maxEvents);CHKERRQ(ierr);
    ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
    eventLog->eventInfo  = eventInfo;
    eventLog->maxEvents *= 2;
  }
  ierr = PetscStrallocpy(ename,&str);CHKERRQ(ierr);

  eventLog->eventInfo[e].name       = str;
  eventLog->eventInfo[e].classid    = classid;
  eventLog->eventInfo[e].collective = PETSC_TRUE;
#if defined(PETSC_HAVE_MPE)
  if (PetscLogPLB == PetscLogEventBeginMPE) {
    const char  *color;
    PetscMPIInt rank;
    int         beginID,endID;

    beginID = MPE_Log_get_event_number();
    endID   = MPE_Log_get_event_number();

    eventLog->eventInfo[e].mpe_id_begin = beginID;
    eventLog->eventInfo[e].mpe_id_end   = endID;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscLogMPEGetRGBColor(&color);CHKERRQ(ierr);
      MPE_Describe_state(beginID,endID,str,(char*)color);
    }
  }
#endif
  *event = e;
  PetscFunctionReturn(0);
}

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
/*@C
  PetscEventPerfLogActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventPerfLog
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with PetscEventRegLogRegister().

  Level: developer

.seealso: PetscEventPerfLogDeactivate()
@*/
PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog eventLog,PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventPerfLog
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with PetscEventRegLogRegister().

  Level: developer

.seealso: PetscEventPerfLogActivate()
@*/
PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog eventLog,PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The PetscEventPerfLog
. eventRegLog - The PetscEventRegLog
- classid      - The class id, for example MAT_CLASSID, SNES_CLASSID,

  Level: developer

.seealso: PetscEventPerfLogDeactivateClass(), PetscEventPerfLogActivate(), PetscEventPerfLogDeactivate()
@*/
PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog eventLog,PetscEventRegLog eventRegLog,PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The PetscEventPerfLog
. eventRegLog - The PetscEventRegLog
- classid - The class id, for example MAT_CLASSID, SNES_CLASSID,

  Level: developer

.seealso: PetscEventPerfLogDeactivateClass(), PetscEventPerfLogDeactivate(), PetscEventPerfLogActivate()
@*/
PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog eventLog,PetscEventRegLog eventRegLog,PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscEventRegLogGetEvent - This function returns the event id given the event name.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventRegLog
- name     - The stage name

  Output Parameter:
. event    - The event id, or -1 if not found

  Level: developer

.seealso: PetscEventRegLogRegister()
@*/
PetscErrorCode  PetscEventRegLogGetEvent(PetscEventRegLog eventLog,const char name[],PetscLogEvent *event)
{
  PetscBool      match;
  int            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(event,3);
  *event = -1;
  for (e = 0; e < eventLog->numEvents; e++) {
    ierr = PetscStrcasecmp(eventLog->eventInfo[e].name,name,&match);CHKERRQ(ierr);
    if (match) {
      *event = e;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogSetVisible - This function determines whether an event is printed during PetscLogView()

  Not Collective

  Input Parameters:
+ eventLog  - The PetscEventPerfLog
. event     - The event to log
- isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_view - Activates log summary

  Level: developer

.seealso: PetscEventPerfLogGetVisible(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog eventLog,PetscLogEvent event,PetscBool isVisible)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].visible = isVisible;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogGetVisible - This function returns whether an event is printed during PetscLogView()

  Not Collective

  Input Parameters:
+ eventLog  - The PetscEventPerfLog
- event     - The event id to log

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_view - Activates log summary

  Level: developer

.seealso: PetscEventPerfLogSetVisible(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog eventLog,PetscLogEvent event,PetscBool  *isVisible)
{
  PetscFunctionBegin;
  PetscValidIntPointer(isVisible,3);
  *isVisible = eventLog->eventInfo[event].visible;
  PetscFunctionReturn(0);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  Input Parameters:
+ stage - The stage number or PETSC_DETERMINE for the current stage
- event - The event number

  Output Parameters:
. info - This structure is filled with the performance information

  Level: Intermediate

.seealso: PetscLogEventGetFlops()
@*/
PetscErrorCode PetscLogEventGetPerfInfo(int stage,PetscLogEvent event,PetscEventPerfInfo *info)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidPointer(info,3);
  if (!PetscLogPLB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (stage < 0) {ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);}
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  *info = eventLog->eventInfo[event];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventGetFlops(PetscLogEvent event,PetscLogDouble *flops)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!PetscLogPLB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  ierr   = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr   = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr   = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  *flops = eventLog->eventInfo[event].flops;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventZeroFlops(PetscLogEvent event)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);

  eventLog->eventInfo[event].flops    = 0.0;
  eventLog->eventInfo[event].flops2   = 0.0;
  eventLog->eventInfo[event].flopsTmp = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventSynchronize(PetscLogEvent event,MPI_Comm comm)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscLogDouble    time = 0.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!PetscLogSyncOn || comm == MPI_COMM_NULL) PetscFunctionReturn(0);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  if (!eventRegLog->eventInfo[event].collective) PetscFunctionReturn(0);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  if (eventLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);

  PetscTimeSubtract(&time);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  PetscTimeAdd(&time);
  eventLog->eventInfo[event].syncTime += time;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  /* Synchronization */
  ierr = PetscLogEventSynchronize(event,PetscObjectComm(o1));CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth++;
  if (eventLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log the performance info */
  eventLog->eventInfo[event].count++;
  eventLog->eventInfo[event].timeTmp = 0.0;
  PetscTimeSubtract(&eventLog->eventInfo[event].timeTmp);
  eventLog->eventInfo[event].flopsTmp       = 0.0;
  eventLog->eventInfo[event].flopsTmp      -= petsc_TotalFlops;
  eventLog->eventInfo[event].numMessages   -= petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventLog->eventInfo[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventLog->eventInfo[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  if (PetscLogMemory) {
    PetscLogDouble usage;
    ierr = PetscMemoryGetCurrentUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].memIncrease -= usage;
    ierr = PetscMallocGetCurrentUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].mallocSpace -= usage;
    ierr = PetscMallocGetMaximumUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].mallocIncrease -= usage;
    ierr = PetscMallocPushMaximumUsage((int)event);CHKERRQ(ierr);
  }
  #if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) 
  eventLog->eventInfo[event].CpuToGpuCount -= petsc_ctog_ct;
  eventLog->eventInfo[event].GpuToCpuCount -= petsc_gtoc_ct;
  eventLog->eventInfo[event].CpuToGpuSize  -= petsc_ctog_sz;
  eventLog->eventInfo[event].GpuToCpuSize  -= petsc_gtoc_sz;
  eventLog->eventInfo[event].GpuFlops      -= petsc_gflops;
  eventLog->eventInfo[event].GpuTime       -= petsc_gtime;
  eventLog->eventInfo[event].energyCons    -= nvmlAPIcumul();
  //nvmlAPIstart();
  #endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndDefault(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth--;
  if (eventLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventLog->eventInfo[event].depth < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");
  /* Log performance info */
  PetscTimeAdd(&eventLog->eventInfo[event].timeTmp);
  eventLog->eventInfo[event].time          += eventLog->eventInfo[event].timeTmp;
  eventLog->eventInfo[event].time2         += eventLog->eventInfo[event].timeTmp*eventLog->eventInfo[event].timeTmp;
  eventLog->eventInfo[event].flopsTmp      += petsc_TotalFlops;
  eventLog->eventInfo[event].flops         += eventLog->eventInfo[event].flopsTmp;
  eventLog->eventInfo[event].flops2        += eventLog->eventInfo[event].flopsTmp*eventLog->eventInfo[event].flopsTmp;
  eventLog->eventInfo[event].numMessages   += petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventLog->eventInfo[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventLog->eventInfo[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  if (PetscLogMemory) {
    PetscLogDouble usage,musage;
    ierr = PetscMemoryGetCurrentUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].memIncrease += usage;
    ierr = PetscMallocGetCurrentUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].mallocSpace += usage;
    ierr = PetscMallocPopMaximumUsage((int)event,&musage);CHKERRQ(ierr);
    eventLog->eventInfo[event].mallocIncreaseEvent = PetscMax(musage-usage,eventLog->eventInfo[event].mallocIncreaseEvent);
    ierr = PetscMallocGetMaximumUsage(&usage);CHKERRQ(ierr);
    eventLog->eventInfo[event].mallocIncrease += usage;
  }
  #if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) 
  eventLog->eventInfo[event].CpuToGpuCount += petsc_ctog_ct;
  eventLog->eventInfo[event].GpuToCpuCount += petsc_gtoc_ct;
  eventLog->eventInfo[event].CpuToGpuSize  += petsc_ctog_sz;
  eventLog->eventInfo[event].GpuToCpuSize  += petsc_gtoc_sz;
  eventLog->eventInfo[event].GpuFlops      += petsc_gflops;
  eventLog->eventInfo[event].GpuTime       += petsc_gtime;
  eventLog->eventInfo[event].energyCons    += nvmlAPIcumul();
  #endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action            *tmpAction;
  PetscLogDouble    start,end;
  PetscLogDouble    curTime;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscCalloc1(petsc_maxActions*2,&tmpAction);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpAction,petsc_actions,petsc_maxActions);CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  PetscTime(&curTime);
  if (petsc_logActions) {
    petsc_actions[petsc_numActions].time    = curTime - petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = ACTIONBEGIN;
    petsc_actions[petsc_numActions].event   = event;
    petsc_actions[petsc_numActions].classid = eventRegLog->eventInfo[event].classid;
    if (o1) petsc_actions[petsc_numActions].id1 = o1->id;
    else petsc_actions[petsc_numActions].id1 = -1;
    if (o2) petsc_actions[petsc_numActions].id2 = o2->id;
    else petsc_actions[petsc_numActions].id2 = -1;
    if (o3) petsc_actions[petsc_numActions].id3 = o3->id;
    else petsc_actions[petsc_numActions].id3 = -1;
    petsc_actions[petsc_numActions].flops = petsc_TotalFlops;

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          -= curTime;
  eventPerfLog->eventInfo[event].flops         -= petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   -= petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndComplete(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action            *tmpAction;
  PetscLogDouble    start,end;
  PetscLogDouble    curTime;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscCalloc1(petsc_maxActions*2,&tmpAction);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpAction,petsc_actions,petsc_maxActions);CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  PetscTime(&curTime);
  if (petsc_logActions) {
    petsc_actions[petsc_numActions].time    = curTime - petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = ACTIONEND;
    petsc_actions[petsc_numActions].event   = event;
    petsc_actions[petsc_numActions].classid = eventRegLog->eventInfo[event].classid;
    if (o1) petsc_actions[petsc_numActions].id1 = o1->id;
    else petsc_actions[petsc_numActions].id1 = -1;
    if (o2) petsc_actions[petsc_numActions].id2 = o2->id;
    else petsc_actions[petsc_numActions].id2 = -1;
    if (o3) petsc_actions[petsc_numActions].id3 = o3->id;
    else petsc_actions[petsc_numActions].id3 = -1;
    petsc_actions[petsc_numActions].flops = petsc_TotalFlops;

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventPerfLog->eventInfo[event].depth < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          += curTime;
  eventPerfLog->eventInfo[event].flops         += petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   += petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  PetscMPIInt       rank;
  int               stage,err;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!petsc_tracetime) PetscTime(&petsc_tracetime);

  petsc_tracelevel++;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log performance info */
  PetscTime(&cur_time);
  ierr = PetscFPrintf(PETSC_COMM_SELF,petsc_tracefile,"%s[%d] %g Event begin: %s\n",petsc_tracespace,rank,cur_time-petsc_tracetime,eventRegLog->eventInfo[event].name);CHKERRQ(ierr);
  ierr = PetscStrncpy(petsc_tracespace,petsc_traceblanks,2*petsc_tracelevel);CHKERRQ(ierr);

  petsc_tracespace[2*petsc_tracelevel] = 0;

  err = fflush(petsc_tracefile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndTrace(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  int               stage,err;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  petsc_tracelevel--;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventPerfLog->eventInfo[event].depth < 0 || petsc_tracelevel < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  if (petsc_tracelevel) {
    ierr = PetscStrncpy(petsc_tracespace,petsc_traceblanks,2*petsc_tracelevel);CHKERRQ(ierr);
  }
  petsc_tracespace[2*petsc_tracelevel] = 0;
  PetscTime(&cur_time);
  ierr = PetscFPrintf(PETSC_COMM_SELF,petsc_tracefile,"%s[%d] %g Event end: %s\n",petsc_tracespace,rank,cur_time-petsc_tracetime,eventRegLog->eventInfo[event].name);CHKERRQ(ierr);
  err  = fflush(petsc_tracefile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  PetscFunctionReturn(0);
}

/*@C
  PetscLogEventSetDof - Set the nth number of degrees of freedom associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The dof index, in [0, 8)
- dof   - The number of dofs

  Database Options:
. -log_view - Activates log summary

  Note: This is to enable logging of convergence

  Level: developer

.seealso: PetscLogEventSetError(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscLogEventSetDof(PetscLogEvent event, PetscInt n, PetscLogDouble dof)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if ((n < 0) || (n > 7)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %D is not in [0, 8)", n);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  eventLog->eventInfo[event].dof[n] = dof;
  PetscFunctionReturn(0);
}

/*@C
  PetscLogEventSetError - Set the nth error associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The error index, in [0, 8)
- error - The error

  Database Options:
. -log_view - Activates log summary

  Note: This is to enable logging of convergence, and enable users to interpret the errors as they wish. For example,
  as different norms, or as errors for different fields

  Level: developer

.seealso: PetscLogEventSetDof(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscLogEventSetError(PetscLogEvent event, PetscInt n, PetscLogDouble error)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if ((n < 0) || (n > 7)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %D is not in [0, 8)", n);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  eventLog->eventInfo[event].errors[n] = error;
  PetscFunctionReturn(0);
}
