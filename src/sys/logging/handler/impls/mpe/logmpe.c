#include <petsc/private/loghandlerimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <mpe.h>

typedef struct _n_PetscEventMPE {
  int start;
  int final;
  int depth;
} PetscEventMPE;

PETSC_LOG_RESIZABLE_ARRAY(MPEArray, PetscEventMPE, void *, NULL, NULL, NULL);

typedef struct _n_PetscLogHandler_MPE *PetscLogHandler_MPE;

struct _n_PetscLogHandler_MPE {
  PetscLogMPEArray events;
};

static PetscErrorCode PetscLogHandlerContextCreate_MPE(PetscLogHandler_MPE *mpe_p)
{
  PetscLogHandler_MPE mpe;

  PetscFunctionBegin;
  PetscCall(PetscNew(mpe_p));
  mpe = *mpe_p;
  PetscCall(PetscLogMPEArrayCreate(128, &mpe->events));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_MPE(PetscLogHandler h)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE) h->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogMPEArrayDestroy(&mpe->events));
  PetscCall(PetscFree(mpe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define PETSC_RGB_COLORS_MAX 39
static const char *PetscLogMPERGBColors[PETSC_RGB_COLORS_MAX] = {"OliveDrab:      ", "BlueViolet:     ", "CadetBlue:      ", "CornflowerBlue: ", "DarkGoldenrod:  ", "DarkGreen:      ", "DarkKhaki:      ", "DarkOliveGreen: ",
                                                                 "DarkOrange:     ", "DarkOrchid:     ", "DarkSeaGreen:   ", "DarkSlateGray:  ", "DarkTurquoise:  ", "DeepPink:       ", "DarkKhaki:      ", "DimGray:        ",
                                                                 "DodgerBlue:     ", "GreenYellow:    ", "HotPink:        ", "IndianRed:      ", "LavenderBlush:  ", "LawnGreen:      ", "LemonChiffon:   ", "LightCoral:     ",
                                                                 "LightCyan:      ", "LightPink:      ", "LightSalmon:    ", "LightSlateGray: ", "LightYellow:    ", "LimeGreen:      ", "MediumPurple:   ", "MediumSeaGreen: ",
                                                                 "MediumSlateBlue:", "MidnightBlue:   ", "MintCream:      ", "MistyRose:      ", "NavajoWhite:    ", "NavyBlue:       ", "OliveDrab:      "};

static PetscErrorCode PetscLogMPEGetRGBColor_Internal(const char *str[])
{
  static int idx = 0;

  PetscFunctionBegin;
  *str = PetscLogMPERGBColors[idx];
  idx  = (idx + 1) % PETSC_RGB_COLORS_MAX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerMPECreateEvent(const char name[], PetscLogMPEArray array)
{
  PetscEventMPE mpe_event;
  PetscMPIInt   rank;

  PetscFunctionBegin;
  MPE_Log_get_state_eventIDs(&mpe_event.start, &mpe_event.final);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    const char *color;

    PetscCall(PetscLogMPEGetRGBColor_Internal(&color));
    MPE_Describe_state(mpe_event.start, mpe_event.final, name, (char *)color);
  }
  PetscCall(PetscLogMPEArrayPush(array, mpe_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerMPEUpdate(PetscLogHandler h)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE) h->ctx;
  PetscLogState state;
  PetscInt      num_events, num_events_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));
  PetscCall(PetscLogMPEArrayGetSize(mpe->events, &num_events_old, NULL));
  for (PetscInt i = num_events_old; i < num_events; i++) {
    PetscLogEventInfo event_info;

    PetscCall(PetscLogStateEventGetInfo(state, (PetscLogEvent)i, &event_info));
    PetscCall(PetscLogHandlerMPECreateEvent(event_info.name, mpe->events));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE) handler->ctx;
  PetscEventMPE     mpe_event;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerMPEUpdate(handler));
  PetscCall(PetscLogMPEArrayGet(mpe->events, event, &mpe_event));
  mpe_event.depth++;
  PetscCall(PetscLogMPEArraySet(mpe->events, event, mpe_event));
  if (mpe_event.depth == 1) PetscCall(MPE_Log_event(mpe_event.start, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_MPE mpe = (PetscLogHandler_MPE) handler->ctx;
  PetscEventMPE     mpe_event;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerMPEUpdate(handler));
  PetscCall(PetscLogMPEArrayGet(mpe->events, event, &mpe_event));
  mpe_event.depth--;
  PetscCall(PetscLogMPEArraySet(mpe->events, event, mpe_event));
  if (mpe_event.depth == 0) PetscCall(MPE_Log_event(mpe_event.final, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_MPE(MPI_Comm comm, PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler              = *handler_p;
  PetscCall(PetscLogHandlerContextCreate_MPE((PetscLogHandler_MPE *) &handler->ctx));
  handler->type        = PETSC_LOG_HANDLER_MPE;
  handler->Destroy     = PetscLogHandlerDestroy_MPE;
  handler->EventBegin  = PetscLogHandlerEventBegin_MPE;
  handler->EventEnd    = PetscLogHandlerEventEnd_MPE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
