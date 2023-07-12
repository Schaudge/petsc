
#include <petscviewer.h>
#include "lognested.h"
#include "xmlviewer.h"

PETSC_INTERN PetscErrorCode PetscLogHandlerNestedSetThreshold(PetscLogHandler h, PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;

  PetscFunctionBegin;
  if (oldThresh) *oldThresh = nested->threshold;
  if (newThresh == (PetscLogDouble)PETSC_DECIDE) newThresh = 0.01;
  if (newThresh == (PetscLogDouble)PETSC_DEFAULT) newThresh = 0.01;
  nested->threshold = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventGetNestedEvent(PetscLogHandler h, PetscLogEvent e, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;
  PetscLogState          state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromEvent(e);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event
    char              name[BUFSIZ];
    PetscLogEventInfo event_info;
    PetscLogEventInfo nested_event_info;

    PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
    PetscCall(PetscLogStateEventGetInfo(nested->state, key.root, &nested_event_info));
    PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, event_info.name));
    PetscCall(PetscLogStateEventRegister(nested->state, name, event_info.classid, nested_event));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStageGetNestedEvent(PetscLogHandler h, PetscLogStage stage, PetscLogEvent *nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  NestedIdPair           key;
  PetscHashIter          iter;
  PetscBool              missing;
  PetscLogState          state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromStage(stage);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    PetscLogStageInfo stage_info;
    char              name[BUFSIZ];

    PetscCall(PetscLogStateStageGetInfo(state, stage, &stage_info));
    if (key.root >= 0) {
      PetscLogEventInfo nested_event_info;

      PetscCall(PetscLogStateEventGetInfo(nested->state, key.root, &nested_event_info));
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s;%s", nested_event_info.name, stage_info.name));
    } else {
      PetscCall(PetscSNPrintf(name, sizeof(name) - 1, "%s", stage_info.name));
    }
    PetscCall(PetscLogStateEventRegister(nested->state, name, nested->nested_stage_id, nested_event));
    PetscCall(PetscNestedHashIterSet(nested->pair_map, iter, *nested_event));
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, nested_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCheckNested(PetscLogHandler h, NestedId leaf, PetscLogEvent nested_event)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  NestedIdPair           key;
  NestedId               val;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = leaf;
  PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
  PetscCheck(val == nested_event, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging events and stages are not nested, nested logging cannot be used");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_Nested(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscLogEventGetNestedEvent(h, e, &nested_event));
  PetscCall(PetscLogHandlerEventBegin(nested->handler, nested_event, o1, o2, o3, o4));
  PetscCall(PetscIntStackPush(nested->stack, nested_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Nested(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_event));
  if (PetscDefined(USE_DEBUG)) PetscCall(PetscLogNestedCheckNested(h, NestedIdFromEvent(e), nested_event));
  PetscCall(PetscLogHandlerEventEnd(nested->handler, nested_event, o1, o2, o3, o4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventSync_Nested(PetscLogHandler h, PetscLogEvent e, MPI_Comm comm)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  if (!nested->handler->eventSync) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventGetNestedEvent(h, e, &nested_event));
  PetscCall(PetscLogHandlerEventSync(nested->handler, nested_event, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Nested(PetscLogHandler h, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  if (nested->nested_stage_id == -1) PetscCall(PetscClassIdRegister("LogNestedStage", &nested->nested_stage_id));
  PetscCall(PetscLogStageGetNestedEvent(h, stage, &nested_event));
  PetscCall(PetscLogHandlerEventBegin(nested->handler, nested_event, NULL, NULL, NULL, NULL));
  PetscCall(PetscIntStackPush(nested->stack, nested_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Nested(PetscLogHandler h, PetscLogStage stage)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;
  PetscLogEvent          nested_event;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_event));
  if (PetscDefined(USE_DEBUG)) PetscCall(PetscLogNestedCheckNested(h, NestedIdFromStage(stage), nested_event));
  PetscCall(PetscLogHandlerEventEnd(nested->handler, nested_event, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextCreate_Nested(MPI_Comm comm, PetscLogHandler_Nested *nested_p)
{
  PetscLogStage          root_stage;
  PetscLogHandler_Nested nested;

  PetscFunctionBegin;
  PetscCall(PetscNew(nested_p));
  nested = *nested_p;
  PetscCall(PetscLogStateCreate(&nested->state));
  PetscCall(PetscIntStackCreate(&nested->stack));
  nested->nested_stage_id = -1;
  nested->threshold       = 0.01;
  PetscCall(PetscNestedHashCreate(&nested->pair_map));
  PetscCall(PetscLogHandlerCreate_Default(comm, &nested->handler));
  PetscCall(PetscLogHandlerSetState(nested->handler, nested->state));
  PetscCall(PetscLogStateStageRegister(nested->state, "", &root_stage));
  PetscAssert(root_stage == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "root stage not zero");
  PetscCall(PetscLogHandlerStagePush(nested->handler, root_stage));
  PetscCall(PetscLogStateStagePush(nested->state, root_stage));
  PetscCall(PetscIntStackPush(nested->stack, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Nested(PetscLogHandler h, PetscObject obj)
{
  PetscClassId           classid;
  PetscInt               num_registered, num_nested_registered;
  PetscLogState          state;
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;

  PetscFunctionBegin;
  // register missing objects
  PetscCall(PetscObjectGetClassId(obj, &classid));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumClasses(nested->state, &num_nested_registered));
  PetscCall(PetscLogStateGetNumClasses(state, &num_registered));
  for (PetscLogClass c = num_nested_registered; c < num_registered; c++) {
    PetscLogClassInfo class_info;
    PetscLogClass     nested_c;

    PetscCall(PetscLogStateClassGetInfo(state, c, &class_info));
    PetscCall(PetscLogStateClassRegister(nested->state, class_info.name, class_info.classid, &nested_c));
  }
  if (nested->handler->objectCreate) PetscCall(PetscLogHandlerObjectCreate(nested->handler, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Nested(PetscLogHandler h, PetscObject obj)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;

  PetscFunctionBegin;
  if (nested->handler->objectDestroy) PetscCall(PetscLogHandlerObjectDestroy(nested->handler, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Nested(PetscLogHandler h)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)h->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateStagePop(nested->state));
  PetscCall(PetscLogHandlerStagePop(nested->handler, 0));
  PetscCall(PetscLogStateDestroy(&nested->state));
  PetscCall(PetscIntStackDestroy(nested->stack));
  PetscCall(PetscNestedHashDestroy(&nested->pair_map));
  PetscCall(PetscLogHandlerDestroy(&nested->handler));
  PetscCall(PetscFree(nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;
  PetscFunctionBegin;

  node      = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt             num_this_descendants = 0;
      PetscNestedEventNode tmp                  = tree[start_loc];
      tree[start_loc]                           = tree[node];
      tree[node]                                = tmp;
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

static PetscErrorCode PetscLogNestedCreatePerfNodes(MPI_Comm comm, PetscLogHandler_Nested nested, PetscLogGlobalNames global_events, PetscNestedEventNode **tree_p, PetscEventPerfInfo **perf_p)
{
  PetscMPIInt           size;
  PetscInt              num_nodes;
  PetscInt              num_map_entries;
  PetscEventPerfInfo   *perf;
  NestedIdPair         *keys;
  NestedId             *vals;
  PetscInt              offset;
  PetscInt              num_descendants;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &num_nodes));
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, node, &tree[node].name));
    tree[node].parent = -1;
  }
  PetscCall(PetscNestedHashGetSize(nested->pair_map, &num_map_entries));
  PetscCall(PetscMalloc2(num_map_entries, &keys, num_map_entries, &vals));
  offset = 0;
  PetscCall(PetscNestedHashGetPairs(nested->pair_map, &offset, keys, vals));
  for (PetscInt k = 0; k < num_map_entries; k++) {
    NestedId root_local = keys[k].root;
    NestedId leaf_local = vals[k];
    PetscInt root_global;
    PetscInt leaf_global;

    PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, leaf_local, &leaf_global));
    if (root_local >= 0) {
      PetscCall(PetscLogGlobalNamesLocalGetGlobal(global_events, root_local, &root_global));
      tree[leaf_global].parent = root_global;
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

  num_descendants = 0;
  PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes, -1, tree, &num_descendants));
  PetscAssert(num_descendants == num_nodes, comm, PETSC_ERR_PLIB, "Failed tree ordering invariant");

  PetscCall(PetscCalloc1(num_nodes, &perf));
  for (PetscInt node = 0; node < num_nodes; node++) {
    PetscInt event_id;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, node, &event_id));
    if (event_id >= 0) {
      PetscEventPerfInfo *event_info;

      PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(nested->handler, 0, event_id, &event_info));
      perf[node] = *event_info;
    } else {
      PetscCall(PetscArrayzero(&perf[node], 1));
    }
  }
  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLogHandlerView_Nested(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested)handler->ctx;
  PetscNestedEventNode  *nodes;
  PetscEventPerfInfo    *perf;
  PetscLogGlobalNames    global_events;
  PetscNestedEventTree   tree;
  PetscViewerFormat      format;
  MPI_Comm               comm = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, nested->state->registry, &global_events));
  PetscCall(PetscLogNestedCreatePerfNodes(comm, nested, global_events, &nodes, &perf));
  tree.comm          = comm;
  tree.global_events = global_events;
  tree.perf          = perf;
  tree.nodes         = nodes;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML) {
    PetscCall(PetscLogHandlerView_Nested_XML(nested, &tree, viewer));
  } else if (format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogHandlerView_Nested_Flamegraph(nested, &tree, viewer));
  } else SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "No nested viewer for this format");
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscFree(tree.nodes));
  PetscCall(PetscFree(tree.perf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Nested(MPI_Comm comm, PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler = *handler_p;
  PetscCall(PetscLogHandlerContextCreate_Nested(comm, (PetscLogHandler_Nested *)&handler->ctx));
  handler->type          = PETSC_LOG_HANDLER_NESTED;
  handler->destroy       = PetscLogHandlerDestroy_Nested;
  handler->stagePush     = PetscLogHandlerStagePush_Nested;
  handler->stagePop      = PetscLogHandlerStagePop_Nested;
  handler->eventBegin    = PetscLogHandlerEventBegin_Nested;
  handler->eventEnd      = PetscLogHandlerEventEnd_Nested;
  handler->eventSync     = PetscLogHandlerEventSync_Nested;
  handler->objectCreate  = PetscLogHandlerObjectCreate_Nested;
  handler->objectDestroy = PetscLogHandlerObjectDestroy_Nested;
  handler->view          = PetscLogHandlerView_Nested;
  PetscFunctionReturn(PETSC_SUCCESS);
}
