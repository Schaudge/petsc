
#include <petsc/private/fnimpl.h> /*I "petscfn.h" I*/

/* Start with a static graph type */

typedef struct _n_PetscFnDAGNode PetscFnDAGNode;

struct _n_PetscFnDAGNode
{
  char          *name;
  PetscFn       fn;
  Vec           outvec;
  PetscInt      id;
};

typedef struct _n_PetscFnDAGEdge PetscFnDAGEdge;

struct _n_PetscFnDAGEdge
{
  IS               fromIS;
  IS               toIS;
  PetscScalar      weight;
  PetscInt         id;
  PetscInt         fromNodeId;
  PetscInt         toNodeId;
  PetscObjectState state;
};

typedef struct _n_PetscFnDAGraph *PetscFnDAGraph;

struct _n_PetscFnDAGraph
{
  MPI_Comm        comm;
  PetscSegBuffer  nodeBuffer;
  PetscSegBuffer  edgeBuffer;
  PetscFnDAGNode *nodes;
  PetscFnDAGEdge *edges;
  PetscInt       *nodeOrder;
  PetscInt       *inEdgeOffsets;
  PetscInt       *outEdgeOffsets;
  PetscInt       *inEdges;
  PetscInt       *outEdges;
  PetscInt        numNodes;
  PetscInt        numEdges;
  PetscInt        refct;
  PetscInt        setupCalled;
};

static PetscErrorCode PetscFnDAGraphDestroy(PetscFnDAGraph *graph_p)
{
  PetscFnDAGraph graph = *graph_p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph && --(graph->refct) == 0) {
    ierr = PetscSegBufferDestroy(&(graph->nodeBuffer));CHKERRQ(ierr);
    ierr = PetscSegBufferDestroy(&(graph->edgeBuffer));CHKERRQ(ierr);
    ierr = PetscFree(graph->nodes);CHKERRQ(ierr);
    ierr = PetscFree(graph->edges);CHKERRQ(ierr);
    ierr = PetscFree(graph->nodeOrder);CHKERRQ(ierr);
    ierr = PetscFree(graph->inEdges);CHKERRQ(ierr);
    ierr = PetscFree(graph->outEdges);CHKERRQ(ierr);
    ierr = PetscFree(graph->inEdgeOffsets);CHKERRQ(ierr);
    ierr = PetscFree(graph->outEdgeOffsets);CHKERRQ(ierr);
    ierr = PetscFree(graph);CHKERRQ(ierr);
  }
  *graph_p = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGraphCreate(MPI_Comm comm, PetscFnDAGraph *graph_p)
{
  PetscFnDAGraph graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&graph);CHKERRQ(ierr);
  graph->comm = comm;
  graph->refct = 1;
  ierr = PetscSegBufferCreate(sizeof(PetscFnDAGNode), 2, &(graph->nodeBuffer));CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscFnDAGEdge), 2, &(graph->edgeBuffer));CHKERRQ(ierr);
  *graph_p = graph;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGraphPushNode(PetscFnDAGraph graph, PetscFnDAGNode **node)
{
  size_t         curSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->setupCalled) SETERRQ(graph->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot push a node onto a PetscFnDAGTape after it has been setup.");
  ierr = PetscSegBufferGetSize(graph->nodeBuffer, &curSize);CHKERRQ(ierr);
  ierr = PetscSegBufferGet(graph->nodeBuffer, 1, (void *) node);CHKERRQ(ierr);
  (*node)->id = (PetscInt) curSize;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGraphPushEdge(PetscFnDAGraph graph, PetscFnDAGEdge **edge)
{
  size_t         curSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->setupCalled) SETERRQ(graph->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot push an edge onto a PetscFnDAGTape after it has been setup.");
  ierr = PetscSegBufferGetSize(graph->edgeBuffer, &curSize);CHKERRQ(ierr);
  ierr = PetscSegBufferGet(graph->edgeBuffer, 1, (void *) edge);CHKERRQ(ierr);
  (*edge)->id = (PetscInt) curSize;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGraphSetUp(PetscFnDAGraph graph)
{
  size_t         numNodes, numEdges;
  PetscInt       *inDegree, *outDegree, i;
  PetscInt       *nodeRank;
  PetscInt       numSources, numSinks;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->setupCalled) PetscFunctionReturn(0);
  graph->setupCalled = PETSC_TRUE;

  /* serialize nodes and edges */
  ierr = PetscSegBufferGetSize(graph->nodeBuffer, &numNodes);CHKERRQ(ierr);
  ierr = PetscSegBufferGetSize(graph->edgeBuffer, &numEdges);CHKERRQ(ierr);
  graph->numNodes = (PetscInt) numNodes;
  graph->numEdges = (PetscInt) numEdges;
  ierr = PetscSegBufferExtractAlloc(graph->nodeBuffer, &(graph->nodes));CHKERRQ(ierr);
  ierr = PetscSegBufferExtractAlloc(graph->edgeBuffer, &(graph->edges));CHKERRQ(ierr);

  /* count node degrees, set up inEdges and outEdges */
  ierr = PetscCalloc2(numNodes, &inDegree, numNodes, &outDegree);CHKERRQ(ierr);
  for (i = 0; i < numEdges; i++) {
    PetscFnDAGEdge *edge = &graph->edges[i];
    PetscInt        from, to;

    from = edge->fromNodeId;
    to   = edge->toNodeId;
    if (from < 0 || from >= numNodes) SETERRQ3(graph->comm, PETSC_ERR_ARG_OUTOFRANGE, "Edge %D has invalid source %D, valid range [0, %D)\n", i, from, numNodes);
    if (to   < 0 || to   >= numNodes) SETERRQ3(graph->comm, PETSC_ERR_ARG_OUTOFRANGE, "Edge %D has invalid target %D, valid range [0, %D)\n", i, to,   numNodes);
    outDegree[from]++;
    inDegree[to]++;
  }
  for (numSources = 0, numSinks = 0, i = 0; i < numEdges; i++) {
    if (!inDegree[i]) numSources++;
    if (!outDegree[i]) numSinks++;
  }
  if (!numSources) SETERRQ(graph->comm, PETSC_ERR_ARG_WRONG, "Graph has no sources, not acyclic\n");
  if (!numSinks) SETERRQ(graph->comm, PETSC_ERR_ARG_WRONG, "Graph has no sinks, not acyclic\n");
  ierr = PetscMalloc1(numNodes+1,&graph->inEdgeOffsets);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNodes+1,&graph->outEdgeOffsets);CHKERRQ(ierr);
  ierr = PetscMalloc1(numEdges,&graph->inEdges);CHKERRQ(ierr);
  ierr = PetscMalloc1(numEdges,&graph->outEdges);CHKERRQ(ierr);
  graph->inEdgeOffsets[0] = 0;
  graph->outEdgeOffsets[0] = 0;
  for (i = 0; i < numNodes; i++) {
    graph->inEdgeOffsets[i+1] = graph->inEdgeOffsets[i] + inDegree[i];
    inDegree[i] = 0;
    graph->outEdgeOffsets[i+1] = graph->outEdgeOffsets[i] + outDegree[i];
    outDegree[i] = 0;
  }
  for (i = 0; i < numEdges; i++) {
    PetscFnDAGEdge *edge = &graph->edges[i];
    PetscInt        from, to;

    from = edge->fromNodeId;
    to   = edge->toNodeId;
    graph->inEdges[graph->inEdgeOffsets[to] + inDegree[to]++] = i;
    graph->outEdges[graph->outEdgeOffsets[from] + outDegree[from]++] = i;
  }
  ierr = PetscFree2(inDegree, outDegree);CHKERRQ(ierr);

  /* rank each node by its longest path from a source */
  ierr = PetscCalloc1(numNodes, &nodeRank);CHKERRQ(ierr);
  ierr = PetscMalloc1(numNodes, &graph->nodeOrder);CHKERRQ(ierr);
  for (i = 0; i < numNodes; i++) graph->nodeOrder[i] = i;
  while (1) {
    PetscBool anyChange = PETSC_FALSE;

    for (i = 0; i < numEdges; i++) {
      PetscFnDAGEdge *edge = &graph->edges[i];
      PetscInt        from, to;
      PetscInt        fromRank, toRank, toRankOrig;

      from = edge->fromNodeId;
      to   = edge->toNodeId;

      fromRank = nodeRank[from];
      toRank = toRankOrig = nodeRank[to];
      nodeRank[to] = toRank = PetscMax(toRank, fromRank + 1);
      if (toRank != toRankOrig) anyChange = PETSC_TRUE;
      if (toRank >= numNodes) SETERRQ(graph->comm, PETSC_ERR_ARG_WRONG, "Cycle detected in graph");
    }
    if (!anyChange) break;
  }
  ierr = PetscSortIntWithArray(numNodes, nodeRank, graph->nodeOrder);CHKERRQ(ierr);
  ierr = PetscFree(nodeRank);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* A tape for keeping track of intermediate computations on a PetscFnDAGraph */

typedef struct _n_PetscFnDAGTape *PetscFnDAGTape;

struct _n_PetscFnDAGTape
{
  PetscFnDAGraph    graph;
  VecScatter       *scatters;
  PetscInt          refct;
};

static PetscErrorCode PetscFnDAGraphCreateTape(PetscFnDAGraph graph, PetscFnDAGTape *tape_p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGTapeDestroy(PetscFnDAGTape *tape_p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

typedef struct _n_PetscFn_DAG PetscFn_DAG;

struct _n_PetscFn_DAG
{
  PetscFnDAGraph  graph;
  PetscFnDAGTape  tape;
  PetscInt        inputNode;
  PetscInt        outputNode;
  PetscInt        setupCalled;
};

static PetscErrorCode PetscFnSetUp_DAG(PetscFn fn)
{
  PetscFn_DAG    *dag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dag  = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGraphSetUp(dag->graph);CHKERRQ(ierr);
  if (!dag->tape) {
    ierr = PetscFnDAGraphCreateTape(dag->graph, &dag->tape);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _n_PetscFnDAGPlan *PetscFnDAGPlan;

static PetscErrorCode PetscFnDAGTapeGetPlan(PetscFnDAGTape tape, PetscInt inputNode, PetscInt outputNode, Vec x, PetscInt der, PetscInt rangeID, const IS subsets[], const Vec subvecs[], Vec y, PetscFnDAGPlan *plan)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGTapeRestorePlan(PetscFnDAGTape tape, PetscInt inputNode, PetscInt outputNode, Vec x, PetscInt der, PetscInt rangeID, const IS subsets[], const Vec subvecs[], Vec y, PetscFnDAGPlan *plan)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscFnDAGPlanGetNumOps(PetscFnDAGPlan plan, PetscInt *numOps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGPlanGetOpArgs(PetscFnDAGPlan plan, PetscInt opNum, PetscInt *node_id, PetscInt *der, PetscInt *rangeId,const IS *subsets[],const Vec *subvecs[], Vec *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGTapeDerVec(PetscFnDAGTape tape, PetscInt node_id, PetscInt der, PetscInt rangeId, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDerivativeVec_DAG(PetscFn fn, Vec x, PetscInt der, PetscInt rangeId, const IS subsets[], const Vec subvecs[], Vec y)
{
  PetscFn_DAG    *dag;
  PetscFnDAGPlan plan;
  PetscInt       numOps, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dag  = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGTapeGetPlan(dag->tape, dag->inputNode, dag->outputNode, x, der, rangeId, subsets, subvecs, y, &plan);CHKERRQ(ierr);
  ierr = PetscFnDAGPlanGetNumOps(plan, &numOps);CHKERRQ(ierr);
  for (i = 0; i < numOps; i++) {
    PetscInt  node_id;
    PetscInt  node_der;
    PetscInt  node_rangeId;
    const IS  *node_subsets;
    const Vec *node_subvecs;
    Vec       node_y;

    ierr = PetscFnDAGPlanGetOpArgs(plan, i, &node_id, &node_der, &node_rangeId, &node_subsets, &node_subvecs, &node_y);CHKERRQ(ierr);
    ierr = PetscFnDAGTapeDerVec(dag->tape, node_id, node_der, node_rangeId, node_subsets, node_subvecs, node_y);CHKERRQ(ierr);
  }
  ierr = PetscFnDAGTapeRestorePlan(dag->tape, dag->inputNode, dag->outputNode, x, der, rangeId, subsets, subvecs, y, &plan);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_DAG(PetscFn fn)
{
  PetscFn_DAG    *dag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dag  = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGraphDestroy(&(dag->graph));CHKERRQ(ierr);
  ierr = PetscFree(fn->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnView_DAG(PetscFn fn, PetscViewer viewer)
{
  PetscBool         isAscii;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isAscii);CHKERRQ(ierr);
  if (!isAscii) PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnDAGAddNode(PetscFn fn, PetscFn nodefn, Vec outvec, const char name[], PetscInt *node_id)
{
  PetscFn_DAG    *dag;
  PetscFnDAGNode *node;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (nodefn) PetscValidHeaderSpecific(nodefn,PETSCFN_CLASSID,2);
  if (outvec) PetscValidHeaderSpecific(outvec,VEC_CLASSID,3);
  if (fn->setupcalled) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE, "Cannot add node after PetscFnSetUp() called");
  if (!nodefn && !outvec) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_NULL, "One of nodefn and outvec must be non-null");
  dag = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGraphPushNode(dag->graph,&node);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)nodefn);CHKERRQ(ierr);
  node->fn = nodefn;
  if (name) {ierr = PetscStrallocpy(name, &node->name);CHKERRQ(ierr);}
  if (!outvec) {
    ierr = PetscFnCreateVecs(nodefn, NULL, NULL, NULL, &outvec);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)outvec);CHKERRQ(ierr);
  }
  node->outvec = outvec;
  *node_id = node->id;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnDAGAddEdge(PetscFn fn, PetscInt nodeFrom, PetscInt nodeTo, IS isFrom, IS isTo, PetscScalar weight, PetscInt *edge_id)
{
  PetscFn_DAG    *dag;
  PetscFnDAGEdge *edge;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,PETSCFN_CLASSID,1);
  if (isFrom) PetscValidHeaderSpecific(isFrom, IS_CLASSID,4);
  if (isTo) PetscValidHeaderSpecific(isTo, IS_CLASSID,5);
  if (fn->setupcalled) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONGSTATE, "Cannot add edge after PetscFnSetUp() called");
  dag = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGraphPushEdge(dag->graph,&edge);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)isFrom);CHKERRQ(ierr);
  edge->fromIS = isFrom;
  ierr = PetscObjectReference((PetscObject)isTo);CHKERRQ(ierr);
  edge->toIS = isTo;
  edge->weight = weight;
  edge->fromNodeId = nodeFrom;
  edge->toNodeId = nodeTo;
  *edge_id = edge->id;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFnCreate_DAG(PetscFn fn)
{
  PetscFn_DAG    *dag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(fn, &dag);CHKERRQ(ierr);
  fn->data = (void *) dag;
  fn->ops->destroy = PetscFnDestroy_DAG;
  ierr = PetscObjectChangeTypeName((PetscObject)fn, PETSCFNDAG);CHKERRQ(ierr);
  ierr = PetscFnDAGraphCreate(PetscObjectComm((PetscObject)fn), &(dag->graph));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
