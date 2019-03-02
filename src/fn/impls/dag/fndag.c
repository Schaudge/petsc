
#include <petsc/private/fnimpl.h> /*I "petscfn.h" I*/

typedef struct _n_PetscFnDAGVec PetscFnDAGVec;

struct _n_PetscFnDAGVec
{
  Vec weakRef;
  Vec mine;
  PetscObjectState stateWeakRef;
  PetscObjectState stateMine;
};

typedef struct _n_PetscFnDAGNode PetscFnDAGNode;

struct _n_PetscFnDAGNode
{
  PetscFnDAGVec input;
  PetscFnDAGVec output;
  char          *name;
  PetscInt      id;
};

typedef struct _n_PetscFnDAGEdge PetscFnDAGEdge;

struct _n_PetscFnDAGEdge
{
  VecScatter  scatter;
  PetscScalar scale;
};

typedef struct _n_PetscFnDAGTape *PetscFnDAGTape;

struct _n_PetscFnDAGTape
{
  MPI_Comm        comm;
  PetscSegBuffer  nodeBuffer;
  PetscSegBuffer  edgeBuffer;
  PetscFnDAGNode *nodes;
  PetscFnDAGNode *edges;
  PetscInt       *nodeOrder;
  PetscInt        numNodes;
  PetscInt        numEdges;
  PetscInt        refct;
  PetscInt        setupCalled;
};

static PetscErrorCode PetscFnDAGTapeCreate(MPI_Comm comm, PetscFnDAGTape *tape_p)
{
  PetscFnDAGTape tape;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&tape);CHKERRQ(ierr);
  tape->comm = comm;
  tape->refct = 1;
  ierr = PetscSegBufferCreate(sizeof(PetscFnDAGNode), 2, &(tape->nodeBuffer));CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscFnDAGEdge), 2, &(tape->edgeBuffer));CHKERRQ(ierr);
  *tape_p = tape;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGTapeDestroy(PetscFnDAGTape *tape_p)
{
  PetscFnDAGTape tape = *tape_p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tape && --(tape->refct) == 0) {
    ierr = PetscSegBufferDestroy(&(tape->nodeBuffer));CHKERRQ(ierr);
    ierr = PetscSegBufferDestroy(&(tape->edgeBuffer));CHKERRQ(ierr);
    ierr = PetscFree(tape->nodes);CHKERRQ(ierr);
    ierr = PetscFree(tape->edges);CHKERRQ(ierr);
    ierr = PetscFree(tape);CHKERRQ(ierr);
  }
  *tape_p = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGSetUp(PetscFnDAGTape tape)
{
  size_t numNodes, numEdges;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tape->setupCalled) PetscFunctionReturn(0);
  tape->setupCalled = PETSC_TRUE;
  ierr = PetscSegBufferGetSize(tape->nodeBuffer, &numNodes);CHKERRQ(ierr);
  ierr = PetscSegBufferGetSize(tape->edgeBuffer, &numEdges);CHKERRQ(ierr);
  tape->numNodes = (PetscInt) numNodes;
  tape->numEdges = (PetscInt) numEdges;
  ierr = PetscSegBufferExtractAlloc(tape->nodeBuffer, &(tape->nodes));CHKERRQ(ierr);
  ierr = PetscSegBufferExtractAlloc(tape->edgeBuffer, &(tape->edges));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGDestroy(PetscFnDAGTape *tape_p)
{
  PetscFnDAGTape tape = *tape_p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tape || --tape->refct > 0) {
    *tape_p = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscFree(tape);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGPushNode(PetscFnDAGTape tape, PetscFnDAGNode **node)
{
  size_t         curSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tape->setupCalled) SETERRQ(tape->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot push a node onto a PetscFnDAGTape after it has been setup.");
  ierr = PetscSegBufferGetSize(tape->nodeBuffer, &curSize);CHKERRQ(ierr);
  ierr = PetscSegBufferGet(tape->nodeBuffer, 1, (void *) node);CHKERRQ(ierr);
  (*node)->id = (PetscInt) curSize;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGPushEdge(PetscFnDAGTape tape, PetscFnDAGEdge **edge)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tape->setupCalled) SETERRQ(tape->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot push an edge onto a PetscFnDAGTape after it has been setup.");
  ierr = PetscSegBufferGet(tape->edgeBuffer, 1, (void *) edge);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDAGTapePushVec(PetscFnDAGTape tape, Vec x, PetscInt node, PetscBool input)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

typedef struct
{
  PetscFnDAGTape  tape;
  PetscInt        inputNode;
  PetscInt        outputNode;
  PetscInt        numNodesMeet;
  PetscInt        *meet;
  PetscInt        *nodeInMeet;
  PetscInt        setupCalled;
} PetscFn_DAG;

static PetscErrorCode PetscFnApply_DAG(PetscFn fn, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* push x into its place in the tape */
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFnDestroy_DAG(PetscFn fn)
{
  PetscFn_DAG    *dag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dag  = (PetscFn_DAG *) fn->data;
  ierr = PetscFnDAGTapeDestroy(&(dag->tape));CHKERRQ(ierr);
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


PetscErrorCode PetscFnCreate_DAG(PetscFn fn)
{
  PetscFn_DAG    *dag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(fn, &dag);CHKERRQ(ierr);
  fn->data = (void *) dag;
  fn->ops->destroy = PetscFnDestroy_DAG;
  fn->ops->apply   = PetscFnApply_DAG;
  ierr = PetscObjectChangeTypeName((PetscObject)fn, PETSCFNDAG);CHKERRQ(ierr);
  ierr = PetscFnDAGTapeCreate(PetscObjectComm((PetscObject)fn), &(dag->tape));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
