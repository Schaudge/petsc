#include <petscdmnetwork.h>
#include <petscts.h>
#include "limiters.h"

/* Function Specification for coupling flux calculations at the vertex */
typedef PetscErrorCode (*VertexFlux)(const void*,const PetscScalar*,const PetscBool*,PetscScalar*,PetscScalar*,const void*);

/* Network Data Structures */

/* Component numbers used for accessing data in DMNetWork*/
typedef enum {FVEDGE=0} EdgeCompNum;
typedef enum {JUNCTION=0,FLUX=1} VertexCompNum;
typedef enum {EDGEIN=0,EDGEOUT=1} EdgeDirection;

struct _p_Junction{
  Mat           mat;
  Vec           X,R;  /* Information for nonlinear solver for coupling flux */
  PetscBool     *dir;     /* In the local ordering whether index i point into or out of the vertex. PetscTrue points out. */
  PetscReal     x; 
  PetscInt      numedges; /* Number of edges connected to this vertex (globally) */
  /* Coupling Context */
  VertexFlux    couplingflux; /* Vertex flux function for coupling junctions (two or more incident edges)*/
  PetscScalar   *flux;        /* Local work array for vertex fluxes. len = dof*numedges */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Junction *Junction;

struct _p_MultirateCtx
{
  PetscInt  tobufferlvl,frombufferlvl; /* Level of the buffer on the to and from ends of the edge. lvl 0 refers to no buffer at all */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_MultirateCtx *MultirateCtx; 

struct _p_EdgeFE
{
  /* identification variables */
  PetscInt    offset_vto,offset_vfrom; /* offsets for placing the reconstruction data and setting flux data
                                          for the edge cells */
  /* solver objects */
  PetscInt    nnodes;   /* number of cells in the discretization of the edge */
  PetscReal   cfl_idt; /* Max allowable value of fvnet->cfl/Delta t on this edge*/
  /* Mesh object */
  PetscReal h; /* discretization size, assumes uniform mesh */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_EdgeFE *EdgeFE;

/* Specification for vertex flux assignment functions */
typedef PetscErrorCode (*VertexFluxAssignment)(const void*,Junction);
typedef PetscErrorCode (*VertexFluxDestroy)(const void*,Junction);

typedef PetscErrorCode (*RiemannFunction)(void*,PetscInt,const PetscScalar*,const PetscScalar*,PetscScalar*,PetscReal*);
typedef PetscErrorCode (*ReconstructFunction)(void*,PetscInt,const PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*);

typedef struct {
  PetscErrorCode                 (*sample1d)(void*,PetscInt,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode                 (*samplenetwork)(void*,PetscInt,PetscReal,PetscReal,PetscReal*,PetscInt);
  PetscErrorCode                 (*inflow)(void*,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode                 (*flux)(void*,PetscReal*,PetscReal*);
  RiemannFunction                riemann;
  ReconstructFunction            characteristic;
  VertexFluxAssignment           vfluxassign;
  VertexFluxDestroy              vfluxdestroy;
  PetscErrorCode                 (*destroy)(void*);
  void                           *user;
  PetscInt                       dof;
  char                           *fieldname[16];
} PhysicsCtx_Net;

/* Global DG information on the entire network. */
struct _p_DGNetwork
{
  MPI_Comm    comm;
  PetscInt    nedge,nvertex;           /* local number of components */
  PetscInt    Nedge,Nvertex;           /* global number of components */
  PetscInt    *edgelist;               /* local edge list */
  Vec         localX,localF;           /* vectors used in local function evalutation */
  Vec         X,Ftmp;                  /* Global vectors used in function evaluations */
  PetscInt    nnodes_loc;              /* num of local nodes */
  PetscInt    basisorder;              /* Order of the DG Basis */
  PetscInt    quadorder; 
  DM          network;
  SNES        snes;                    /* Temporary hack to hold a nonlinear solver. Used for the nonlinear riemann invariant solver.
                                          should be moved back to junct structure to reuse jacobian matrix? */
  KSP         ksp; 
  PetscInt    moni;
  PetscBool   view,linearcoupling,lincouplediff; 
  PetscReal   ymin,ymax,length;
  DMNetworkMonitor  monitor;
  char        prefix[256];
  void        (*limit)(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
  PetscErrorCode (*gettimestep)(TS ts, PetscReal *dt);

  /* DG Basis Evaluations and Quadrature */
  PetscQuadrature quad; 
  PetscReal *LegEval;
  PetscReal *LegEvalD;
  PetscReal *LegEvaL_bdry;
  PetscReal *Leg_L2; 

  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *uLR;             /* Solution at left and right of a cell, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */
  PetscReal   *speeds;          /* Speeds of each wave */
  PetscReal   *uPlus;           /* Solution at the left of the interfacce in conservative variables, len = dof  uPlus_|_uL___cell_i___uR_|_ */
  PetscReal   cfl;
  PetscInt    initial,networktype,ndaughters;
  PetscBool   simulation;
  PetscBool   exact;
  PetscInt    hratio;
  PetscInt    Mx;               /* Variable used to specify smallest number of cells for an edge in a problem */
  /* Junction */
  Junction    junction;
  /* Edges */
  EdgeFE      edgefe;
  /* FV Context */
  /* We assume for efficiency and simplicity that the network has
     a single discretization on all edges and the same physics.
     So that context information is stored here in the network object. The
     solvers and rhs functions in the edges will call this info when
     actually performing the cell updates */
  PhysicsCtx_Net physics;
  /* Multirate Context */
  /* All of these IS are on MPI_COMM_SELF*/
  IS          slow_edges,fast_edges,buf_slow_vert,slow_vert,fast_vert;
  PetscInt    bufferwidth;
}PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_DGNetwork *DGNetwork;

/* FV Functions */
extern PetscErrorCode PhysicsDestroy_SimpleFree_Net(void*);
extern PetscErrorCode RiemannListAdd_Net(PetscFunctionList*,const char*,RiemannFunction);
extern PetscErrorCode RiemannListFind_Net(PetscFunctionList,const char*,RiemannFunction*);
extern PetscErrorCode ReconstructListAdd_Net(PetscFunctionList*,const char*,ReconstructFunction);
extern PetscErrorCode ReconstructListFind_Net(PetscFunctionList,const char*,ReconstructFunction*);

/* Set up the FVNetworkComponents and 'blank' network data to be read by the other functions.
   Allocate the work array data for FVNetwork */
extern PetscErrorCode DGNetworkCreate(DGNetwork,PetscInt,PetscInt);
/* set the components into the network and the number of variables
   each component requires. Also construct the local ordering for the
   edges of a vertex */
extern PetscErrorCode DGNetworkSetComponents(DGNetwork);
/* Delete the unneeded data built by FVNetworkCreate. Removes
   the edgelist data, fvedges, junctions, that have been set
    into the network by FVNetworkSetComponents */
extern PetscErrorCode DGNetworkCleanUp(DGNetwork);
/* After distributing the network, build the dynamic data required
   by the components. This includes physics data as well as building
   the vertex data structures needed for evaluating the edge data they
   'steal' */
extern PetscErrorCode FVNetworkCreateVectors(DGNetwork);
/* Assign the coupling condition functions to the vertices of the network based 
   user provided vfluxassign function */
extern PetscErrorCode FVNetworkAssignCoupling(DGNetwork);
/* Add dynamic data to the distributed network. */
extern PetscErrorCode FVNetworkBuildDynamic(DGNetwork);

/* Destroy allocated data */
extern PetscErrorCode FVNetworkDestroy(DGNetwork);
/* Set Initial Solution */
extern PetscErrorCode DGNetworkProject(DGNetwork,Vec,PetscReal);
/* RHS Function */
extern PetscErrorCode DGNetRHS(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode DGNetRHS_SingleCoupleEval(TS,PetscReal,Vec,Vec,void*);
/* Time step length functions */
extern PetscErrorCode DGNetworkPreStep(TS);
extern PetscErrorCode DGNetwork_GetTimeStep_Fixed(TS,PetscReal*);
extern PetscErrorCode DGNetwork_GetTimeStep_Adaptive(TS,PetscReal*);