#include <petscdmnetwork.h>
#include <petsc.h>

/* Finite Volume Data Strcutures, temporary as they are built right for this application. 
   but they work for now */
#include "finitevolume1d.h"
#include "limiters.h"

/* Finite Volume Data Structures */
typedef enum {JUNCT=1,RESERVOIR=2,VALVE=3,DEMAND=4,INFLOW=5,STAGE=6,TANK=7,OUTFLOW=8} VertexType;

/* Network Data Structures */

/* Component numbers used for accessing data in DMNetWork*/
typedef enum {FVEDGE=0} EdgeCompNum;
typedef enum {JUNCTION=0,FLUX=1} VertexCompNum;   
typedef enum {EDGEIN=0,EDGEOUT=1} EdgeDirection; 

struct _p_Junction{
  PetscInt	    id;        /* global index */
  PetscInt      tag;       /* external id */
  VertexType    type;               
  Mat           *jacobian;
  PetscReal     x; /* x-coordinates */
  PetscBool     *dir; /*In the local ordering whether index i point into or out of the vertex. PetscTrue points out. */
  PetscInt      numedges; /* Number of edges connected to this vertex (globally) (it feels like this info should 
                             live in the dmnetwork, but I don't see how to access it.)*/
                  
  /* Finite Volume Context */
  /*RiemannFunction_2WaySplit couplingflux; Need to figure out how to build a function pointer within a network component in a sensible way. */
     


  /* boundary data structures - To be added*/

  /* Multirate Context */
  PetscInt   multirateoffset[3]; /* offset to index from the input index (X) to the output index (F) in slow/buffer/fast
                                 rhs function evals (local indexing). How to generate 
                                 in general?! I'm currently using techniques based on knowing the underlying vector 
                                 representation of dmnetwork, which is really not the way of doing this.*/
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Junction *Junction;

typedef enum {EVALNONE,EVALTO,EVALFROM,EVALBOTH} VertexEval; /* terrible name... */

struct _p_FVEdge
{
  /* identification variables */
  PetscInt    id;
  PetscInt    networkid; /* which network this pipe belongs */
  PetscInt    vto_offset,vfrom_offset; /* offset for accessing the data 'belonging' to this 
                                           edge contained in the 'to' and 'from' vertices */
  PetscInt    vto_recon_offset,vfrom_recon_offset; /* offsets for placing the reconstruction data and setting flux data 
                                                      for the edge cells */
  /* solver objects */
  /* Note that this object holds no solution data. This is held 
     by the DMnetwork. This object merely gives the appropriate context 
     for the data belonging to the given edge */

  PetscInt    nnodes;   /* number of nodes in da discretization */
  Mat         *jacobian;
 /*void                *user;*/ /* user inputted data, need for function evaluations. However not 
                                   sure how do this right, as this data will have to be set after partitioning, 
                                   so the user will have to provide a function to set these based on id I think.
                                   worry about it later */

  /* FV object */
  PetscReal h; /* discretization size, assumes uniform mesh*/

  /* Multirate ODE Context */ 
  PetscInt   multirateoffset; /* offset to index from the input index (X) to the output index (F) in slow/buffer/medium
                                 rhs function evals (local indexing). How to generate 
                                 in general?! I'm currently using techniques based on knowing the underlying vector 
                                 representation of dmnetwork, which is really not the way of doing this.*/
  VertexEval  vertexeval; /* What vertex data this edge should evaluate */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_FVEdge *FVEdge;

typedef struct {
  PetscErrorCode                 (*sample)(void*,PetscInt,FVBCType,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode                 (*inflow)(void*,PetscReal,PetscReal,PetscReal*);
  RiemannFunction_2WaySplit      riemann;
  ReconstructFunction_2WaySplit  characteristic;
  PetscErrorCode                 (*destroy)(void*);
  void                           *user;
  PetscInt                       dof;
  char                           *fieldname[16];
} PhysicsCtx_Net;

/* Global FV information on the entire network. */
struct _p_FVNetwork 
{
  MPI_Comm    comm;
  PetscInt    nedge,nvertex;           /* local number of components */
  PetscInt    Nedge,Nvertex;           /* global number of components */
  PetscInt    *edgelist;               /* local edge list */
  Vec         localX,localF;           /* vectors used in local function evalutation */
  Vec         X;                       /* Global vector used in function evaluations */
  PetscInt    nnodes_loc;              /* num of global and local nodes */
  PetscInt    stencilwidth;
  DM          network;
  char        prefix[256];
  void        (*limit)(const PetscScalar*,const PetscScalar*,PetscScalar*,PetscInt);
  RiemannFunction_2WaySplit couplingflux; /* Structure for performing the coupling flux. Should be attached 
                                             to a junction instead of the global network structure. Also not sure 
                                             if this is the right function type for this. But we will see. */

  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;          /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof uL____cell_i____uR*/
  PetscScalar *cslope;          /* Limited slope, written in characteristic basis */
  PetscScalar *uLR;             /* Solution at left and right of a cell, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */
  PetscReal   *speeds;          /* Speeds of each wave */
  PetscReal   *uPlus;           /* Solution at the left of the interfacce in conservative variables, len = dof  uPlus_|_uL___cell_i___uR_|_ */

  PetscReal   cfl_idt;          /* Max allowable value of 1/Delta t */
  PetscReal   cfl;
  PetscInt    initial;
  PetscBool   simulation;
  PetscBool   exact;
  PetscInt    hratio;
  PetscInt    Mx;               /* Variable used to specify smallest number of cells for an edge in a problem */

    
  /* Junction */
  Junction    junction;

  /* Edges */
  FVEdge      fvedge;

  /* FV Context */ 
  /* We assume for efficiency and simplicity that the network has
     a single discretization on all edges/vertices and the same physics. 
     So that context information is stored here in the network object. The 
     solvers and rhs functions in the edges/vertices will call this info when 
     actually performing the cell updates */ 

  PhysicsCtx_Net physics; 
  /* Multirate Context */
  PetscInt    *i_slow, *i_fast,*i_buffer;        /* On processor edges for the slow and fast 
                                                    residual functions respectively, as well 
                                                    as index for the 'buffer' vertices. */
  PetscInt    i_slow_size, i_fast_size,i_buffer_size;    
  PetscInt    bufferwidth; 
}PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar)); 
typedef struct _p_FVNetwork *FVNetwork; 

PetscErrorCode FVNetCharacteristicLimit(FVNetwork,PetscScalar*,PetscScalar*,PetscScalar*);
/* Set up the FVNetworkComponents and 'blank' network data to be read by the other functions. 
   Allocate the work array data for FVNetwork */
PetscErrorCode FVNetworkCreate(PetscInt,FVNetwork,PetscInt);
/* set the components into the network and the number of variables
   each component requires. Also construct the local ordering for the
   edges of a vertex */ 
PetscErrorCode FVNetworkSetComponents(FVNetwork);
/* Delete the unneeded data built by FVNetworkCreate. Removes 
   the edgelist data, fvedges, junctions, that have been set 
    into the network by FVNetworkSetComponents */
PetscErrorCode FVNetworkCleanUp(FVNetwork);
/* After distributing the network, build the dynamic data required 
   by the components. This includes physics data as well as building 
   the vertex data structures needed for evaluating the edge data they 
   'steal' */ 
PetscErrorCode FVNetworkSetupPhysics(FVNetwork);
/* Create the multirate data structures the components require */
PetscErrorCode FVNetworkSetupMultirate(FVNetwork,PetscInt*,PetscInt*,PetscInt*); 
/* Destroy allocated data */
PetscErrorCode FVNetworkDestroy(FVNetwork);

/*RHS Function*/
PetscErrorCode FVNetRHS(TS,PetscReal,Vec,Vec,void*);

