#include <petscdmnetwork.h>
#include <petsc.h>

/* Finite Volume Data Strcutures, temporary as they are built right for this application. 
   but they work for now */
#include "finitevolume1d.h"

/* Finite Volume Data Structures */
typedef enum {NONE,JUNCTION=1,RESERVOIR=2,VALVE=3,DEMAND=4,INFLOW=5,STAGE=6,TANK=7,OUTFLOW=8} VertexType;

/* Network Data Structures */

/* Component numbers used for accessing data in DMNetWork*/
typedef enum {FVEDGE=0} EdgeCompNum;
typedef enum {JUNCTION=0,FLUX=1,EDGEDIR=2} VertexCompNum;   

struct _p_Junction{
  PetscInt	    id;        /* global index */
  PetscInt      tag;       /* external id */
  VertexType    type;               
  Mat           *jacobian;
  PetscReal     x; /* x-coordinates */
                  
  /* Finite Volume Context */
  RiemannFunction_2WaySplit couplingflux; 
  /* Function for computing the numerical flux at each connected edge. For now I'm using 
     the same input structure as a Riemann function. In general could be many things. So will
     be experimenting on a good function for this. */

  /* boundary data structures - To be added*/

  /* Multirate Context */
  PetscInt   multirateoffset; /* offset to index from the input index (X) to the output index (F) in slow/buffer/medium
                                 rhs function evals (local indexing). How to generate 
                                 in general?! I'm currently using techniques based on knowing the underlying vector 
                                 representation of dmnetwork, which is really not the way of doing this.*/
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Junction *Junction;

typedef enum {NONE,EVALTO,EVALFROM,EVALBOTH} VertexEval; /* terrible name... */

struct _p_FVEdge
{
  /* identification variables */
  PetscInt    id;
  PetscInt    networkid; /* which network this pipe belongs */
  PetscInt    vto_offset, vfrom_offest; /* offset for accessing the data 'belonging' to this 
                                           edge contained in the 'to' and 'from' vertices */
  PetscInt    vto_recon_offset, from_recon_offset; /* offsets for placing the reconstruction data and setting flux data 
                                                      for the edge cells */
  /* solver objects */
  /* Note that this object holds no solution data. This is held 
     by the DMnetwork. This object merely gives the appropriate context 
     for the data belonging to the given edge */

  PetscInt    nnodes;   /* number of nodes in da discretization */
  Mat         *jacobian;
 /*void                *user;*/ /* user inputted data, need for function evaluations. However not 
                                   sure how do this right, as this data will have to set after partitioning, 
                                   so the user will have to provide a function to set these based on id I think 
                                   worry about it later */

  /* FV object */
  /* Note: Includes physics object for now. To be reworked */
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
  PetscErrorCode      (*sample)(void*,PetscInt,FVBCType,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
  PetscErrorCode      (*inflow)(void*,PetscReal,PetscReal,PetscReal*);
  RiemannFunction     riemann;
  ReconstructFunction characteristic;
  PetscErrorCode      (*destroy)(void*);
  void                *user;
  PetscInt            dof;
  char                *fieldname[16];
} PhysicsCtx_Net;

/* Global FV information on the entire network. */
struct _p_FVNetwork 
{
  MPI_Comm    comm;
  PetscInt    nedge,nvertex;           /* local number of components */
  PetscInt    Nedge,Nvertex;           /* global number of components */
  PetscInt    *edgelist;               /* local edge list */
  Vec         localX;                  /* vectors used in local function evalutation */
  PetscInt    nnodes_loc;              /* num of global and local nodes */
  PetscInt    stencilwidth;
  void        *user;
  DM          network; 
    
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
/* set the components into the network and the number of variables
   each component requires.*/
PetscErrorCode FVNetworkSetComponents(FVNetwork); 
/* After distributing the network, build the dynamic data required 
   by the components. This includes physics data as well as building 
   the vertex data structures needed for evaluating the edge data they 
   'steal' */ 
PetscErrorCode FVNetworkSetupPhysics(FVNetwork);
/* Create the multirate data structures the components require */
PetscErrorCode FVNetworkSetupMultirate(FVNetwork,PetscInt*,PetscInt*,PetscInt*); 

