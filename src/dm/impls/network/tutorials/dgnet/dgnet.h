#if !defined(__DGNET_H)
  #define __DGNET_H

  #include <petscdmnetwork.h>
  #include <petscts.h>
  #include <petscriemannsolver.h>
  #include <petscnetrs.h>
  #include <petscnetrp.h>

PETSC_EXTERN PetscLogEvent DGNET_Limiter;
PETSC_EXTERN PetscLogEvent DGNET_Edge_RHS;
PETSC_EXTERN PetscLogEvent DGNET_RHS_COMM;
PETSC_EXTERN PetscLogEvent DGNET_SetUP;
PETSC_EXTERN PetscLogEvent DGNET_RHS_Vert;

/* Component numbers used for accessing data in DMNetWork*/
typedef enum {
  FVEDGE = 0
} EdgeCompNum;
typedef enum {
  DGNETJUNCTION = 0
} VertexCompNum;

struct _p_DGNETJunction {
  PetscReal x, y;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_DGNETJunction *DGNETJunction;

struct _p_EdgeFE {
  DM dm;
  PetscInt  nnodes;
  PetscReal length; /* Used to setup the DMPLex, to be refactored out. */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_EdgeFE *EdgeFE;


typedef PetscErrorCode (*ReconstructFunction)(void *, PetscInt, const PetscScalar *, PetscScalar *, PetscScalar *, PetscReal *);

typedef struct {
  PetscErrorCode (*samplenetwork)(void *, PetscInt, PetscReal, PetscReal, PetscReal *, PetscInt);
  PetscErrorCode (*flux)(void *, const PetscReal *, PetscReal *);
  ReconstructFunction  characteristic;
  PetscErrorCode (*destroy)(void *);
  void                  *user;
  PetscInt               dof;
  PetscInt              *order;
  PetscInt               maxorder;
  char                  *fieldname[16];
  RiemannSolver          rs;
  PetscPointFlux         flux2;
  PetscPointFluxEig      fluxeig;
  RiemannSolverRoeAvg    roeavg;
  RiemannSolverRoeMatrix roemat;
  RiemannSolverEigBasis  eigbasis;
  PetscPointFluxDer      fluxder;
  LaxCurve               laxcurve;
} PhysicsCtx_Net;

/* Global DG information on the entire network. Needs a creation function .... */
struct _p_DGNetwork {
  MPI_Comm  comm;
  Vec       localX, localF;    /* vectors used in local function evalutation */
  Vec       X;                 /* Global vectors used in function evaluations */
  Vec       RiemannData, Flux; /*used with NetRS*/
  DM        network;
  PetscInt  moni;
  PetscBool view, linearcoupling, lincouplediff, tabulated, laxcurve, adaptivecouple;
  PetscBool viewglvis, viewfullnet;
  PetscReal ymin, ymax, length, M, dx;
  char      prefix[256];
  NetRS netrs;

  /* DG Basis Evaluations and Quadrature */
  /* These are arrays with LegEval[fieldtotab[f]] giving the legendre evaluations for the given field
     We only construct a single set of tabulations/quadrature for single dg order. Different fields can have
     different dg order polynomials, but if field 1 and field 2 share the same order, then they will share the same
     tabulation.
    */

  PetscQuadrature quad;
  PetscReal     **LegEval;
  PetscReal     **LegEvalD;
  PetscReal     **LegEvaL_bdry;
  PetscReal     **Leg_L2;

  /* Viewer Object (probably refactor as a viewer for dgnet)*/
  PetscInt   *numviewpts;
  PetscReal **LegEval_equispaced; /* tabulation for viewing */

  /* DG WorkSpace Stuff */
  PetscReal *pteval;
  PetscReal *fluxeval; /*refactor this to be a single contiguous array of data */
  /*
    TODO : the above evaluations should be replaced by PETSCTabulation objects. However the petsctabulation object needs
    some additional features to make it complete, including viewers and proper general interfaces. Wrapper functions for Legendre evaluations
    and etc. That way you don't have to think about things. Also maybe should be stored internally as Mat objects?
  */

  PetscInt *fieldtotab; /* size is dof */
  PetscInt *taborder;
  PetscInt  tabordersize;

  /* Work arrays for the limiter/characterstic basis */
  PetscReal *charcoeff;
  PetscBool *limitactive;
  PetscReal *cbdryeval_L, *cbdryeval_R, *cuAvg, *uavgs;
  PetscReal  jumptol;
  PetscReal *cjmpLR;

  /* Local work arrays for numerical flux */
  PetscScalar *R, *Rinv;   /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *uLR, *cuLR; /* Solution at left and right of a cell, conservative variables, len=2*dof */
  PetscScalar *flux;       /* Flux across interface */
  PetscReal   *speeds;     /* Speeds of each wave */
  PetscReal   *uPlus;      /* Solution at the left of the interface in conservative variables, len = dof  uPlus_|_uL___cell_i___uR_|_ */
  PetscReal    cfl;
  PetscInt     initial, networktype, ndaughters;
  PetscBool    simulation;
  PetscBool    exact;
  PetscInt     hratio;
  PetscInt     Mx; /* Variable used to specify smallest number of cells for an edge in a problem */

  /* Junction */
  DGNETJunction junction;

  /* Edges */
  EdgeFE    edgefe;
  PetscReal edgethickness;

  /* We assume for efficiency and simplicity that the network has
     a single discretization on all edges and the same physics.
     So that context information is stored here in the network object. The
     solvers and rhs functions in the edges will call this info when
     actually performing the cell updates */
  PhysicsCtx_Net physics;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_DGNetwork *DGNetwork;

typedef struct _p_DGNetworkMonitorList *DGNetworkMonitorList;
struct _p_DGNetworkMonitorList {
  PetscViewer          viewer;
  Vec                  v;
  PetscInt             element, field, vsize;
  DGNetworkMonitorList next;
};

typedef struct _p_DGNetworkMonitor *DGNetworkMonitor;
struct _p_DGNetworkMonitor {
  MPI_Comm             comm;
  DGNetwork            dgnet;
  DGNetworkMonitorList firstnode;
};

typedef struct _p_DGNetworkMonitorList_Glvis *DGNetworkMonitorList_Glvis;
struct _p_DGNetworkMonitorList_Glvis {
  PetscViewer                viewer;
  DGNetwork                  dgnet;
  Vec                        v, *v_work;
  DM                         viewdm;
  DM                        *dmlist;
  PetscSection               stratumoffset;
  PetscInt                   element, nfields, *dim, numdm;
  PetscInt                   snapid;
  char                     **fec_type;
  DGNetworkMonitorList_Glvis next;
};

typedef struct _p_DGNetworkMonitor_Glvis *DGNetworkMonitor_Glvis;
struct _p_DGNetworkMonitor_Glvis {
  MPI_Comm                   comm;
  DGNetwork                  dgnet;
  DGNetworkMonitorList_Glvis firstnode;
};

/*
  Interface for multiple DGNetworks simulations on the same network (and mesh) topology.
  just holds multiple DGNetwork objects, and uses vecnest to build a global vector to integrate

  Needs to be improved (and dgnetwork needs to be made a proper object instead of the nonsense
  it currently is). But works for now.

  This is used to various testing routines, to compare coupling condtions/riemann solvers/polynomial
  orders specifically. We assume the following are shared on all dgnetworks (but don't enforce for now)

  . network topology
  . mesh topology
  . physics

  Everything else is free to be altered as you see fit.

  NOTE: Look up DMComposite Maybe the right thing for this situation ...
*/
struct _p_DGNetwork_Nest {
  PetscInt                numsimulations, numwrkvec, nummonitors;
  DGNetwork              *dgnets;
  DGNetworkMonitor       *monitors;
  Vec                    *wrk_vec; /* using for calculation in post-step functions as needed*/
  DGNetworkMonitor_Glvis *monitors_glvis;
};
typedef struct _p_DGNetwork_Nest *DGNetwork_Nest;

/* Set up the DGNetworkComponents and 'blank' network data to be read by the other functions.
   Allocate the work array data for DGNetwork */
extern PetscErrorCode DGNetworkCreate(DGNetwork, PetscInt, PetscInt);
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
extern PetscErrorCode DGNetworkCreateVectors(DGNetwork);
/* Assign the coupling condition functions to the vertices of the network based
   user provided vfluxassign function */
extern PetscErrorCode DGNetworkAssignCoupling(DGNetwork);
/* Add dynamic data to the distributed network. */
extern PetscErrorCode DGNetworkBuildDynamic(DGNetwork);

extern PetscErrorCode ViewDiscretizationObjects(DGNetwork, PetscViewer);

extern PetscErrorCode DGNetworkViewEdgeDMs(DGNetwork, PetscViewer);
extern PetscErrorCode DGNetworkViewEdgeGeometricInfo(DGNetwork, PetscViewer);

extern PetscErrorCode DGNetworkBuildTabulation(DGNetwork);
extern PetscErrorCode DGNetworkBuildEdgeDM(DGNetwork);

/* Destroy allocated data */
extern PetscErrorCode DGNetworkDestroy(DGNetwork);
extern PetscErrorCode DGNetworkDestroyTabulation(DGNetwork);
extern PetscErrorCode DGNetworkDestroyPhysics(DGNetwork);

extern PetscErrorCode DGNetworkProject(DGNetwork, Vec, PetscReal);
extern PetscErrorCode DGNetworkProject_Coarse_To_Fine(DGNetwork, DGNetwork, Vec, Vec);

extern PetscErrorCode PhysicsDestroy_SimpleFree_Net(void *);

extern PetscErrorCode DGNetworkMonitorCreate(DGNetwork, DGNetworkMonitor *);
extern PetscErrorCode DGNetworkMonitorPop(DGNetworkMonitor);
extern PetscErrorCode DGNetworkMonitorDestroy(DGNetworkMonitor *);
extern PetscErrorCode DGNetworkMonitorAdd(DGNetworkMonitor, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, PetscBool);
extern PetscErrorCode DGNetworkMonitorView(DGNetworkMonitor, Vec);
extern PetscErrorCode DGNetworkAddMonitortoEdges(DGNetwork, DGNetworkMonitor);

extern PetscErrorCode DGNetworkMonitorCreate_Glvis(DGNetwork, DGNetworkMonitor_Glvis *);
extern PetscErrorCode DGNetworkMonitorPop_Glvis(DGNetworkMonitor_Glvis);
extern PetscErrorCode DGNetworkMonitorDestroy_Glvis(DGNetworkMonitor_Glvis *);
extern PetscErrorCode DGNetworkMonitorAdd_Glvis(DGNetworkMonitor_Glvis, PetscInt, const char[], PetscViewerGLVisType);
extern PetscErrorCode DGNetworkMonitorView_Glvis(DGNetworkMonitor_Glvis, Vec);
extern PetscErrorCode DGNetworkAddMonitortoEdges_Glvis(DGNetwork, DGNetworkMonitor_Glvis, PetscViewerGLVisType);

extern PetscErrorCode DGNetworkMonitorAdd_Glvis_3D(DGNetworkMonitor_Glvis, PetscInt, const char[], PetscViewerGLVisType);
extern PetscErrorCode DGNetworkAddMonitortoEdges_Glvis_3D(DGNetwork, DGNetworkMonitor_Glvis, PetscViewerGLVisType);

extern PetscErrorCode DGNetworkNormL2(DGNetwork, Vec, PetscReal *);

extern PetscErrorCode DMPlexAdd_Disconnected(DM *, PetscInt, DM *, PetscSection *);
extern PetscErrorCode DGNetworkCreateNetworkDMPlex(DGNetwork, const PetscInt[], PetscInt, DM *, PetscSection *);
extern PetscErrorCode DGNetworkCreateNetworkDMPlex_3D(DGNetwork, const PetscInt[], PetscInt, DM *, PetscSection *, DM **, PetscInt *);
extern PetscErrorCode DGNetworkCreateNetworkDMPlex_2D(DGNetwork, const PetscInt[], PetscInt, DM *, PetscSection *, DM **, PetscInt *);

extern PetscErrorCode DGNetworkMonitorAdd_Glvis_2D_NET(DGNetworkMonitor_Glvis, const char[], PetscViewerGLVisType);
extern PetscErrorCode DGNetworkMonitorAdd_Glvis_3D_NET(DGNetworkMonitor_Glvis, const char[], PetscViewerGLVisType);
extern PetscErrorCode DGNetworkMonitorView_Glvis_NET(DGNetworkMonitor_Glvis, Vec);

extern PetscErrorCode TVDLimit_1D(DGNetwork, const PetscScalar *, const PetscScalar *, const PetscScalar *, PetscScalar *, PetscScalar *, PetscReal *, PetscSection, PetscInt);
extern PetscErrorCode Limit_1D_onesided(DGNetwork, const PetscScalar *, const PetscScalar *, PetscReal *, PetscSection, PetscInt, PetscReal);

extern PetscErrorCode DGNetlimiter(TS, PetscReal, PetscInt, Vec *);

/* Nest stuff. For use with concurrent simulations. WIP */
extern PetscErrorCode DGNetlimiter_Nested(TS, PetscReal, PetscInt, Vec *);

extern PetscErrorCode DGNetRHS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode DGNetRHS_V2(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode DGNetRHS_V3(TS, PetscReal, Vec, Vec, void *);

extern PetscErrorCode DGNetworkAssignNetRS(DGNetwork);
extern PetscErrorCode DGNetworkAssignNetRS_Traffic(DGNetwork);
extern PetscErrorCode DGNetRHS_NETRS_Nested(TS, PetscReal, Vec, Vec, void *);
#endif
