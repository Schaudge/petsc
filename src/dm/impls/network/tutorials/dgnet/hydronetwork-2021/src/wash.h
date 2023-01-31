#ifndef WASH_H
#define WASH_H

#include <petscdmnetwork.h>
#include "river.h"
#include "washdata.h"

/* junction              */
/*-----------------------*/
typedef enum {
  Q = 1,
  H = 2
} BoundaryType; /* specify either Q or H is provided by user at boundary */

struct _p_Junction {
  PetscInt  id;        /* global index */
  PetscInt  networkid; /* which network this pipe belongs */
  PetscInt  tag;       /* external id */
  Mat      *jacobian;
  PetscReal latitude, longitude; /* GPS data */
  PetscInt  nin, nout;           /* number of into and from supporting edges */

  /* physics variables */
  VertexType type;
  PetscInt   id_phy; /* physical id */
  PetscReal  elev;   /* elevation */

  /* boundary data structures */
  Reservoir    reservoir;
  Inflow       inflow;
  Stage        stage;
  Tank         tank;
  BoundaryType btype; /* specify either Q or H is provided by user at boundary */
  RiverField   bval;  /* holds boundary values provided by user */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Junction *Junction;

extern PetscErrorCode JunctionCreateJacobian(DM, PetscInt, Mat *, Mat *[]);
extern PetscErrorCode JunctionDestroyJacobian(DM, PetscInt, Junction);

struct _p_Pump {
  PetscInt id;        /* id */
  PetscInt networkid; /* which network this pipe belongs */
  PetscInt tag;
  PetscInt fr_tag;
  PetscInt to_tag;

  PetscInt node1;        /* From node */
  PetscInt node2;        /* to node */
  char     param[16];    /* curve parameter (HEAD or ENERGY or EFFICIENCY) */
  PetscInt paramid;      /* Id of the curve parameter in the CURVE data */
  struct {               /* one point curve */
    PetscScalar flow[3]; /* flow (gpm) */
    PetscScalar head[3]; /* head (ft) */
    PetscInt    npt;     /* Number of given points */
  } headcurve;
  /* Parameters for pump headloss equation hL = h0 - rQ^n */
  PetscScalar h0;
  PetscScalar r;
  PetscScalar n;

  PetscInt id_phy; /* physical id */
  PetscInt fr_phy; /* physical from node */
  PetscInt to_phy; /* phyiscal to node   */

} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Pump *Pump;

/* subnet */
struct _p_WashSubnet {
  PetscInt nedge, nvertex, *edgelist;

  /* Junction */
  Junction junction;
  PetscInt njunction;

  /* River */
  River    river;
  PetscInt nriver;

  /* Pump */
  Pump     pump;
  PetscInt npump;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_WashSubnet *WashSubnet;

/* wash                   */
/*------------------------*/
#if 0
typedef struct _p_Wash *Wash;

typedef struct _WashOps *WashOps;
struct _WashOps {
  /* Generic Operations */
  PetscErrorCode (*destroy)(Wash);
};
#endif
struct _p_Wash {
  //PETSCHEADER(struct _WashOps);

  MPI_Comm  comm;
  PetscInt  nedge, nvertex; /* global number of components */
  PetscInt  nnodes_loc;     /* num of global and local nodes */
  PetscInt  caseid;         /* built-in test case number */
  PetscInt  subcaseid;      /* built-in test sbucase number */
  PetscBool test_mscale;    /* flag for testing multi-scale TS */

  DM         dm;                           /* dmnetowrk associate with this wash */
  Vec        X, Xold, Xtmp;                /* global vector */
  Vec        localX, localXtmp, localF, F; /* vectors used in local function evalutation */
  SNES       snes_junc;                    /* solver context for algebraic junction and edge boundary equations */
  Vec        Xjunc, Fjunc;                 /* vectors for associated to snes_junc */
  VecScatter vscat_junc;
  PetscReal  dt;
  PetscBool  userJac;

  PetscInt keyJunction;
  PetscInt keyRiver;
  PetscInt keyPump;

  PetscScalar QMax, QMin, HMax, HMin; /* max and min Q and H used for plotting*/

  PetscInt    nsubnet, ncsubnet;
  WashSubnet *subnet;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Wash *Wash;

extern PetscErrorCode WashCreate(MPI_Comm, PetscInt, PetscInt, Wash *);
extern PetscErrorCode WashAddSubnet(PetscInt, PetscInt, const char[], PetscMPIInt, Wash);
extern PetscErrorCode WashCleanUp(Wash, PetscInt **);
extern PetscErrorCode WashTSSetUp(Wash, TS);
extern PetscErrorCode TSWashPreStep(TS);
extern PetscErrorCode TSWashPostStep(TS);
extern PetscErrorCode TSWashPostStage(TS, PetscReal, PetscInt, Vec *);
extern PetscErrorCode TSWashGetTimeStep(TS, PetscReal *);
extern PetscErrorCode WashIFunction(TS, PetscReal, Vec, Vec, Vec, void *);
extern PetscErrorCode WashSetInitialSolution(DM, Wash);
extern PetscErrorCode WashRHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode TSDMNetworkMonitor(TS, PetscInt, PetscReal, Vec, void *);
extern PetscErrorCode WashVecView(Wash, Vec);
extern PetscErrorCode WashCreateVecs(Wash);
extern PetscErrorCode WashDestroyVecs(Wash);
extern PetscErrorCode WashJuncSNESFuncFieldsplit(SNES, Vec, Vec, void *);
extern PetscErrorCode WashJuncSNESFunc(SNES, Vec, Vec, void *);
extern PetscErrorCode WashPostSNESSetUpFieldsplit_River(SNES);
extern PetscErrorCode WashPostSNESSetUp_River(SNES);
extern PetscErrorCode WashGetJuncLocalSize(Wash, PetscInt *);
extern PetscErrorCode RiverView(Vec, DM, Wash);
extern PetscErrorCode WashReadInputFile(PetscInt, char[][PETSC_MAX_PATH_LEN]);
extern PetscErrorCode WashDestroy(Wash);
extern PetscErrorCode WashSetUpCoupleVertices(Wash);
extern PetscErrorCode WashJunctionView(Wash);

#endif
