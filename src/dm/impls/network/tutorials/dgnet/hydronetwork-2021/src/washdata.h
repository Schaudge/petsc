/* Modified from petsc/src/snes/examples/tutorials/network/water/water.h */

#ifndef WASHDATA_H
#define WASHDATA_H

#include <petscdmnetwork.h>
#include <petscctable.h>

#define MAXLINE 1000
typedef enum {
  NONE,
  JUNCTION  = 1,
  RESERVOIR = 2,
  VALVE     = 3,
  DEMAND    = 4,
  INFLOW    = 5,
  STAGE     = 6,
  TANK      = 7
} VertexType;

#define EDGE_TYPE_PIPE     0
#define EDGE_TYPE_PUMP     1
#define PIPE_STATUS_OPEN   0
#define PIPE_STATUS_CLOSED 1
#define PIPE_STATUS_CV     2

#define GPM_CFS 0.0022280023234587 /* Scaling constant for GPM to CFS conversion */

typedef struct {
  PetscInt    id;         /* id */
  PetscScalar demand;     /* demand (gpm) */
  PetscInt    dempattern; /* demand pattern id */
} JunctionData;

typedef struct {
  PetscInt    id;   /* id */
  PetscScalar flow; /* flow hydrograph [m3/s] */
} Inflow;

typedef struct {
  PetscInt    id;   /* id */
  PetscScalar head; /* flow head [m] */
} Stage;

typedef struct {
  PetscInt    id;          /* id */
  PetscScalar head;        /* head (ft) */
  PetscInt    headpattern; /* head pattern */
} Reservoir;

typedef struct {
  PetscInt    id;          /* id */
  PetscScalar head;        /* head (ft) */
  PetscScalar initlvl;     /* initial level (ft) */
  PetscScalar minlvl;      /* minimum level (ft) */
  PetscScalar maxlvl;      /* maximum level (ft) */
  PetscScalar diam;        /* diameter (ft) */
  PetscScalar minvolume;   /* minimum volume (ft^3) */
  PetscInt    volumecurve; /* Volume curve id */
} Tank;

struct _p_VERTEX_Water {
  PetscInt    id; /* id */
  PetscScalar elev;
  PetscInt    type; /* vertex type (junction, reservoir) */

  Inflow       inflow; /* flow hydrograph data */
  Stage        stage;  /* flow head data  */
  JunctionData junc;   /* junction data */
  Reservoir    res;    /* reservoir data */
  Tank         tank;   /* tank data */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_VERTEX_Water *VERTEX_Water;

typedef struct {
  PetscInt    id;        /* id */
  PetscInt    node1;     /* From node */
  PetscInt    node2;     /* to node */
  PetscScalar length;    /* channel length [m] */
  PetscScalar width;     /* channel width [m] */
  PetscScalar roughness; /* roughness (dimensionless) */
  PetscScalar slope;     /* bed slope [m/m] */
  PetscScalar qInitial;  /* initial q*/
  PetscScalar hInitial;  /* initial h */
  PetscScalar minorloss; /* minor losses */
  char        stat[16];  /* Status */
  PetscInt    status;    /* Pipe status (see PIPE_STATUS_XXX definition on top) */
  PetscScalar n;         /* Exponent for h = kQ^n */
  PetscScalar k;
} PipeData;

typedef struct {
  PetscInt id;           /* id */
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
} PumpData;

struct _p_EDGE_Water {
  PetscInt id;   /* id */
  PetscInt type; /* edge type (pump, pipe) */
  PipeData pipe; /* pipe data */
  PumpData pump; /* pump data */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_EDGE_Water *EDGE_Water;

/* EPANET top-level data structure */
struct _p_WATERDATA {
  PetscInt     nvertex;
  PetscInt     nedge;
  PetscInt     njunction;
  PetscInt     ninflow;
  PetscInt     nstage;
  PetscInt     nreservoir;
  PetscInt     ntank;
  PetscInt     npipe;
  PetscInt     npump;
  PetscInt     id_max;
  PetscTable   table;
  VERTEX_Water vertex;
  EDGE_Water   edge;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_WATERDATA WATERDATA;

extern PetscErrorCode WaterReadData(WATERDATA *, const char *);
extern PetscErrorCode GetListofEdges_Water(WATERDATA *, PetscInt *);
#endif
