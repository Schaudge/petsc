#ifndef RIVER_H
#define RIVER_H
#define GRAV 9.806

#include <petscts.h>
#include <petscdmda.h>

typedef struct {
  PetscScalar q; /* depth */
  PetscScalar h; /* flow rate */
} RiverField;

typedef struct {
  PetscScalar Q0, H0; /* boundary values in upstream */
  PetscScalar QL, HL; /* boundary values in downstream */
} RiverBoundary;

/* river                 */
/*----------------------*/
struct _p_River {
  /* identification variables */
  PetscInt id;
  PetscInt networkid; /* which network this pipe belongs */

  /* solver objects */
  MPI_Comm    comm;
  Vec         x_old;
  RiverField *xold;
  DM          da;
  PetscReal   dt;
  PetscInt    ncells; /* number of cells in da discretization */
  Mat        *jacobian;
  PetscScalar flux[4];  /* flux[0:1]: flux at cell[0]; flux[2:3]: flux at cell[ncells-1]; See RiverIFunctionLocal() */
  PetscScalar ubexv[4]; /* upstream ubexvLh[0], ubexvRh[1], ubexvLq[2],ubexvRq[3]; See RiverIFunctionLocal() */
  PetscScalar dbexv[4]; /* downstream dbexvLh[0], dbexvRh[1], dbexvLq[2],dbexvRq[3]; See RiverIFunctionLocal() */

  /* physics */
  PetscInt   id_phy; /*physical id*/
  PetscInt   fr_phy; /*physical from node*/
  PetscInt   to_phy; /*phyiscal to node*/
  PetscReal  length;
  PetscReal  width;
  PetscReal  roughness;
  PetscReal  slope;
  PetscReal  q0;       /* initial q */
  PetscReal  h0;       /* initial h */
  PetscReal *z, *zMax; /* bed elevation */
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_River *River;

extern PetscErrorCode geoFun(PetscScalar, PetscScalar, PetscScalar, const PetscScalar, PetscScalar *, PetscScalar *);
extern PetscErrorCode wetbed(PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, const PetscScalar, const PetscScalar, const PetscScalar, PetscScalar *, PetscScalar *);
extern PetscErrorCode exactRiemannSolution(RiverField *, RiverField *, RiverField *);
extern PetscErrorCode RiverGetTimeStep(River, RiverField *, PetscReal *);
extern PetscErrorCode RiverRHSFunctionLocal(River, RiverField *, RiverField *);
extern PetscErrorCode RiverCreateJacobian(River, Mat *, Mat *[]);
extern PetscErrorCode RiverCleanup(River);
extern PetscErrorCode RiverSetParameters(River, PetscInt);
extern PetscErrorCode RiverSetNumCells(River, PetscReal);
extern PetscErrorCode RiverSetInitialSolution(PetscInt, PetscInt, River, RiverField *, PetscReal, PetscReal);
extern PetscErrorCode RiverSetUp(River);
extern PetscErrorCode RiverSetElevation(River, PetscReal, PetscReal);
extern PetscErrorCode hydroRecreate(PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar *, PetscScalar *, PetscScalar *);
extern PetscErrorCode FroudNumber(PetscScalar, PetscScalar, PetscScalar *);
extern PetscErrorCode SuperbeeLimiter(PetscScalar, PetscScalar, PetscScalar *);
extern PetscErrorCode VanLeerLimiter(PetscScalar, PetscScalar, PetscScalar *);
extern PetscErrorCode VanAlbadaLimiter(PetscScalar, PetscScalar, PetscScalar *);
extern PetscErrorCode MinModLimiter(PetscScalar, PetscScalar, PetscScalar *);
#endif
