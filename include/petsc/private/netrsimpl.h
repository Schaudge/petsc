/* Network Riemann Solver implementation */
/* A WIP, use at your own risk */

/* Implementation of the Riemann Objects */

#if !defined(__NETRSIMPL_H)
#define __NETRSIMPL_H

#include <petscriemannsolver.h>
#include <petscnetrs.h>
#include <petsc/private/petscimpl.h>
#include <petscmat.h>
#include <petscsnes.h>

PETSC_EXTERN PetscBool NetRSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NetRSRegisterAll(void);

typedef enum {EDGEIN=0,EDGEOUT=1} EdgeDirection; /* temporary, to be replace (I think) */


typedef struct _NetRSOps *NetRSOps;
struct _NetRSOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,NetRS);
  PetscErrorCode (*setup)(NetRS);
  PetscErrorCode (*view)(NetRS,PetscViewer);
  PetscErrorCode (*destroy)(NetRS);
  PetscErrorCode (*reset)(NetRS);
  PetscErrorCode (*evaluate)(NetRS,const PetscReal*,const PetscBool*,PetscReal*); /* form is U, dir, numedges,flux */
};

struct _p_NetRS {
  PETSCHEADER(struct _NetRSOps);
  PetscBool      setupcalled; 
  void           *data; /* implementation object */
  void           *user; /* user context */
  PetscInt       numfields;
  PetscInt       numedges; 
  RiemannSolver  rs; /* For holding physics information, a hack for now to be replaced by FluxFunction */
  PetscReal      *flux_wrk; /* work array for the flux outputs size is numedges*numfields */
  /* The implementations are responsible for the creation and management of the objects needed for their evaluation 
  routines */
  NRSErrorEstimator estimate; 
};

#endif