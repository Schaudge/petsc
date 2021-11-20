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

typedef struct _NetRSOps *NetRSOps;
struct _NetRSOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,NetRS);
  PetscErrorCode (*setup)(NetRS);
  PetscErrorCode (*view)(NetRS,PetscViewer);
  PetscErrorCode (*destroy)(NetRS);
  PetscErrorCode (*reset)(NetRS);
  PetscErrorCode (*evaluate)(NetRS,const PetscReal*,const EdgeDirection*,PetscReal*,PetscReal*); /* form is U, dir, numedges,flux */
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

  /* Error Estimator Support
    Maybe make error estimator a distinct class? Would help with these work arrays 
    and switching/print routines (maybe) */ 

  NRSErrorEstimator estimate;
  PetscReal         *est_wrk;  /* numfields work array for error estimators to use */
  PetscReal         *est_wrk2; /* numfields work array for error estimators to use */
  PetscReal         *error;    /* storage for the error estimator outputs  numedge entries */
  PetscBool         useestimator; 

  /* Adaptivity Support */ 

  /* 
    Note: We do adaptivity of netRS by swapping to a "finer" netrs when the internal error estimator goes above 
    the user set tolerance. The logic for this is controlled by the NetRSEvaluate() function. In principle 
    this allows of arbitrary recursion for the netrs, but I intend to only use 2-level systems. 
  */
  NetRS  fine; 
  PetscReal finetol; /* tolerance to switch to a finer NetRS */
  PetscReal coarsetol; /* tolerance to swtich to a coarser NetRS */ 
  PetscBool useadaptivity; 
  NetRSType finetype; /* used to set what type the fine type should be. By default the is the "exact lax curve solver" if available.
                        FOR NOW AS EXACT SOLVERS ARE HACKED TOGETHER WILL ONLY WORK FOR SWE */
};

#endif