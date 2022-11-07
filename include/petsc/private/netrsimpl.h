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
#include <petscdm.h>
#include <petscdmlabel.h>

PETSC_EXTERN PetscBool NetRSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NetRSRegisterAll(void);

typedef enum {Network_Not_Created, Network_Internal, Network_User} NetRSNetworkState; 


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

  DM             network; /* internal DMNetwork for storing data about the topology of the Riemann Problem*/
  NetRSNetworkState network_state;
  

  /* TODO: Make PetscFlux class */
  PetscInt       numfields; /* store inside the DM ? No, store in the PetscFlux Class */
  RiemannSolver  rs; /* For holding physics information, a hack for now to be replaced by FluxFunction */


  /* For setting up the preallocation of objects and constructing the sub NetRS problems */

  DMLabel        subgraphs; /* TODO : Name better. Each stratum corresponds to the set of vertices associated with a 
  specific NetRS solver  */

  PetscHSetI     vertexdegrees_total; /* set of all vertex degrees in the full local network */
  PetscHSetI     *vertexdegrees; /* set of all vertex degrees for each subgraph induced by the DMLabel */


  PetscHMapI     hmap_total; /* hash map for all work arrays and etc for the total network (shared among all sub NetRS solvers)*/
  PetscHMapI     hmap; /* hash map or all work arrays/ solvers for the sub NetRS solvers */

  PetscReal      **flux_wrk; /* work array for the flux outputs, flux_wrk[hmap[numedges]] is an array of size numedges*numfields as a work array */
  /* The implementations are responsible for the creation and management of the objects needed for their evaluation 
  routines */

  /* Solver Work variable: These are managed by the NetRS class and creation/destruction/efficient managament are done by this 
  NetRS interface class. Particular implementations are then free to essentially ignore any memory managment for solvers, and can do 
  just implementations for the local network solvers and let this NetRS class manage efficient reuse of resources */


  /* Note that not all of these may actually be allocated or used, the implementation is responsible for marking 
  in its creation routine what solvers/objects it will actually need. The implementations are also responsible for setting up the particulars 
  of the solvers if they want non-default options */

  Mat            **mat_wrk; 
  SNES           **snes_wrk; 
  KSP            **ksp_wrk;

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