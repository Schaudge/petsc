/* Class for Network Riemann Solvers. Used for coupling conditions in networks */


#if !defined(PETSCNETRS_H)
#define PETSCNETRS_H
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>
#include <petscriemannsolver.h>

typedef struct _p_NetRS* NetRS; 

PETSC_EXTERN PetscClassId NETRS_CLASSID;
PETSC_EXTERN PetscFunctionList NetRSList;

typedef const char* NetRSType;
#define NETRSEXACTSWE "exactswe"
#define NETRSLINEAR "netrslinear"
#define NETRSOUTFLOW "netrsoutflow"
#define NETRSRIEMANN "netrsriemann"
#define NETRSEXACTSWESTAR "exactswestar"
#define NETRSLINEARSTAR  "netrslinearstar"

/* Error Estimator for NetRS solves. Used when using adaptive 
NetRS solvers. WIP/Current Research on how to do this robustely
 This function definition may change as I experiment */ 

 /* 
    Note: May become a class itself as these might get complicated 
 */
/*
Input: 
.ctx 
.rs
.dir
.u
.ustar 
Output: 
.errorestimate
*/
typedef PetscErrorCode (*NRSErrorEstimator )(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*); /* Riemann solver to be replaced by flux function class */

typedef enum {EDGEIN=0,EDGEOUT=1} EdgeDirection; 

PETSC_EXTERN PetscErrorCode NetRSInitializePackage(void);
PETSC_EXTERN PetscErrorCode NetRSFinalizePackage(void);

PETSC_EXTERN PetscErrorCode NetRSCreate(MPI_Comm,NetRS*);
PETSC_EXTERN PetscErrorCode NetRSDestroy(NetRS*);
PETSC_EXTERN PetscErrorCode NetRSReset(NetRS);
PETSC_EXTERN PetscErrorCode NetRSDuplicate(NetRS,NetRS*);

PETSC_EXTERN PetscErrorCode NetRSSetType(NetRS,NetRSType);
PETSC_EXTERN PetscErrorCode NetRSGetType(NetRS,NetRSType*);
PETSC_EXTERN PetscErrorCode NetRSRegister(const char[], PetscErrorCode (*)(NetRS));

PETSC_EXTERN PetscErrorCode NetRSSetFromOptions(NetRS);
PETSC_EXTERN PetscErrorCode NetRSSetUp(NetRS);

PETSC_EXTERN PetscErrorCode NetRSSetRiemannSolver(NetRS,RiemannSolver);
PETSC_EXTERN PetscErrorCode NetRSSetNumEdges(NetRS,PetscInt);

PETSC_EXTERN PetscErrorCode NetRSEvaluate(NetRS,const PetscReal*,const EdgeDirection*,PetscReal**,PetscReal**);

PETSC_EXTERN PetscErrorCode NetRSSetApplicationContext(NetRS,void*);
PETSC_EXTERN PetscErrorCode NetRSGetApplicationContext(NetRS,void*);

/* Error Estimator Support WIP */
PETSC_EXTERN PetscErrorCode NetRSErrorEstimate(NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSSetErrorEstimate(NetRS,NRSErrorEstimator);

/* error estimator implementations (perhaps its own class at a later date) */ 
PETSC_EXTERN PetscErrorCode NetRSLaxErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSRoeErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSTaylorErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);


PETSC_EXTERN PetscErrorCode NetRSView(NetRS,PetscViewer);
#endif