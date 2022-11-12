/* Class for Network Riemann Solvers. Used for coupling conditions in networks */


#if !defined(PETSCNETRS_H)
#define PETSCNETRS_H
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>
#include <petscriemannsolver.h>
#include <petscdm.h>
#include <petscdmnetwork.h>
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashseti.h>
#include <petscnetrp.h>

typedef struct _p_NetRS* NetRS; 

PETSC_EXTERN PetscClassId NETRS_CLASSID;
PETSC_EXTERN PetscFunctionList NetRSList;

typedef const char* NetRSType;
#define  NETRSBASIC "basic"

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
typedef PetscErrorCode (*NRSErrorEstimator )(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode NetRSInitializePackage(void);
PETSC_EXTERN PetscErrorCode NetRSFinalizePackage(void);

PETSC_EXTERN PetscErrorCode NetRSCreate(MPI_Comm,NetRS*);
PETSC_EXTERN PetscErrorCode NetRSDestroy(NetRS*);
PETSC_EXTERN PetscErrorCode NetRSReset(NetRS);
PETSC_EXTERN PetscErrorCode NetRSResetVectorSpace(NetRS);
PETSC_EXTERN PetscErrorCode NetRSDuplicate(NetRS,NetRS*);

PETSC_EXTERN PetscErrorCode NetRSSetType(NetRS,NetRSType);
PETSC_EXTERN PetscErrorCode NetRSGetType(NetRS,NetRSType*);
PETSC_EXTERN PetscErrorCode NetRSRegister(const char[], PetscErrorCode (*)(NetRS));

PETSC_EXTERN PetscErrorCode NetRSSetFromOptions(NetRS);
PETSC_EXTERN PetscErrorCode NetRSSetUp(NetRS);

/* Replace with proper flux class */
PETSC_EXTERN PetscErrorCode NetRSSetFlux(NetRS,RiemannSolver);
PETSC_EXTERN PetscErrorCode NetRSGetFlux(NetRS,RiemannSolver*); 

PETSC_EXTERN PetscErrorCode NetRSSetNetwork(NetRS,DM); 
PETSC_INTERN PetscErrorCode NetRSGetNetwork(NetRS,DM*); /* Could potentially make public but ensure if is safe/smart */


PETSC_EXTERN PetscErrorCode NetRSSolve(NetRS,Vec,Vec); 

PETSC_EXTERN PetscErrorCode NetRSSetApplicationContext(NetRS,void*);
PETSC_EXTERN PetscErrorCode NetRSGetApplicationContext(NetRS,void*);

PETSC_EXTERN PetscErrorCode NetRSAddNetRPatVertex(NetRS,PetscInt,NetRP); 
// PETSC_EXTERN PetscErrorCode NetRSAddNetRPatVertices(NetRS,PetscInt,PetscInt*,NetRP); 



/* Error Estimator Support WIP */

/*
PETSC_EXTERN PetscErrorCode NetRSErrorEstimate(NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSSetErrorEstimate(NetRS,NRSErrorEstimator);
PETSC_EXTERN PetscErrorCode NetRSUseErrorEstimator(NetRS,PetscBool);
PETSC_EXTERN PetscErrorCode NetRSIsUsingErrorEstimator(NetRS,PetscBool*);
*/

/* error estimator implementations (perhaps its own class at a later date) */ 
/*
PETSC_EXTERN PetscErrorCode NetRSLaxErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSRoeErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NetRSTaylorErrorEstimate(void*,NetRS,PetscInt,const PetscReal*,const PetscReal*,PetscReal*);
*/

/* Adaptivity Support */ 

/*
PETSC_EXTERN PetscErrorCode NetRSUseAdaptivity(NetRS,PetscBool);
PETSC_EXTERN PetscErrorCode NetRSIsUsingeAdaptivity(NetRS,PetscBool*);
PETSC_EXTERN PetscErrorCode NetRSSetFineTol(NetRS,PetscReal);
*/

PETSC_EXTERN PetscErrorCode NetRSView(NetRS,PetscViewer);


/* Helper Functions, to be migrated to the correct locations later */

PETSC_EXTERN  PetscErrorCode DMNetworkCreateLocalEdgeNumbering(NetRS , DM); 
PETSC_EXTERN  PetscErrorCode DMNetworkCacheVertexDegrees(NetRS , DM);

PETSC_INTERN PetscErrorCode DMNetworkComputeUniqueVertexDegrees_Local(NetRS,DM,DMLabel,PetscHSetI*,PetscHSetI);
#endif