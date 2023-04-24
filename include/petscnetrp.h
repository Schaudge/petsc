#include "petscmacros.h"
#include "petscsystypes.h"
#if !defined(PETSCNETRP_H)
  #define PETSCNETRP_H

  #include <petscdm.h>
  #include <petscmat.h>
  #include <petscsnes.h>
  #include <petscksp.h>
  #include <petscriemannsolver.h>
  #include <petscvec.h>
  #include <petsctao.h>

typedef struct _p_NetRP *NetRP;

PETSC_EXTERN PetscClassId      NETRP_CLASSID;
PETSC_EXTERN PetscFunctionList NetRPList;

typedef enum {
  Linear,
  Nonlinear,
  Optimization,
  Other
} NetRPSolveType;
typedef enum {
  Generic,
  Specific
} NetRPPhysicsGenerality;

typedef enum {
  UndirectedVDeg,
  DirectedVDeg
} NetRPCacheType;

typedef enum {
  No_Default,
  Yes_Default,
  Yes_Manual,
  No_Manual
} NetRPCacheDirectedU;

typedef PetscErrorCode (*NetRPSolveStar_User)(NetRP, PetscInt, PetscBool *, Vec, Vec);        /* form is: NumEdges,EdgeIn? Array, U, UStar */
typedef PetscErrorCode (*NetRPSolveFlux_User)(NetRP, PetscInt, PetscBool *, Vec, Vec);        /* form is: NumEdges,EdgeIn? Array, U, Flux */
typedef PetscErrorCode (*NetRPCreateLinearStar)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat); /* form is: NumEdges,EdgeIn? Array, U, Linear System for solving for Ustar */
typedef PetscErrorCode (*NetRPCreateLinearFlux)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat); /* form is: NumEdges,EdgeIn? Array, U, Linear System for solving for Flux */
typedef PetscErrorCode (*NetRPNonlinearEval)(NetRP, PetscInt, PetscBool *, Vec, Vec, Vec);    /* form is: NumEdges,EdgeIn? Array, U, F(u), where F(U) is the nonlinear eval for the nonlinear Network Riemann Problem */
typedef PetscErrorCode (*NetRPNonlinearJac)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat);     /* form is: NumEdges,EdgeIn? Array, U, Jacobian of the NonlinearEval */
typedef PetscErrorCode (*NetRPSetSolverCtx)(NetRP, PetscInt, PetscInt, void **);
typedef PetscErrorCode (*NetRPDestroySolverCtx)(NetRP, PetscInt, PetscInt, void *);

typedef PetscErrorCode (*NetRPPreSolveFunc)(NetRP, PetscInt, PetscInt, PetscBool *, Vec, void *);
typedef PetscErrorCode (*NetRPPostSolveFunc)(NetRP, PetscInt, PetscInt, PetscBool *, Vec, Vec, void *);

typedef const char *NetRPType;
  #define NETRPBLANK               "netrpblank"
  #define NETRPLINEARIZED          "netrplinearized"
  #define NETRPOUTFLOW             "netrpoutflow"
  #define NETRPEXACTSWE            "netrpexactswe"
  #define NETRPTRAFFICLWR          "netrptrafficLWR"
  #define NETRPTRAFFICLWR_PRIORITY "netrptrafficLWRpriority"
  #define NETRPCONSTANT            "netrpconstant"

PETSC_EXTERN PetscErrorCode NetRPInitializePackage(void);
PETSC_EXTERN PetscErrorCode NetRPFinalizePackage(void);

PETSC_EXTERN PetscErrorCode NetRPCreate(MPI_Comm, NetRP *);
PETSC_EXTERN PetscErrorCode NetRPDestroy(NetRP *);
PETSC_EXTERN PetscErrorCode NetRPReset(NetRP);
PETSC_EXTERN PetscErrorCode NetRPDuplicate(NetRP, NetRP *);
PETSC_EXTERN PetscErrorCode NetRPView(NetRP, PetscViewer);

PETSC_EXTERN PetscErrorCode NetRPSetType(NetRP, NetRPType);
PETSC_EXTERN PetscErrorCode NetRPGetType(NetRP, NetRPType *);
PETSC_EXTERN PetscErrorCode NetRPRegister(const char[], PetscErrorCode (*)(NetRP));

PETSC_EXTERN PetscErrorCode NetRPSetFromOptions(NetRP);
PETSC_EXTERN PetscErrorCode NetRPSetUp(NetRP);
PETSC_EXTERN PetscErrorCode NetRPisSetup(NetRP, PetscBool *);

PETSC_EXTERN PetscErrorCode NetRPSetSolveType(NetRP, NetRPSolveType);
PETSC_EXTERN PetscErrorCode NetRPGetSolveType(NetRP, NetRPSolveType *);

PETSC_EXTERN PetscErrorCode NetRPSetPhysicsGenerality(NetRP, NetRPPhysicsGenerality);
PETSC_EXTERN PetscErrorCode NetRPGetPhysicsGenerality(NetRP, NetRPPhysicsGenerality *);

PETSC_EXTERN PetscErrorCode NetRPGetNumFields(NetRP, PetscInt *);

PETSC_EXTERN PetscErrorCode NetRPSetApplicationContext(NetRP, void *);
PETSC_EXTERN PetscErrorCode NetRPGetApplicationContext(NetRP, void *);

PETSC_EXTERN PetscErrorCode NetRPSetFlux(NetRP, RiemannSolver);
PETSC_EXTERN PetscErrorCode NetRPGetFlux(NetRP, RiemannSolver *);

PETSC_EXTERN PetscErrorCode NetRPCanSolveStar(NetRP, PetscBool *);

PETSC_EXTERN PetscErrorCode NetRPSolveStar(NetRP, PetscInt, PetscInt, PetscBool *, Vec, Vec);
PETSC_EXTERN PetscErrorCode NetRPSolveFlux(NetRP, PetscInt, PetscInt, PetscBool *, Vec, Vec);

/* Providing extra information to the cacheing ability of the problem */

PETSC_INTERN PetscErrorCode NetRPAddVertexDegrees_internal(NetRP, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode NetRPAddDirVertexDegrees_internal(NetRP, PetscInt, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode NetRPCacheSolvers(NetRP, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode NetRPGetNumCached(NetRP, PetscInt *);
PETSC_EXTERN PetscErrorCode NetRPClearCache(NetRP);
PETSC_EXTERN PetscErrorCode NetRPGetCacheType(NetRP, NetRPCacheType *);
PETSC_EXTERN PetscErrorCode NetRPSetCacheType(NetRP, NetRPCacheType);
PETSC_EXTERN PetscErrorCode NetRPSetCacheUDirected(NetRP, PetscBool);
PETSC_EXTERN PetscErrorCode NetRPGetCacheUDirected(NetRP, PetscBool *);

PETSC_EXTERN PetscErrorCode NetRPSetSolverCtxFunc(NetRP, NetRPSetSolverCtx);
PETSC_EXTERN PetscErrorCode NetRPGetSolverCtx(NetRP, PetscInt, PetscInt, void **);
PETSC_EXTERN PetscErrorCode NetRPSetDestroySolverCtxFunc(NetRP, NetRPDestroySolverCtx);

PETSC_EXTERN PetscErrorCode NetRPCreateVec(NetRP, PetscInt, Vec *);

/* 
  Set internal ops, for usage when a user is using the default blank netrp, and wnat to specifically set there routines 
  This is an alternative the complexity of having to create an entire implementation just for a physics specific riemann problem 
  
  Will error out if anything but the "blank" netrp is used, as specific implementations should not allow these to be messed with 
*/

PETSC_INTERN PetscErrorCode NetRPPostSolve(NetRP, PetscInt, PetscInt, PetscBool *, Vec, Vec);
PETSC_EXTERN PetscErrorCode NetRPSetPostSolve(NetRP, NetRPPostSolveFunc);

PETSC_INTERN PetscErrorCode NetRPPreSolve(NetRP, PetscInt, PetscInt, PetscBool *, Vec);
PETSC_EXTERN PetscErrorCode NetRPSetPreSolve(NetRP, NetRPPreSolveFunc);

PETSC_EXTERN PetscErrorCode NetRPSetSolveStar(NetRP, NetRPSolveStar_User);
PETSC_EXTERN PetscErrorCode NetRPSetSolveFlux(NetRP, NetRPSolveFlux_User);

PETSC_EXTERN PetscErrorCode NetRPSetCreateLinearStar(NetRP, NetRPCreateLinearStar);
PETSC_EXTERN PetscErrorCode NetRPSetCreateLinearFlux(NetRP, NetRPCreateLinearFlux);

PETSC_EXTERN PetscErrorCode NetRPSetNonlinearEval(NetRP, NetRPNonlinearEval);
PETSC_EXTERN PetscErrorCode NetRPSetNonlinearJac(NetRP, NetRPNonlinearJac);

/* internal setup routines for solvers. Not called directly by users */
PETSC_INTERN PetscErrorCode NetRPCreateLinear(NetRP, PetscInt, Mat *, Vec *);
PETSC_INTERN PetscErrorCode NetRPCreateKSP(NetRP, PetscInt, KSP *);
PETSC_INTERN PetscErrorCode NetRPCreateSNES(NetRP, PetscInt, SNES *);
PETSC_INTERN PetscErrorCode NetRPCreateTao(NetRP, PetscInt, PetscInt, void *, Tao *);

/* internal access routines */
/* internal for now, as these could be easily used to shoot yourself in the foot */

/*
  TODO: Implement these as necessary
  PETSC_INTERN PetscErrorCode NetRPGetSNES(NetRP,PetscInt,SNES*);
  PETSC_INTERN PetscErrorCode NetRPGetKSP(NetRP,PetscInt,KSP*); 
  PETSC_INTERN PetscErrorCode NetRPGetMat(NetRP,PetscInt,Mat*); 
*/

/* Traffic Specific Functions */

typedef PetscErrorCode (*NetRPTrafficDistribution)(NetRP, PetscInt, PetscInt, Mat);
typedef PetscErrorCode (*NetRPTrafficPriorityVec)(NetRP, PetscInt, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode NetRPTrafficSetDistribution(NetRP, NetRPTrafficDistribution);
PETSC_EXTERN PetscErrorCode NetRPTrafficSetFluxMaximumPoint(NetRP, PetscReal);
PETSC_EXTERN PetscErrorCode NetRPTrafficGetFluxMaximumPoint(NetRP, PetscReal *);

PETSC_EXTERN PetscErrorCode NetRPTrafficSetPriorityVec(NetRP, NetRPTrafficPriorityVec);
PETSC_EXTERN PetscErrorCode NetRPTrafficSetPriorityWeight(NetRP, PetscReal);
PETSC_EXTERN PetscErrorCode NetRPTrafficGetPriorityWeight(NetRP, PetscReal *);

/* constant specific functions */

PETSC_EXTERN PetscErrorCode NetRPConstantSetData(NetRP, PetscScalar *);
#endif