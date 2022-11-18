#if !defined(PETSCNETRP_H)
#define PETSCNETRP_H

#include <petscdm.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscksp.h>
#include <petscvec.h>

typedef struct _p_NetRP* NetRP;


PETSC_EXTERN PetscClassId NETRP_CLASSID;
PETSC_EXTERN PetscFunctionList NetRPList;

typedef enum {Linear,Nonlinear,Other} NetRPSolveType; 
typedef enum {Generic,Specific} NetRPPhysicsGenerality; 

  typedef PetscErrorCode (*NetRPSolveStar_User)(NetRP,PetscInt,PetscBool*, Vec, Vec); /* form is: NumEdges,EdgeIn? Array, U, UStar */
  typedef PetscErrorCode (*NetRPSolveFlux_User)(NetRP,PetscInt,PetscBool*, Vec, Vec); /* form is: NumEdges,EdgeIn? Array, U, Flux */
  typedef PetscErrorCode (*NetRPCreateLinearStar)(NetRP,PetscInt,PetscBool*,Vec,Vec,Mat); /* form is: NumEdges,EdgeIn? Array, U, Linear System for solving for Ustar */
  typedef PetscErrorCode (*NetRPCreateLinearFlux)(NetRP,PetscInt,PetscBool*,Vec,Vec,Mat); /* form is: NumEdges,EdgeIn? Array, U, Linear System for solving for Flux */
  typedef PetscErrorCode (*NetRPNonlinearEval)(NetRP,PetscInt,PetscBool*,Vec,Vec,Vec); /* form is: NumEdges,EdgeIn? Array, U, F(u), where F(U) is the nonlinear eval for the nonlinear Network Riemann Problem */
  typedef PetscErrorCode (*NetRPNonlinearJac)(NetRP,PetscInt,PetscBool*,Vec,Vec,Mat);  /* form is: NumEdges,EdgeIn? Array, U, Jacobian of the NonlinearEval */

typedef const char* NetRPType;
#define NETRPBLANK       "netrpblank"
#define NETRPLINEARIZED "netrplinearized"
#define NETRPOUTFLOW "netrpoutflow"
#define NETRPEXACTSWE "netrpexactswe"

PETSC_EXTERN PetscErrorCode NetRPInitializePackage(void);
PETSC_EXTERN PetscErrorCode NetRPFinalizePackage(void);

PETSC_EXTERN PetscErrorCode NetRPCreate(MPI_Comm,NetRP*);
PETSC_EXTERN PetscErrorCode NetRPDestroy(NetRP*);
PETSC_EXTERN PetscErrorCode NetRPReset(NetRP);
PETSC_EXTERN PetscErrorCode NetRPDuplicate(NetRP,NetRP*);
PETSC_EXTERN PetscErrorCode NetRPView(NetRP,PetscViewer); 

PETSC_EXTERN PetscErrorCode NetRPSetType(NetRP,NetRPType);
PETSC_EXTERN PetscErrorCode NetRPGetType(NetRP,NetRPType*);
PETSC_EXTERN PetscErrorCode NetRPRegister(const char[], PetscErrorCode (*)(NetRP));

PETSC_EXTERN PetscErrorCode NetRPSetFromOptions(NetRP);
PETSC_EXTERN PetscErrorCode NetRPSetUp(NetRP);
PETSC_EXTERN PetscErrorCode NetRPisSetup(NetRP,PetscBool*); 

PETSC_EXTERN PetscErrorCode NetRPSetSolveType(NetRP,NetRPSolveType); 
PETSC_EXTERN PetscErrorCode NetRPGetSolveType(NetRP,NetRPSolveType*); 

PETSC_EXTERN PetscErrorCode NetRPSetPhysicsGenerality(NetRP,NetRPPhysicsGenerality);
PETSC_EXTERN PetscErrorCode NetRPGetPhysicsGenerality(NetRP,NetRPPhysicsGenerality*);

PETSC_EXTERN PetscErrorCode NetRPGetNumFields(NetRP,PetscInt*);

PETSC_EXTERN PetscErrorCode NetRPSetApplicationContext(NetRP,void*);
PETSC_EXTERN PetscErrorCode NetRPGetApplicationContext(NetRP,void*);

PETSC_EXTERN PetscErrorCode NetRPSetFlux(NetRP,RiemannSolver);
PETSC_EXTERN PetscErrorCode NetRPGetFlux(NetRP,RiemannSolver*);

PETSC_EXTERN PetscErrorCode NetRPCanSolveStar(NetRP,PetscBool*); 

PETSC_EXTERN PetscErrorCode NetRPSolveStar(NetRP,PetscInt,PetscBool*, Vec, Vec);
PETSC_EXTERN PetscErrorCode NetRPSolveFlux(NetRP,PetscInt,PetscBool*, Vec, Vec);

/* Providing extra information to the cacheing ability of the problem */

PETSC_EXTERN PetscErrorCode NetRPAddVertexDegrees(NetRP,PetscInt,PetscInt*); 
PETSC_EXTERN PetscErrorCode NetRPGetVertexDegrees(NetRP,PetscInt*,PetscInt**); 
PETSC_EXTERN PetscErrorCode NetRPClearCache(NetRP); 

/* 
  Set internal ops, for usage when a user is using the default blank netrp, and wnat to specifically set there routines 
  This is an alternative the complexity of having to create an entire implementation just for a physics specific riemann problem 
  
  Will error out if anything but the "blank" netrp is used, as specific implementations should not allow these to be messed with 
*/

PETSC_EXTERN PetscErrorCode NetRPSetSolveStar(NetRP,NetRPSolveStar_User); 
PETSC_EXTERN PetscErrorCode NetRPSetSolveFlux(NetRP,NetRPSolveFlux_User); 
PETSC_EXTERN PetscErrorCode NetRPSetCreateLinearStar(NetRP,NetRPCreateLinearStar);
PETSC_EXTERN PetscErrorCode NetRPSetCreateLinearFlux(NetRP,NetRPCreateLinearFlux);
PETSC_EXTERN PetscErrorCode NetRPSetNonlinearEval(NetRP,NetRPNonlinearEval);
PETSC_EXTERN PetscErrorCode NetRPSetNonlinearJac(NetRP,NetRPNonlinearJac);

/* internal setup routines for solvers. Not called directly by users */
PETSC_INTERN PetscErrorCode NetRPCreateLinear(NetRP,PetscInt,Mat*,Vec*); 
PETSC_INTERN PetscErrorCode NetRPCreateKSP(NetRP,PetscInt,KSP*); 
PETSC_INTERN PetscErrorCode NetRPCreateSNES(NetRP,PetscInt,SNES*); 


/* internal access routines */
/* internal for now, as these could be easily used to shoot yourself in the foot */

/*

TODO: Implement these as necessary 

PETSC_INTERN PetscErrorCode NetRPGetSNES(NetRP,PetscInt,SNES*);
PETSC_INTERN PetscErrorCode NetRPGetKSP(NetRP,PetscInt,KSP*); 
PETSC_INTERN PetscErrorCode NetRPGetMat(NetRP,PetscInt,Mat*); 
*/
#endif