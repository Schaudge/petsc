
/* Implementation of the Local Riemann Solver. To be used entirely by the NetRS class. */

/* This is concerned entirely with the local Riemann problem, the details of setting up the global 
riemann problem and connecting these problems to an actual network is handled by the NetRS class. 

Essentially an interface to generate riemann problems/solvers. However, note that the solver setup and etc are not mandatory 
to be generated by a user explicitly as there are default options to build and construct things.

*/

#if !defined(__NETRPIMPL_H)
  #define __NETRPIMPL_H

  #include <petsc/private/petscimpl.h>
  #include <petscriemannsolver.h> /* for the physics for now */
  #include <petscnetrp.h>
  #include <petsc/private/hashmapi.h>
  #include <petsc/private/hashseti.h>
  #include <petsc/private/hashmapij.h>

PETSC_EXTERN PetscLogEvent NetRP_Solve_Total;
PETSC_EXTERN PetscLogEvent NetRP_Solve_SetUp;
PETSC_EXTERN PetscLogEvent NetRP_Solve_System;

PETSC_EXTERN PetscBool      NetRPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NetRPRegisterAll(void);

typedef struct _NetRPOps *NetRPOps;

struct _NetRPOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems *, NetRP);
  PetscErrorCode (*setup)(NetRP);
  PetscErrorCode (*view)(NetRP, PetscViewer);
  PetscErrorCode (*destroy)(NetRP);
  PetscErrorCode (*reset)(NetRP);
  PetscErrorCode (*clearcache)(NetRP);
  PetscErrorCode (*setsolverctx)(NetRP, PetscInt, PetscInt, void **);
  PetscErrorCode (*destroysolverctx)(NetRP, PetscInt, PetscInt, void *);
  PetscErrorCode (*setupmat)(NetRP, PetscInt, Mat);
  PetscErrorCode (*setupksp)(NetRP, PetscInt, KSP);
  PetscErrorCode (*setupsnes)(NetRP, PetscInt, SNES);
  PetscErrorCode (*setuptao)(NetRP, PetscInt, PetscInt, Tao); // edges in and edges out
  PetscErrorCode (*setupjac)(NetRP, PetscInt, Mat);
  PetscErrorCode (*solveStar)(NetRP, PetscInt, PetscBool *, Vec, Vec);             /* form is: DMNetwork, Vertex, U, UStar */
  PetscErrorCode (*solveFlux)(NetRP, PetscInt, PetscBool *, Vec, Vec);             /* form is: DMNetwork, Vertex, U, Flux */
  PetscErrorCode (*createLinearStar)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat); /* form is: DMNetwork, Vertex, U, Linear System for solving for Ustar */
  PetscErrorCode (*createLinearFlux)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat); /* form is: DMNetwork, Vertex, U, Linear System for solving for Flux */
  PetscErrorCode (*NonlinearEval)(NetRP, PetscInt, PetscBool *, Vec, Vec, Vec);    /* form is: DMNetwork, Vertex,U, Ustar, F(ustar), where F(U) is the nonlinear eval for the nonlinear Network Riemann Problem */
  PetscErrorCode (*NonlinearJac)(NetRP, PetscInt, PetscBool *, Vec, Vec, Mat);     /* form is: DMNetwork, Vertex, U,Ustar Jacobian of the NonlinearEval */

  PetscErrorCode (*PostSolve)(NetRP, PetscInt, PetscInt, PetscBool *, Vec, Vec, void *);
  PetscErrorCode (*PreSolve)(NetRP, PetscInt, PetscInt, PetscBool *, Vec, void *); /* form is vdegin, vdegout, edgein, U, solver_ctx */

  /* TAO Stuff */
  /* Note: This entire frameWork needs to be redone. Honestly, I think a generic 
  batched solvers attached to DM's is necessary, which requires more interaction with the batched stuff */
};

struct _p_NetRP {
  PETSCHEADER(struct _NetRPOps);
  PetscBool setupcalled;
  void     *data; /* implementation object */
  void     *user; /* user context */

  /* physics information: To be reworked into different class */
  RiemannSolver flux;

  /* parameter values. These determine functionality and behavior of NetRP*/
  NetRPCacheDirectedU    cacheU; /*  whether to cache the directed input vectors */
  NetRPCacheType         cachetype;
  NetRPSolveType         solvetype;
  NetRPPhysicsGenerality physicsgenerality;
  PetscInt               numfields; /* the problems number of fields, if physics generality is general this is copied from the physics 
                                otherwise this must be set manually and will error if the the physics does not match */

  /* internal storage */

  /* Cached solver objects. When using built-in petsc solvers, implementations do not have to worry about creating and managing 
     these solvers and they will be cached and managed automatically */
  Mat  *mat;
  Vec  *vec;
  KSP  *ksp;
  SNES *snes;
  Tao  *tao;

  Vec *Uin, *Uout; /* Vectors for storing the Uin, and Uout components of the input U vector. 
  This is an optional cache didcted by cacheU */

  void **solver_ctx; /* User ctx for the cached solvers. */

  /* Cache Type; 
      UndirectedVDeg : Assumes that solvers are parameterized by vdeg. 
      Space based on this assumption 
      DirectVdeg     : Assumes the solvers are parameterized by invdeg X outvdeg 

      This changes the type of hash map used
   */
  /* Only one hmap is actually used, depending on the structure of the Riemann Problem, specified by NetRPCacheType */
  PetscHMapI  hmap;    /* map from vertexdegree -> index in the solver arrays for cached solver objects for that vertex degree problem */
  PetscHMapIJ dirhmap; /* map from vdegin X vdegout -> index in the solver arrays for cached solver objects for that vdegin X vdegout degree problem **/
  /* Type of Solver */
  /* What generality of physics the local "solver" can handle.*/
};
#endif