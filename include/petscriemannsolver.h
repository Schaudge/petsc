#if !defined(PETSCRIEMANNSOLVER_H)
#define PETSCRIEMANNSOLVER_H
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>
/* Name Objects similarly to petscds */

/* Flux function in balance law evaluated at a point, specifically in an equation of the form 
   \partial_t u + \div (F(u)) = G(u,x) (a Balance law)
   
   where F : \mathbb{R}^dim \to \mathbb{R}^m 
   dim : dimension of the equation (1,2,3 dimensional in general)
   m   : Size of the system of equations 

   This function is then the specification for F. Currently only using it for dim = 1 problems 
   so there is no assumptions on the format of the output (yet!) in terms of assumed m x dim or dim x m output
   structure for the c array. 
*/

/* 
    Note : Perhaps this would be better as a class as well, instead of a patterned function call? Frankly though 
    I won't know unless I start working with UFL, TSFC, Firedrake fenics etc... to get a sense of what is 
    needed/convenient. Will work for now. 
*/
typedef void (*PetscPointFlux)(void*,const PetscReal*,PetscReal*);

/* 
    WIP: structure for holding nonlinear (or maybe linear) function calls. This intended to hold the "physics"
    components of the Riemann Solver. Namely flux function point evaluations, point derivatives eigen expansions etc.
    as well as memory alloation for storing all these outputs. This is a simple refactoring to move components associated 
    with the RiemannSolver to this class instead, so it can be reused by other codes, namely DG kernels, this RS, 
    Netowork RiemannSolvers (RSN) 

    This is needed as we have a seperate class for Network Riemann Solvers which also needs this physics data, so it 
    doesn't make sense to associate this with the RS directly. Also makes generating physics for testing 
    and etc much much easier. 
*/


/* 
   Similarily I need to experiment with how to compute/store PointFlux derivative information. Honestly this may be 
   better to store in a "Flux" class. This is the same (or similar) problem that PetscDS is tring to solve. For now 
   it will be a patterned function call. 
*/

typedef PetscErrorCode (*PetscPointFluxDer)(void*,const PetscReal*,Mat);

/* 
   And again riemann solvers need to be able to compute eigenvalues of the system, for usage in wave-speed estimates. 
   Should design a seperate flux class to store all of these things. Then again, I guess these function callbacks 
   should be exposed anyways, in the case when the physics is not specified in a "PETSc" way. 

   Note: This allows for user specied analytically eigenvalue functions to be specified. In general, when such 
   a function is not available, the riemann solver should default to using the derivative information 
   and slepc (if available, or another eigenvalue solver compiled with petsc) to compute the eigenvalue information. 
   If the derivative is not specified analytically, finite differencing could be used to approximate it as well, 
   however this would be a terrible idea for performance so would only be recommened for debugging. 
   
   This function returns a pointer to an internal eigenvalue work array
*/

typedef void (*PetscPointFluxEig)(void*,const PetscReal*,PetscScalar*); 


/*S
     RiemannSolver - Abstract PETSc Riemann Solver Object

   Level: beginner

.seealso:  TODO
S*/
typedef struct _p_RiemannSolver* RiemannSolver; 

/* Eigen Decomposition Support */ 
/* Note: Should be a seperate class in itself I think. Maybe */
typedef PetscErrorCode (*RiemannSolverEigBasis)(void*,const PetscReal*,Mat);

typedef PetscErrorCode (*RiemannSolverMaxWaveSpeed)(RiemannSolver,const PetscReal*,const PetscReal*,PetscReal*);
typedef PetscErrorCode (*RiemannSolverRoeMatrix)(void*,const PetscReal*,const PetscReal*, Mat); 
typedef PetscErrorCode (*RiemannSolverRoeAvg)(void*,const PetscReal*,const PetscReal*,PetscReal*); 

/* Lax Curve Support (as with other flux function specific data, should be associated with the flux function not the riemann solver direclty) */ 

/* This is to be redone as I have more knowledge of lax curve analytical formulation. */ 

/* For now I'm making the naive assumption that the curve is paramaterized by the first conservative variable ( so in the SWE case 
paramaterized by hbar) */ 

/* my notation is that bar refers to points along the curve and star refers only to those specific points on the curve that solve the riemann 
problem */ 

/* 
Inputs: 
   rs - RiemannSolver (to be replaced wih flux function) 
   u  - state variable generating the curve 
   xi - paramaterization variable for the curve to evaluate at (note that in other portions of the code it will be assumed that this corresponds 
         to the first conservative variables)
   wavenumber - integer corresponding to the particular lax curve to compute (indexed starting from 1 as is standard in the literature)
Outputs: 
   ubar - state variable on the curve corresponding to the point xi (allocated by caller)
*/ 

typedef PetscErrorCode (*LaxCurve)(RiemannSolver,const PetscReal*,PetscReal,PetscInt,PetscReal*);

/*J
    RiemannSolverType - String with the name of a PETSc RiemmanSolver

   Level: beginner

.seealso: TODO 
J*/

typedef const char* RiemannSolverType;
#define RIEMANNLAXFRIEDRICH "lax"

/* Logging support */
PETSC_EXTERN PetscClassId RIEMANNSOLVER_CLASSID;

PETSC_EXTERN PetscErrorCode RiemannSolverInitializePackage(void);
PETSC_EXTERN PetscErrorCode RiemannSolverFinalizePackage(void);

PETSC_EXTERN PetscErrorCode RiemannSolverCreate(MPI_Comm,RiemannSolver*);
PETSC_EXTERN PetscErrorCode RiemannSolverDestroy(RiemannSolver*);
PETSC_EXTERN PetscErrorCode RiemannSolverReset(RiemannSolver);

PETSC_EXTERN PetscFunctionList RiemannSolverList;
PETSC_EXTERN PetscErrorCode RiemannSolverSetType(RiemannSolver,RiemannSolverType);
PETSC_EXTERN PetscErrorCode RiemannSolverGetType(RiemannSolver,RiemannSolverType*);
PETSC_EXTERN PetscErrorCode RiemannSolverRegister(const char[], PetscErrorCode (*)(RiemannSolver));

PETSC_EXTERN PetscErrorCode RiemannSolverSetFromOptions(RiemannSolver);
PETSC_EXTERN PetscErrorCode RiemannSolverSetUp(RiemannSolver);

PETSC_EXTERN PetscErrorCode RiemannSolverEvaluate(RiemannSolver,const PetscReal*,const PetscReal*, PetscReal**,PetscReal*);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeEig(RiemannSolver,const PetscReal*,PetscScalar**);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeMaxSpeed(RiemannSolver,const PetscReal*,const PetscReal*,PetscReal*);

/* Callbacks and interface for Roe Matrices and related solvers. Currently a WIP, as I implement 
more Roe Solvers and see what I need/ what is convenient */
PETSC_EXTERN PetscErrorCode RiemannSolverComputeRoeMatrix(RiemannSolver,const PetscReal*,const PetscReal*,Mat*);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeRoeMatrixInv(RiemannSolver,const PetscReal*,const PetscReal*,Mat*);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeRoeEig(RiemannSolver,const PetscReal*,const PetscReal*,PetscScalar**);
PETSC_EXTERN PetscErrorCode RiemannSolverSetRoeMatrixFunct(RiemannSolver,RiemannSolverRoeMatrix);
PETSC_EXTERN PetscErrorCode RiemannSolverCharNorm(RiemannSolver, const PetscReal*, const PetscReal*, PetscInt,PetscReal*);

/* Avoid using these, I think they are unnecessary and will be removed. If the roe matrix/decomposition can be computed 
using an explicit roe average state (instead of roe matrix directly), they will know and can pass that formulation into 
the RiemannSolver on their end. Not necessary for the Riemann Solver to know this. Will see as I add new implementations */

PETSC_EXTERN PetscErrorCode RiemannSolverComputeRoeAvg(RiemannSolver,const PetscReal*,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode RiemannSolverSetRoeAvgFunct(RiemannSolver,RiemannSolverRoeAvg);

PETSC_EXTERN PetscErrorCode RiemannSolverSetJacobian(RiemannSolver,PetscPointFluxDer);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeJacobian(RiemannSolver,const PetscReal*,Mat*);

PETSC_EXTERN PetscErrorCode RiemannSolverSetFluxEig(RiemannSolver ,PetscPointFluxEig);
PETSC_EXTERN PetscErrorCode RiemannSolverSetEigBasis(RiemannSolver,RiemannSolverEigBasis);
PETSC_EXTERN PetscErrorCode RiemannSolverComputeEigBasis(RiemannSolver,const PetscReal*,Mat*);
PETSC_EXTERN PetscErrorCode RiemannSolverChangetoEigBasis(RiemannSolver,const PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode RiemannSolverSetFlux(RiemannSolver,PetscInt,PetscInt,PetscPointFlux);
PETSC_EXTERN PetscErrorCode RiemannSolverSetMaxSpeedFunct(RiemannSolver ,RiemannSolverMaxWaveSpeed);
PETSC_EXTERN PetscErrorCode RiemannSolverSetFluxDim(RiemannSolver,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode RiemannSolverSetApplicationContext(RiemannSolver,void*);
PETSC_EXTERN PetscErrorCode RiemannSolverGetApplicationContext(RiemannSolver,void*);

/* Diagnostic Functions */ 
PETSC_EXTERN PetscErrorCode RiemannSolverTestEigDecomposition(RiemannSolver,PetscInt,const PetscReal**,PetscReal,PetscBool*,PetscReal*,PetscViewer);
PETSC_EXTERN PetscErrorCode RiemannSolverTestRoeMat(RiemannSolver,PetscInt,const PetscReal**,const PetscReal**,PetscReal,PetscBool*,PetscReal*,PetscViewer);

PETSC_EXTERN PetscErrorCode RiemannSolverView(RiemannSolver,PetscViewer);

/* Flux Function Stuff (to be made its own class) */

PETSC_EXTERN PetscErrorCode RiemannSolverSetLaxCurve(RiemannSolver,LaxCurve);
PETSC_EXTERN PetscErrorCode RiemannSolverEvalLaxCurve(RiemannSolver,const PetscReal*,PetscReal,PetscInt,PetscReal*);



#endif