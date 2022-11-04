/* Experimental Riemann Solver Class. To all for functor riemann solvers (storing parameters for example, allowing functions
to generate riemann solver objects, abstract riemann solver selection, internal storage of solver objects
(i.e. linear and nonlinear solvers etc)) */

/* A WIP, use at your own risk */

/* Implementation of the Riemann Objects */

#if !defined(__RIEMANNSOLVERIMPL_H)
#define __RIEMANNSOLVERIMPL_H

#include <petscriemannsolver.h>
#include <petsc/private/petscimpl.h>
#include <petscmat.h>
#include <petscsnes.h>

PETSC_EXTERN PetscBool RiemannSolverRegisterAllCalled;
PETSC_EXTERN PetscErrorCode RiemannSolverRegisterAll(void);

typedef struct _PetscRiemannOps *PetscRiemannOps;
struct _PetscRiemannOps {
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,RiemannSolver);
  PetscErrorCode (*setup)(RiemannSolver);
  PetscErrorCode (*view)(RiemannSolver,PetscViewer);
  PetscErrorCode (*destroy)(RiemannSolver);
  PetscErrorCode (*reset)(RiemannSolver);
  /* Apply the Riemann Solver 
    evaluate (uL*,uR*,flux*,&maxspeed) 
    Inputs
      ul : left state 
      ur : right state 
    Internal Updates 
      flux_wrk : Numerical Flux function output 
      maxspeed : maximum wave speed calculated (for use in cfl computations elsewhere) 

  NOTE: 
    1. Does not work for spatially dependant flux functions 
    2. Might be a good idea to have different versions of evaluate for different inputs? I.e uL, uR as 
       PetscVectors instead of c-style arrays? 
    3. Internally store maxspeeds?
    4. Still need to figure out the right format for higher dimensional results. Tensor formats? 
    5. For now it uses point-wise outputs, but having a time dependant function, or the abilituy 
       to extend the results outside of the point of evualation might be useful 
    6. Also I'm assuming a standard riemann solver, what about the generalized riemann solvers 
    of Toro (related to 5.) and the generalized riemann solvers for networks. These should all be part of the 
    same class right? 
  */
  PetscErrorCode (*evaluate)(RiemannSolver,const PetscReal*, const PetscReal*); 
/*
   Development Note: This might not be the correct way to do this. As I think further it seems sensible that 
   particular implementations might have their own version for this function. I guess I'll write it as a RiemannSolver 
   operation like evaulaute? Some implementations would prefer to have the option of outputting wave speeds and wave
   decomposition, instead of directly outputting flux. 
*/
/* 
   Function specificaiton for computing the maximum wave speed between two points in a riemann problem.
   Default behavior in the convex fluxfun case will be given by 
   maxspeed = max(max(abs(eig(Df(uL)))),max(abs(eig(Df(uR))))), 
   which will always give the correct answer when f is convex. In the non-convex case I need to look up 
   proper default behavior. 
*/

 RiemannSolverMaxWaveSpeed computemaxspeed; 
};

struct _p_RiemannSolver {
  PETSCHEADER(struct _PetscRiemannOps);

  PetscBool      setupcalled;
  void           *data; /* implementation object */
  void           *user; /* user context */
  DM             dm;    /* internal dm that manages the mesh for the Riemann Problem. Solvers types/implementations 
  are dispatched based on the dm type and properties of the dm. 
  Note: we have a seperate specific class for RiemannSolvers on DMNetwork, as the Riemann Problem for Networks 
  is distinct from standard riemann problems, and (I think) requires a distinct interface. However they 
  will shared many many features, so we shall see how to make this work. Ideally, shared stuff
  will be seperated into different classes that both NetRS and RS use. 


  The dimension of the problem and etc will be dispatched based on this dm, so a 2-D DMPlex will then solve a 2D 
  Riemann Problem. 

  Note: That while this can definetely be a clone of the DM used in the underlying simulation (and that is the purpose of
  setting this up), it need not be. An inteneded usage is to also have object used just to solve 1D Riemann problems one 
  at a time and nothing else, with the caller managing quadrature and etc when trying to solve higher dimensional problems. 

  I expect that will be the most used option. In order to support that option, you can either feed RiemannSolver a dm directly, 
  probably cloned from your application dm, or you can manually set on up internally by using RiemannSolverSetDim(dim)... , 
  which will internally build a dim dimensional plex dm, to solve that basic RiemannProblem, but will error out of if a 
  dm has already been provided or if one is subsequenlty provided. 
  
   */


  /* to be reworked to use d,*/
  PetscInt       numfields;
  PetscInt       dim;   /* dimension of the domain of the flux function ASSUMED 1 FOR NOW!!! */
  PetscReal      *flux_wrk; /* A work array holding the output flux of evaluation, numfield*dim entries*/ 
  PetscReal      *eig_wrk;  /* A work array holding the output eigenvalues of DF, numfield entries (FOR NOW ASSUMING 1D) */
  PetscReal      maxspeed; /* max wave speed computed */
  Mat            mat,eigen;/* Matrix describing the eigen decomposition of DF at a given state point */ 
  Mat            Df;       /* Matrix storing the flux jacobian at a point u */ 
  SNES           snes;
  KSP            ksp,eigenksp,dfksp;
  PetscPointFlux fluxfun;
  PetscPointFluxDer fluxderfun; 
  PetscPointFluxEig fluxeigfun; 
  PetscBool      fluxfunconvex; /* Is the flux function convex. This affects how maximum wave speeds can be computed */
  
  RiemannSolverEigBasis computeeigbasis;
  Vec                   u,ueig;  

  /* Lax Curve Information (move this to flux function) */ 
  LaxCurve evallaxcurve; 


  /* Not a huge fan of how these work but ehh.... */
  /* Roe matrix structures. Not always needed so may refactor to be an optional additional struct. That
  is a have a seperate roe matrix struct and the riemann solver only contains a pointer to one if roe 
  matrices are enabled by the solver/user. */
  RiemannSolverRoeMatrix computeroemat; 
  Mat            roemat, roeeigen;
  KSP            roeksp;
  Vec            u_roebasis; 
  Vec            roevec_conservative, roevec_characteristic; /* Vectors for usage with roe solvers */
  RiemannSolverRoeAvg roeavg;
};
#endif