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
    I think that riemann solvers need (or at least should have the option to) to switch between at minimum 
    conservative and characteristic variables. There should be an interface for this. 
  */
};

struct _p_RiemannSolver {
  PETSCHEADER(struct _PetscRiemannOps);

  PetscBool      setupcalled; 
  void           *data; /* implementation object */
  void           *user; /* user context */
  PetscInt       numfields;
  PetscInt       dim;   /* dimension of the domain of the flux function ASSUMED 1 FOR NOW!!! */
  PetscReal      *flux_wrk; /* A work array holding the output flux of evaluation, numfield*dim entries*/ 
  PetscReal      *eig_wrk;  /* A work array holding the output eigenvalues of F', numfield entries (FOR NOW ASSUMING 1D) */
  PetscReal      maxspeed; /* max wave speed computed */
  Mat            mat;
  SNES           snes;
  KSP            ksp;
  PetscPointFlux fluxfun;
  PetscPointFluxDer fluxderfun; 
  PetscPointFluxEig fluxeigfun; 
};
#endif