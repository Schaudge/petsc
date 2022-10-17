#if !defined(__FLUXFUN_H)
#define __FLUXFUN_H

#include <petscriemannsolver.h>

/* Prototype flux function class, for use in NETRS testing. To be refactored as proper class itself */

/* Contains the analytic formulas for flux function stuff */ 

typedef struct _p_FluxFunction* FluxFunction; 

struct  _p_FluxFunction{
  PetscErrorCode                 (*destroy)(FluxFunction*);
  void                           *user;
  PetscInt                       dof;
  char                           *fieldname[16];
  PetscPointFlux                 flux; 
  PetscPointFluxEig              fluxeig;    
  RiemannSolverRoeAvg            roeavg;
  RiemannSolverRoeMatrix         roemat; 
  RiemannSolverEigBasis          eigbasis; 
  PetscPointFluxDer              fluxder; 
  LaxCurve                       laxcurve; 
};


/* 1D Shallow Water Equations (No Source) */

PETSC_EXTERN PetscErrorCode PhysicsCreate_Shallow(FluxFunction*);


#endif