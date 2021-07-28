/* Experimental Riemann Solver Class. To all for functor riemann solvers (storing paramters for example, allowing functions
to generater riemann solver objects, abstract riemann solver selection, internal storage of solver objects
(i.e. linear and nonlinear solvers etc)) */

/* Implementation of the Riemann Objects */


/*
    TODO : Learn how to create petsc classes and refactor this as a petsc class 
*/

#include <petscriemann.h>
#include <petsc/private/petscimpl.h>
#include <petscmat.h>
#include <petscsnes.h>

typedef struct _PetscRiemannOps *PetscRiemannOps;
struct _PetscRiemannOps {
  PetscErrorCode (*setfromoptions)(PetscRiemann);
  PetscErrorCode (*setup)(PetscRiemann);
  PetscErrorCode (*view)(PetscRiemann,PetscViewer);
  PetscErrorCode (*destroy)(PetscRiemann);
  PetscErrorCode (*evaluate)(const PetscReal*, const PetscReal*, PetscReal*);
};

struct _p_PetscRiemann {
  PETSCHEADER(struct _PetscRiemannOps);
  void           *data; /* implementation object */
  PetscInt       numfields; 
  Mat            mat; 
  SNES           snes; 
  PetscPointFlux flux; 
};

