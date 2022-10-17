/* Creation functions for the physics used in the tests */


#if !defined(__DGPHYSICS_H)
#define __DGPHYSICS_H
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "dgnet.h"
#include <petsc/private/kernels/blockinvert.h>
#include <petscriemannsolver.h>
#include <petscnetrs.h>

/* 1D Shallow Water Equations (No Source) */

PETSC_EXTERN PetscErrorCode PhysicsCreate_Shallow(DGNetwork);

/* Single-variable traffic flow model f(u) = u(1-u) for the flux */
PETSC_EXTERN PetscErrorCode PhysicsCreate_Traffic(DGNetwork);

#endif