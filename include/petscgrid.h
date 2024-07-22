#ifdef PETSC_HAVE_GRID
#ifndef PETSCGRID_H
#define PETSCGRID_H
#endif
//#include <../../arch-darwin-c-complex-debug/include/Grid/Grid.h>

#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/

typedef enum {
  GRID_LATTICE_COLD,
  GRID_LATTICE_TEPID,
  GRID_LATTICE_HOT,
  GRID_LATTICE_FILE,
  GRID_LATTICE_NUM_TYPES,
} GRID_LOAD_TYPE;

PETSC_EXTERN PetscErrorCode PetscSetGauge_Grid(DM, PetscReal, GRID_LOAD_TYPE, int, char**, const char*);
PETSC_EXTERN PetscErrorCode PetscSetGauge_Grid5D(DM, GRID_LOAD_TYPE, PetscBool, int, char**, const char*);
PETSC_EXTERN PetscErrorCode PetscCheckDwfWithGrid(DM, Mat, Vec,Vec);
#endif