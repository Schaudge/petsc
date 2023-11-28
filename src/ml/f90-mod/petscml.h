!
!  Used by petscmlmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscml.h"

      type tPetscRegressor
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscRegressor
      type tML
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tML

      ML, parameter :: PETSC_NULL_ML = tML(0)
      PetscRegressor, parameter :: PETSC_NULL_PETSCREGRESSOR = tPetscRegressor(0)

