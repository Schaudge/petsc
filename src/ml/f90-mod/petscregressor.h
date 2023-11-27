!
!  Used by petscmlmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscregressor.h"

      type tPetscRegressor
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type
      PetscRegressor, parameter :: PETSC_NULL_PETSCREGRESSOR = tPetscRegressor(0)
