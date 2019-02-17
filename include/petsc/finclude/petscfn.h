!
!
!  Include file for Fortran use of the Mat package in PETSc
!
#if !defined (__PETSCFNDEF_H)
#define __PETSCFNDEF_H

#include "petsc/finclude/petscmat.h"

#define PetscFn type(tPetscFn)

#define PetscFnType character*(80)
#define PetscFnOperation PetscEnum

!
!  PetscFn types
!
#define PETSCFNSHELL       'shell'
#define PETSCFNDAG         'dag'

#endif
