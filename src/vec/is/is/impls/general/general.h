#ifndef PETSC_IS_GENERAL_H
#define PETSC_IS_GENERAL_H

#include <petsc/private/isimpl.h>

/* Defines the data structure used for the general index set */

typedef struct {
  PetscBool sorted;    /* indicates the indices are sorted */
  PetscBool allocated; /* did we allocate the index array ourselves? */
  PetscInt *idx;
} IS_General;

#endif // PETSC_IS_GENERAL_H
