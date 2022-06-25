#include <petsc/private/deviceimpl.h>

#if !PetscDefined(HAVE_CXX)
#define PetscManagedTypeAllocate   PetscNew
#define PetscManagedTypeDeallocate PetscFree

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Scalar
#define PetscTypeSuffix_L scalar
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Real
#define PetscTypeSuffix_L real
#include "managedtype.inl"

/* -------------------------------------------------------------------------------- */

#define PetscTypeSuffix   Int
#define PetscTypeSuffix_L int
#include "managedtype.inl"
#endif /* !PETSC_HAVE_CXX */
