
#include <petsc/private/petscfeimpl.h>
#include <mfem.hpp>

#if defined(PETSC_HAVE_MFEM)

extern "C" {
  PETSC_INTERN PetscErrorCode PetscFECreate_MFEM(PetscFE);
}


/*MC
  PETSCFEMFEM = "mem" - A PetscFE object implemented by the MFEM library

M*/

PetscErrorCode PetscFECreate_MFEM(PetscFE fem)
{
  PetscFunctionBegin;
  fem->data = NULL;
  PetscFunctionReturn(0);
}

#endif // PETSC_HAVE_MFEM
