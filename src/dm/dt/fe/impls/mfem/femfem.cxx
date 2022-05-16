
#include <petsc/private/petscfeimpl.h>

extern "C" {
  PETSC_INTERN PetscErrorCode PetscFECreate_MFEM(PetscFE);
}

#if defined(PETSC_HAVE_MFEM)

#include "mfem/fem/fe/fe_base.hpp"
#include <mfem.hpp>

using namespace mfem;


static PetscErrorCode PetscFECreateTabulation_MFEM(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  PetscFunctionBegin;
  PetscCall(PetscFESetUp(fem));
  FiniteElement *mfem = (FiniteElement *) fem->data;
  if (mfem == NULL) SETERRQ(PetscObjectComm((PetscObject)fem), PETSC_ERR_PLIB, "PetscFESetUp() did not create an mfem::FiniteElement");

  DM dm;
  PetscInt dim, pdim, Nc;
  PetscCall(PetscDualSpaceGetDM(fem->dualSpace, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscDualSpaceGetDimension(fem->dualSpace, &pdim));
  PetscCall(PetscFEGetNumComponents(fem, &Nc));

  /* Evaluate the prime basis functions at all points */
  PetscReal       *tmpB = NULL, *tmpD = NULL, *tmpH = NULL;
  if (K >= 0) PetscCall(DMGetWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &tmpB));
  if (K >= 1) PetscCall(DMGetWorkArray(dm, npoints*pdim*Nc*dim, MPIU_REAL, &tmpD));
  if (K >= 2) PetscCall(DMGetWorkArray(dm, npoints*pdim*Nc*dim*dim, MPIU_REAL, &tmpH));

  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEDestroy_MFEM(PetscFE fem)
{
  PetscFunctionBegin;
  if (fem->data != NULL) {
    FiniteElement *mfem = (FiniteElement *) fem->data;
    delete mfem;
    fem->data = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFESetUp_MFEM(PetscFE fem)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEInitialize_MFEM(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->destroy = PetscFEDestroy_MFEM;
  fem->ops->setup = PetscFESetUp_MFEM;
  fem->ops->createtabulation = PetscFECreateTabulation_MFEM;
  PetscFunctionReturn(0);
}
#endif // PETSC_HAVE_MFEM

/*MC
  PETSCFEMFEM = "mem" - A PetscFE object implemented by the MFEM library

M*/

PetscErrorCode PetscFECreate_MFEM(PetscFE fem)
{
  PetscFunctionBegin;
  fem->data = NULL;
  if (PetscDefined(PETSC_HAVE_MFEM)) {
    PetscCall(PetscFEInitialize_MFEM(fem));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)fem), PETSC_ERR_SUP, "PETSc was installed with MFEM support: configure with --download-mfem or --with-mfem=<mfem installation>");
  }
  PetscFunctionReturn(0);
}

