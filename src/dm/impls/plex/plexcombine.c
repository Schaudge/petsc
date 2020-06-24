#include <petsc/private/dmpleximpl.h>  /*I "petscdmplex.h" I*/

/* Logging Support */

PetscErrorCode DMPlexCombine(DM dma, DM dmb, DMLabel dmapts, DMLabel dmbpts, DM *dmc)
{
  PetscInt        an, bn;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dma, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmb, DM_CLASSID, 2);
  PetscValidHeaderSpecific(dmapts, DMLABEL_CLASSID, 3);
  PetscValidHeaderSpecific(dmbpts, DMLABEL_CLASSID, 4);
  PetscValidPointer(dmc, 5);
  ierr = DMLabelGetNumValues(dmapts, &an);CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(dmbpts, &bn);CHKERRQ(ierr);
  if (!an || !bn) PetscFunctionReturn(0);
  ierr = DMClone(dma, dmc);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dma, *dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
