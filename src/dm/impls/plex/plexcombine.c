#include <petsc/private/dmpleximpl.h>  /*I "petscdmplex.h" I*/

/* Logging Support */

PetscErrorCode DMPlexCombine(DM dma, DM dmb, DMLabel dmapts, DMLabel dmbpts, DM *dmc)
{
  MPI_Comm              comma, commb;
  PetscMPIInt           commsame;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dma, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmb, DM_CLASSID, 2);
  PetscValidHeaderSpecific(dmapts, DMLABEL_CLASSID, 3);
  PetscValidHeaderSpecific(dmbpts, DMLABEL_CLASSID, 4);
  PetscValidPointer(dmc, 5);
  ierr = PetscObjectGetComm((PetscObject) dma, &comma);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dmb, &commb);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comma, commb, &commsame);CHKERRQ(ierr);
  if ((commsame != MPI_IDENT) && (commsame != MPI_CONGRUENT)) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMECOMM, "Both DM's must be on the same cummunicators");
  }
  ierr = DMPlexCreate(comma, dmc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
