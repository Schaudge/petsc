#ifndef PETSCBT_H
#define PETSCBT_H

#include <petscbttype.h>
#include <petscviewer.h>

static inline PetscErrorCode PetscBTView(PetscInt m, const PetscBT bt, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (m < 1) PetscFunctionReturn(PETSC_SUCCESS);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_SELF, &viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  for (PetscInt i = 0; i < m; ++i) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT " %d\n", i, (int)PetscBTLookup(bt, i)));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* PETSCBT_H */
