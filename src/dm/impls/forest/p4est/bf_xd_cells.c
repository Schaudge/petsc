#if defined(PETSC_HAVE_P4EST)

#include "bf_xd.h"

static PetscErrorCode DMBF_XD_P4estCreate(DM dm, p4est_connectivity_t *connectivity, p4est_t **p4est)
{
  PetscInt       initLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity does not exist");
  ierr = DMForestGetInitialRefinement(dm,&initLevel);CHKERRQ(ierr);
  PetscStackCallP4estReturn(
      *p4est,p4est_new_ext,
      ( PetscObjectComm((PetscObject)dm),
        connectivity,
        0,           /* minimum number of quadrants per processor */
        initLevel,   /* level of refinement */
        1,           /* uniform refinement */
        0,           /* we don't allocate any per quadrant data */
        NULL,        /* there is no special quadrant initialization */
        (void*)dm )  /* this DM is the user context */
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_P4estDestroy(DM dm, p4est_t *p4est)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_destroy,(p4est));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_GhostCreate(p4est_t *p4est, p4est_ghost_t **ghost)
{
  PetscFunctionBegin;
  PetscStackCallP4estReturn(*ghost,p4est_ghost_new,(p4est,P4EST_CONNECT_FULL));
  //TODO which connect flag, P4EST_CONNECT_FULL, P4EST_CONNECT_FACE, ...?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_GhostDestroy(p4est_ghost_t *ghost)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_ghost_destroy,(ghost));
  PetscFunctionReturn(0);
}

struct _p_DM_BF_XD_Cells {
  p4est_t       *p4est;
  p4est_ghost_t *ghost;
};

PetscErrorCode DMBF_XD_CellsCreate(DM dm, DM_BF_XD_Topology *topology, DM_BF_XD_Cells **cells)
{
  p4est_connectivity_t *connectivity;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(dm,cells);CHKERRQ(ierr);
  ierr = DMBF_XD_TopologyGetConnectivity(topology,&connectivity);CHKERRQ(ierr);
  ierr = DMBF_XD_P4estCreate(dm,connectivity,&(*cells)->p4est);CHKERRQ(ierr);
  (*cells)->ghost = PETSC_NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_CellsDestroy(DM dm, DM_BF_XD_Cells *cells)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cells->ghost) {
    ierr = DMBF_XD_GhostDestroy(cells->ghost);CHKERRQ(ierr);
  }
  ierr = DMBF_XD_P4estDestroy(dm,cells->p4est);CHKERRQ(ierr);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_CellsGetP4est(DM_BF_XD_Cells *cells, void *p4est)
{
  PetscFunctionBegin;
  *(void**)p4est = cells->p4est;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_CellsGetGhost(DM_BF_XD_Cells *cells, void *ghost)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cells->ghost) {
    ierr = DMBF_XD_GhostCreate(cells->p4est,&cells->ghost);CHKERRQ(ierr);
  }
  *(void**)ghost = cells->ghost;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
