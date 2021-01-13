#if defined(PETSC_HAVE_P4EST)

#include "bf_xd.h"
#if !defined(P4_TO_P8)
#include "bf_2d_amr.h"
#else
#include "bf_3d_amr.h"
#endif

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
        0,          /* minimum number of quadrants per processor */
        initLevel,  /* level of refinement */
        1,          /* uniform refinement */
        0,          /* quadrant data size */
        NULL,       /* quadrant init function */
        (void*)dm ) /* this DM is the user context */
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_P4estDestroy(DM dm, p4est_t *p4est)
{
  PetscFunctionBegin;
  p4est->data_size = 0; /* avoid that p4est destroys quadrant data */
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
  ierr = DMBF_XD_GhostCreate((*cells)->p4est,&(*cells)->ghost);CHKERRQ(ierr);
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

PetscErrorCode DMBF_XD_CellsClone(DM_BF_XD_Cells *srcCells, DM_BF_XD_Cells **trgCells, DM trgDm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(trgDm,trgCells);CHKERRQ(ierr);
  PetscStackCallP4estReturn((*trgCells)->p4est,p4est_copy_ext,(srcCells->p4est,0/*copy data*/,1/*duplicate mpicomm*/));
  ierr = DMBF_XD_GhostCreate((*trgCells)->p4est,&(*trgCells)->ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_CellsCoarsen(DM_BF_XD_Cells *srcCells, DM_BF_XD_Cells **trgCells, DM trgDm, PetscInt minLevel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(trgDm,trgCells);CHKERRQ(ierr);
  PetscStackCallP4estReturn((*trgCells)->p4est,p4est_copy_ext,(srcCells->p4est,0/*copy data*/,1/*duplicate mpicomm*/));
  ierr = DMBF_XD_AmrCoarsenUniformly((*trgCells)->p4est,minLevel);CHKERRQ(ierr);
  ierr = DMBF_XD_GhostCreate((*trgCells)->p4est,&(*trgCells)->ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_CellsRefine(DM_BF_XD_Cells *srcCells, DM_BF_XD_Cells **trgCells, DM trgDm, PetscInt maxLevel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(trgDm,trgCells);CHKERRQ(ierr);
  PetscStackCallP4estReturn((*trgCells)->p4est,p4est_copy_ext,(srcCells->p4est,0/*copy data*/,1/*duplicate mpicomm*/));
  ierr = DMBF_XD_AmrRefineUniformly((*trgCells)->p4est,maxLevel);CHKERRQ(ierr);
  ierr = DMBF_XD_GhostCreate((*trgCells)->p4est,&(*trgCells)->ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_GetSizes(DM dm, DM_BF_XD_Cells *cells, PetscInt *nLocal, PetscInt *nGlobal, PetscInt *nGhost)
{
  PetscFunctionBegin;
  PetscValidIntPointer(nLocal,3);
  PetscValidIntPointer(nGlobal,4);
  PetscValidIntPointer(nGhost,5);
  *nLocal  = (PetscInt)(cells->p4est->local_num_quadrants);
  *nGlobal = (PetscInt)(cells->p4est->global_num_quadrants);
  if (cells->ghost) *nGhost = (PetscInt)(cells->ghost->ghosts.elem_count);
  else              *nGhost = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_GetLocalToGlobalIndices(DM dm, DM_BF_XD_Cells *cells, PetscInt *fromIdx, PetscInt *toIdx)
{
  p4est_t          *p4est = cells->p4est;
  p4est_ghost_t    *ghost = cells->ghost;
  p4est_locidx_t    n, ng, lid, i;
  p4est_gloidx_t    offset, gid;
  p4est_quadrant_t *quad;
  p4est_topidx_t    t;
  int               rank;

  PetscFunctionBegin;
  PetscValidIntPointer(fromIdx,3);
  PetscValidIntPointer(toIdx,4);
  /* get sizes */
  n      = p4est->local_num_quadrants;
  ng     = ghost->ghosts.elem_count;
  offset = p4est->global_first_quadrant[p4est->mpirank];
  /* set indices of owned cells */
  for(i = 0; i < n; i++) {
    fromIdx[i] = (PetscInt)i;
    toIdx[i]   = (PetscInt)(offset + i);
  }
  /* set indices of ghost cells */
  for(i = 0; i < ng; i++) {
    quad = sc_array_index(&ghost->ghosts,i);           /* get ghost quadrant i */
    t    = quad->p.piggy3.which_tree;                  /* get tree # of ghost quadrant i */
    rank = p4est_quadrant_find_owner(p4est,t,-1,quad); /* get mpirank of ghost quadrant i */
    lid  = quad->p.piggy3.local_num;                   /* get local id of ghost quadrant i on mpirank rank */
    gid  = p4est->global_first_quadrant[rank] + lid;   /* translate local id to global id */
    fromIdx[n + i] = (PetscInt)(n + i);
    toIdx[n + i]   = (PetscInt)gid;
  }
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
