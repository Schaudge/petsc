#if defined(PETSC_HAVE_P4EST)

  #include "bf_xd.h"
  #if !defined(P4_TO_P8)
    #include "bf_2d_amr.h"
  #else
    #include "bf_3d_amr.h"
  #endif

/* default definitions, to be overwritten when this files is included */
  #if !defined(DM_BF_XD_Cells)
typedef struct _p_DM_BF_XD_Cells DM_BF_XD_Cells;
  #endif

static PetscErrorCode DMBF_XD_P4estCreate(DM dm, p4est_connectivity_t *connectivity, p4est_t **p4est)
{
  PetscInt       initLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheck(connectivity, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Connectivity does not exist");
  ierr = DMForestGetInitialRefinement(dm, &initLevel);
  CHKERRQ(ierr);
  PetscCallP4estReturn(*p4est, p4est_new_ext,
                       (PetscObjectComm((PetscObject)dm), connectivity, 0, /* minimum number of quadrants per processor */
                        initLevel,                                         /* level of refinement */
                        1,                                                 /* uniform refinement */
                        0,                                                 /* quadrant data size */
                        NULL,                                              /* quadrant init function */
                        (void *)dm)                                        /* this DM is the user context */
  );
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_P4estDestroy(DM dm, p4est_t *p4est)
{
  PetscFunctionBegin;
  p4est->data_size = 0; /* avoid that p4est destroys quadrant data */
  PetscCheck(!p4est->user_data_pool, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "p4est should not allocate user data memory");
  PetscCallP4est(p4est_destroy, (p4est));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_GhostCreate(p4est_t *p4est, p4est_ghost_t **ghost)
{
  PetscFunctionBegin;
  PetscCallP4estReturn(*ghost, p4est_ghost_new, (p4est, P4EST_CONNECT_FULL));
  //TODO which connect flag, P4EST_CONNECT_FULL, P4EST_CONNECT_FACE, ...?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_GhostDestroy(p4est_ghost_t *ghost)
{
  PetscFunctionBegin;
  PetscCallP4est(p4est_ghost_destroy, (ghost));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_P4estMeshCreate(p4est_t *p4est, p4est_ghost_t *ghost, p4est_mesh_t **mesh)
{
  PetscFunctionBegin;
  PetscCallP4estReturn(*mesh, p4est_mesh_new, (p4est, ghost, P4EST_CONNECT_FULL));
  //TODO which connect flag, P4EST_CONNECT_FULL, P4EST_CONNECT_FACE, ...?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_XD_P4estMeshDestroy(p4est_mesh_t *mesh)
{
  PetscFunctionBegin;
  PetscCallP4est(p4est_mesh_destroy, (mesh));
  PetscFunctionReturn(0);
}

struct _p_DM_BF_XD_Cells {
  p4est_t       *p4est;
  p4est_ghost_t *ghost;
  p4est_mesh_t  *mesh;
};

  #if !defined(DMBF_XD_CellsCreate)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsCreate(DM dm, DM_BF_XD_Topology *topology, DM_BF_XD_Cells **cells, PetscErrorCode (*setUpUserFnAfterP4est)(DM, void *))
{
  p4est_connectivity_t *connectivity;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscNew(cells);
  CHKERRQ(ierr);
  ierr = DMBF_XD_TopologyGetConnectivity(topology, &connectivity);
  CHKERRQ(ierr);
  ierr = DMBF_XD_P4estCreate(dm, connectivity, &(*cells)->p4est);
  CHKERRQ(ierr);
  if (setUpUserFnAfterP4est) {
    ierr = setUpUserFnAfterP4est(dm, (void *)(*cells)->p4est);
    CHKERRQ(ierr);
  }
  ierr = DMBF_XD_GhostCreate((*cells)->p4est, &(*cells)->ghost);
  CHKERRQ(ierr);
  //ierr = DMBF_XD_P4estMeshCreate((*cells)->p4est,(*cells)->ghost,&(*cells)->mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsDestroy)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsDestroy(DM dm, DM_BF_XD_Cells *cells)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cells->mesh) {
    ierr = DMBF_XD_P4estMeshDestroy(cells->mesh);
    CHKERRQ(ierr);
  }
  if (cells->ghost) {
    ierr = DMBF_XD_GhostDestroy(cells->ghost);
    CHKERRQ(ierr);
  }
  ierr = DMBF_XD_P4estDestroy(dm, cells->p4est);
  CHKERRQ(ierr);
  ierr = PetscFree(cells);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsClone)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsClone(/*IN    */ DM_BF_XD_Cells  *origCells,
                     /*OUT   */ DM_BF_XD_Cells **clonedCells,
                     /*IN/OUT*/ DM               clonedDm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(clonedCells);
  CHKERRQ(ierr);
  PetscCallP4estReturn((*clonedCells)->p4est, p4est_copy_ext, (origCells->p4est, 0 /*!copy data*/, 1 /*duplicate mpicomm*/));
  ierr = DMBF_XD_GhostCreate((*clonedCells)->p4est, &(*clonedCells)->ghost);
  CHKERRQ(ierr);
  //ierr = DMBF_XD_P4estMeshCreate((*clonedCells)->p4est,(*clonedCells)->ghost,&(*clonedCells)->mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsCoarsen)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsCoarsen(/*IN    */ DM_BF_XD_Cells  *origCells,
                       /*OUT   */ DM_BF_XD_Cells **coarseCells,
                       /*IN/OUT*/ DM               coarseDm,
                       /*IN    */ PetscInt         minLevel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(coarseCells);
  CHKERRQ(ierr);
  PetscCallP4estReturn((*coarseCells)->p4est, p4est_copy_ext, (origCells->p4est, 0 /*!copy data*/, 1 /*duplicate mpicomm*/));
  ierr = DMBF_XD_AmrCoarsenUniformly((*coarseCells)->p4est, minLevel);
  CHKERRQ(ierr);
  ierr = DMBF_XD_GhostCreate((*coarseCells)->p4est, &(*coarseCells)->ghost);
  CHKERRQ(ierr);
  //ierr = DMBF_XD_P4estMeshCreate((*coarseCells)->p4est,(*coarseCells)->ghost,&(*coarseCells)->mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsRefine)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsRefine(/*IN    */ DM_BF_XD_Cells  *origCells,
                      /*OUT   */ DM_BF_XD_Cells **fineCells,
                      /*IN/OUT*/ DM               fineDm,
                      /*IN    */ PetscInt         maxLevel)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(fineCells);
  CHKERRQ(ierr);
  PetscCallP4estReturn((*fineCells)->p4est, p4est_copy_ext, (origCells->p4est, 0 /*!copy data*/, 1 /*duplicate mpicomm*/));
  ierr = DMBF_XD_AmrRefineUniformly((*fineCells)->p4est, maxLevel);
  CHKERRQ(ierr);
  ierr = DMBF_XD_GhostCreate((*fineCells)->p4est, &(*fineCells)->ghost);
  CHKERRQ(ierr);
  //ierr = DMBF_XD_P4estMeshCreate((*fineCells)->p4est,(*fineCells)->ghost,&(*fineCells)->mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsAmrAdapt)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsAmrAdapt(/*IN    */ DM_BF_XD_Cells  *origCells,
                        /*OUT   */ DM_BF_XD_Cells **adapCells,
                        /*IN/OUT*/ DM               adapDm,
                        /*IN    */ DM_BF_AmrOps *amrOps, PetscInt minLevel, PetscInt maxLevel, const DM_BF_Shape *cellMemoryShape)
{
  p4est_t       *orig_p4est = origCells->p4est;
  p4est_t       *adap_p4est;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(adapCells);
  CHKERRQ(ierr);
  /* create copy of p4est */
  PetscCallP4estReturn((*adapCells)->p4est, p4est_copy_ext, (orig_p4est, 0 /*!copy data*/, 1 /*duplicate mpicomm*/));
  adap_p4est = (*adapCells)->p4est;
  /* adapt cells of p4est */
  ierr = DMBF_XD_AmrAdapt(adap_p4est, minLevel, maxLevel);
  CHKERRQ(ierr);
  /* create and setup cell data owned by p4est */
  PetscCallP4est(p4est_reset_data, (adap_p4est, cellMemoryShape->size, NULL /*init_fn*/, orig_p4est->user_pointer));
  ierr = DMBF_XD_IterateSetUpP4estCells(adapDm, cellMemoryShape);
  CHKERRQ(ierr);
  /* adapt cell data */
  ierr = DMBF_XD_AmrAdaptData(orig_p4est, adap_p4est, adapDm, amrOps);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsAmrPartition)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsAmrPartition(/*IN/OUT*/ DM_BF_XD_Cells *cells)
{
  PetscFunctionBegin;
  CHKERRQ(DMBF_XD_AmrPartition(cells->p4est));
  CHKERRQ(DMBF_XD_GhostCreate(cells->p4est, &cells->ghost));
  //CHKERRQ( DMBF_XD_P4estMeshCreate(cells->p4est,cells->ghost,&cells->mesh) );
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsAmrFinalize)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsAmrFinalize(/*IN/OUT*/ DM dm, DM_BF_XD_Cells *cells, DM_BF_Cell *bfCells,
                           /*IN    */ const DM_BF_Shape *cellMemoryShape)
{
  PetscFunctionBegin;
  CHKERRQ(DMBF_XD_IterateCopyP4estCells(dm, bfCells, cellMemoryShape));
  PetscCallP4est(p4est_reset_data, (cells->p4est, 0 /*data_size*/, NULL /*init_fn*/, cells->p4est->user_pointer));
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_GetSizes)
static
  #endif
  PetscErrorCode
  DMBF_XD_GetSizes(DM dm, DM_BF_XD_Cells *cells, PetscInt *nLocal, PetscInt *nGlobal, PetscInt *nGhost)
{
  PetscFunctionBegin;
  PetscAssertPointer(nLocal, 3);
  PetscAssertPointer(nGlobal, 4);
  PetscAssertPointer(nGhost, 5);
  *nLocal  = (PetscInt)(cells->p4est->local_num_quadrants);
  *nGlobal = (PetscInt)(cells->p4est->global_num_quadrants);
  if (cells->ghost) *nGhost = (PetscInt)(cells->ghost->ghosts.elem_count);
  else *nGhost = 0;
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_GetLocalToGlobalIndices)
static
  #endif
  PetscErrorCode
  DMBF_XD_GetLocalToGlobalIndices(DM dm, DM_BF_XD_Cells *cells, PetscInt *fromIdx, PetscInt *toIdx)
{
  p4est_t          *p4est = cells->p4est;
  p4est_ghost_t    *ghost = cells->ghost;
  p4est_locidx_t    n, ng, lid, i, j, k, l, idx;
  PetscInt          blockSize[3] = {1, 1, 1}, bs;
  p4est_gloidx_t    offset, gid;
  p4est_quadrant_t *quad;
  p4est_topidx_t    t;
  int               rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* get sizes */
  ierr = DMBFGetBlockSize(dm, blockSize);
  CHKERRQ(ierr);
  bs     = blockSize[0] * blockSize[1] * blockSize[2];
  n      = p4est->local_num_quadrants;
  ng     = ghost->ghosts.elem_count;
  offset = p4est->global_first_quadrant[p4est->mpirank];
  if (0 < (n + ng)) {
    PetscAssertPointer(fromIdx, 3);
    PetscAssertPointer(toIdx, 4);
  }
  /* set indices of owned cells */
  for (i = 0; i < n * bs; i++) {
    fromIdx[i] = (PetscInt)i;
    toIdx[i]   = (PetscInt)(offset * bs + i);
  }
  /* set indices of ghost cells */
  for (i = 0; i < ng; i++) {
    quad = sc_array_index(&ghost->ghosts, i);             /* get ghost quadrant i */
    t    = quad->p.piggy3.which_tree;                     /* get tree # of ghost quadrant i */
    rank = p4est_quadrant_find_owner(p4est, t, -1, quad); /* get mpirank of ghost quadrant i */
    lid  = quad->p.piggy3.local_num;                      /* get local id of ghost quadrant i on mpirank rank */
    gid  = p4est->global_first_quadrant[rank] + lid;      /* translate local id to global id */
    switch (P4EST_DIM) {
    case 2:
      for (k = 0; k < blockSize[1]; k++) {
        for (j = 0; j < blockSize[0]; j++) {
          idx          = (PetscInt)n * bs + bs * i + blockSize[0] * k + j;
          fromIdx[idx] = idx;
          toIdx[idx]   = (PetscInt)bs * gid + blockSize[0] * k + j;
        }
      }
      break;
    case 3:
      for (l = 0; l < blockSize[2]; l++) {
        for (k = 0; k < blockSize[1]; k++) {
          for (j = 0; j < blockSize[0]; j++) {
            idx          = (PetscInt)bs * n + bs * i + blockSize[0] * blockSize[1] * l + blockSize[0] * k + j;
            fromIdx[idx] = idx;
            toIdx[idx]   = (PetscInt)bs * gid + blockSize[0] * blockSize[1] * l + blockSize[0] * k + j;
          }
        }
      }
      break;
    }
  }
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsGetP4est)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsGetP4est(DM_BF_XD_Cells *cells, void *p4est)
{
  PetscFunctionBegin;
  *(void **)p4est = cells->p4est;
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsGetGhost)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsGetGhost(DM_BF_XD_Cells *cells, void *ghost)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cells->ghost) {
    ierr = DMBF_XD_GhostCreate(cells->p4est, &cells->ghost);
    CHKERRQ(ierr);
  }
  *(void **)ghost = cells->ghost;
  PetscFunctionReturn(0);
}

  #if !defined(DMBF_XD_CellsGetP4estMesh)
static
  #endif
  PetscErrorCode
  DMBF_XD_CellsGetP4estMesh(DM_BF_XD_Cells *cells, void *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!cells->mesh) {
    if (!cells->ghost) {
      ierr = DMBF_XD_GhostCreate(cells->p4est, &cells->ghost);
      CHKERRQ(ierr);
    }
    ierr = DMBF_XD_P4estMeshCreate(cells->p4est, cells->ghost, &cells->mesh);
    CHKERRQ(ierr);
  }
  *(void **)mesh = cells->mesh;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
