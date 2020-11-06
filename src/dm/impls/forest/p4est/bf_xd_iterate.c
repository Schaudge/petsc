#include <petscdmbf.h> /*I "petscdmbf.h" I*/
#include <petscdm.h>   /*I "petscdm.h" I*/

#if defined(PETSC_HAVE_P4EST)

#include "bf_xd.h"

static inline DM_BF_Cell *_p_getCellPtr(const char *cells, const size_t cellSize,
                                        p4est_t *p4est, const p4est_topidx_t treeid, const p4est_locidx_t quadid, const int8_t is_ghost)
{
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees,treeid);

    return (DM_BF_Cell*)(cells + cellSize * ((size_t)(tree->quadrants_offset + quadid)));
  } else {
    return (DM_BF_Cell*)(cells + cellSize * ((size_t)(p4est->local_num_quadrants + quadid)));
  }
}

static void _p_getVecView(/*IN    */ const PetscScalar **vecViewRead, PetscInt nVecsRead,
                                     PetscScalar **vecViewReadWrite, PetscInt nVecsReadWrite,
                          /*IN/OUT*/ DM_BF_Cell *cell)
{
  PetscInt i;

  for (i=0; i<nVecsRead; i++) {
    cell->vecViewRead[i] = &vecViewRead[i][cell->indexLocal];
  }
  for (i=0; i<nVecsReadWrite; i++) {
    cell->vecViewReadWrite[i] = &vecViewReadWrite[i][cell->indexLocal];
  }
}

/***************************************
 * CELL ITERATORS
 **************************************/

typedef struct _p_DM_BF_CellIterCtx {
  /* DM-specifc info (required) */
  DM                dm;
  char              *cells;
  size_t            cellSize;
  /* iterator-specific info */
  PetscErrorCode    (*iterCell)(DM,DM_BF_Cell*,void*);
  void              *userIterCtx;
  const PetscScalar **vecViewRead, **cellVecViewRead;
  PetscScalar       **vecViewReadWrite, **cellVecViewReadWrite;
  PetscInt          nVecsRead, nVecsReadWrite;
} DM_BF_CellIterCtx;

static void _p_iterVolume(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_CellIterCtx *iterCtx = ctx;
  DM_BF_Cell        *cell    = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,info->treeid,info->quadid,0/*!ghost*/);
  PetscErrorCode    ierr;

  /* assign vector view to cell */
  cell->vecViewRead      = iterCtx->cellVecViewRead;
  cell->vecViewReadWrite = iterCtx->cellVecViewReadWrite;
  /* get vector view */
  _p_getVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
  /* call cell function */
  ierr = iterCtx->iterCell(iterCtx->dm,cell,iterCtx->userIterCtx);CHKERRV(ierr);
  /* remove vector view from cell */
  cell->vecViewRead      = PETSC_NULL;
  cell->vecViewReadWrite = PETSC_NULL;
}

PetscErrorCode DMBF_XD_IterateOverCellsVectors(DM dm, char *cells, size_t cellSize,
                                               PetscErrorCode (*iterCell)(DM,DM_BF_Cell*,void*), void *userIterCtx,
                                               Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF_CellIterCtx iterCtx;
  PetscInt          i;
  PetscErrorCode    ierr;
  p4est_t           *p4est;
  p4est_ghost_t     *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterCell,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  /* set iterator context */
  iterCtx.dm             = dm;
  iterCtx.cells          = cells;
  iterCtx.cellSize       = cellSize;
  iterCtx.iterCell       = iterCell;
  iterCtx.userIterCtx    = userIterCtx;
  iterCtx.nVecsRead      = nVecsRead;
  iterCtx.nVecsReadWrite = nVecsReadWrite;
  if (0 < iterCtx.nVecsRead) {
    ierr = PetscMalloc1(iterCtx.nVecsRead,&iterCtx.cellVecViewRead);CHKERRQ(ierr);
    ierr = PetscMalloc1(iterCtx.nVecsRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecGetArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  if (0 < iterCtx.nVecsReadWrite) {
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite,&iterCtx.cellVecViewReadWrite);CHKERRQ(ierr);
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterVolume,NULL,NULL));
  /* clear iterator context */
  if (0 < iterCtx.nVecsRead) {
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellVecViewRead);CHKERRQ(ierr);
  }
  if (0 < iterCtx.nVecsReadWrite) {
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellVecViewReadWrite);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/***************************************
 * FACE ITERATORS
 **************************************/

typedef struct _p_DM_BF_FaceIterCtx {
  /* DM-specifc info (required) */
  DM                dm;
  char              *cells;
  size_t            cellSize;
  /* iterator-specific info */
  PetscErrorCode    (*iterFace)(DM,DM_BF_Face*,void*);
  void              *userIterCtx;
  DM_BF_Face        face;
} DM_BF_FaceIterCtx;

static void _p_iterFace(p4est_iter_face_info_t *info, void *ctx)
{
  DM_BF_FaceIterCtx    *iterCtx   = ctx;
  DM_BF_Face           *face      = &iterCtx->face;
  const PetscBool      isBoundary = (1 == info->sides.elem_count);
  PetscInt             i;
  PetscErrorCode       ierr;

#if defined(PETSC_USE_DEBUG)
  face->cellL[0] = PETSC_NULL;
  face->cellL[1] = PETSC_NULL;
  face->cellL[2] = PETSC_NULL;
  face->cellL[3] = PETSC_NULL;
  face->cellR[0] = PETSC_NULL;
  face->cellR[1] = PETSC_NULL;
  face->cellR[2] = PETSC_NULL;
  face->cellR[3] = PETSC_NULL;
#endif

  /* get cell and vector data */
  if (isBoundary) {
    p4est_iter_face_side_t *side = p4est_iter_fside_array_index_int(&info->sides,0);

    face->nCellsL = 1;
    face->nCellsR = 0;
    face->cellL[0] = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,
                                   side->treeid,side->is.full.quadid,0/*!ghost*/);
  } else { /* !isBoundary */
    p4est_iter_face_side_t *sideL = p4est_iter_fside_array_index_int(&info->sides,0);
    p4est_iter_face_side_t *sideR = p4est_iter_fside_array_index_int(&info->sides,1);

    face->nCellsL = (sideL->is_hanging ? 2 : 1); //TODO only 2D
    face->nCellsR = (sideR->is_hanging ? 2 : 1); //TODO only 2D
    if ( !(1 <= face->nCellsL && 1 <= face->nCellsR && (face->nCellsL + face->nCellsR) <= 3) ) { //TODO only 2D
      //TODO error
    }
    if (sideL->is_hanging) {
      for (i=0; i<face->nCellsL; i++) {
        face->cellL[i] = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,
                                       sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellL[0] = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,
                                     sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost);
    }
    if (sideR->is_hanging) {
      for (i=0; i<face->nCellsR; i++) {
        face->cellR[i] = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,
                                       sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellR[0] = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,
                                     sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost);
    }
  }
  /* call face function */
  ierr = iterCtx->iterFace(iterCtx->dm,face,iterCtx->userIterCtx);CHKERRV(ierr);
}

PetscErrorCode DMBF_XD_IterateOverFaces(DM dm, char *cells, size_t cellSize,
                                        PetscErrorCode (*iterFace)(DM,DM_BF_Face*,void*), void *userIterCtx)
{
  DM_BF_FaceIterCtx iterCtx;
  PetscErrorCode    ierr;
  p4est_t           *p4est;
  p4est_ghost_t     *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  /* set iterator context */
  iterCtx.dm          = dm;
  iterCtx.cells       = cells;
  iterCtx.cellSize    = cellSize;
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  /* run iterator */
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,NULL,_p_iterFace,NULL));
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
