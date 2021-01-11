#include <petscdmbf.h> /*I "petscdmbf.h" I*/
#include <petscdm.h>   /*I "petscdm.h" I*/

#if defined(PETSC_HAVE_P4EST)

#include "bf_xd.h"

static inline DM_BF_Cell *_p_getCellPtr(const DM_BF_Cell *cells, const size_t cellSize,
                                        p4est_t *p4est, const p4est_topidx_t treeid, const p4est_locidx_t quadid, const int8_t is_ghost)
{
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees,treeid);

    return (DM_BF_Cell*)(((char*)cells) + cellSize * ((size_t)(tree->quadrants_offset + quadid)));
  } else {
    return (DM_BF_Cell*)(((char*)cells) + cellSize * ((size_t)(p4est->local_num_quadrants + quadid)));
  }
}

#define _p_getCellDataPtr(cell,size) (((char*)(cell)) + (size))

static void _p_getInfo(/*IN */ p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost,
                       /*OUT*/ DM_BF_Cell *cell)
{
  const p4est_qcoord_t qlength = P4EST_QUADRANT_LEN(quad->level);
  double               vertex1[3], vertex2[3];

  /* get vertex coordinates of opposite corners */
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x,quad->y,
#if defined(P4_TO_P8)
                         quad->z,
#endif
                         vertex1);
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x+qlength,quad->y+qlength,
#if defined(P4_TO_P8)
                         quad->z+qlength,
#endif
                         vertex2);
  /* set cell data */
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees,treeid);

    cell->indexLocal  = (PetscInt)(tree->quadrants_offset + quadid);
    cell->indexGlobal = cell->indexLocal + (PetscInt)p4est->global_first_quadrant[p4est->mpirank];
  } else {
    cell->indexLocal  = (PetscInt)(p4est->global_first_quadrant[p4est->mpirank+1] + quadid);
    cell->indexGlobal = -1;
  }
  cell->level         = (PetscInt)quad->level;
  cell->corner[0]     = (PetscReal)vertex1[0];
  cell->corner[1]     = (PetscReal)vertex1[1];
  cell->corner[2]     = (PetscReal)vertex1[2];
  //TODO set all 4/8 corners
  //TODO set volume
  cell->sidelength[0] = (PetscReal)(vertex2[0] - vertex1[0]);
  cell->sidelength[1] = (PetscReal)(vertex2[1] - vertex1[1]);
  cell->sidelength[2] = (PetscReal)(vertex2[2] - vertex1[2]);
  //TODO set side lengths to NAN if warped geometry
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
 * CELL SETUP
 **************************************/

typedef struct _p_DM_BF_SetUpCtx {
  /* DM-specifc info (required) */
  DM                dm;
  DM_BF_Cell        *cells;
  size_t            cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
} DM_BF_SetUpCtx;

static void _p_iterSetUp(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetUpCtx *iterCtx = ctx;
  DM_BF_Cell     *cell    = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,info->treeid,info->quadid,0/*!ghost*/);

  /* get cell info */
  _p_getInfo(info->p4est,info->quad,info->treeid,info->quadid,0,cell);
  cell->dataRead      = (const PetscScalar*)_p_getCellDataPtr(cell,iterCtx->cellOffsetDataRead);
  cell->dataReadWrite = (PetscScalar*)      _p_getCellDataPtr(cell,iterCtx->cellOffsetDataReadWrite);
  /* assign cell to forest quadrant */
  info->quad->p.user_data = cell;
}

PetscErrorCode DMBF_XD_IterateSetUpCells(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite)
{
  DM_BF_SetUpCtx iterCtx;
  PetscErrorCode ierr;
  p4est_t        *p4est;
  p4est_ghost_t  *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  /* run iterator */
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
#if defined(P4_TO_P8)
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterSetUp,NULL,NULL,NULL));
#else
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterSetUp,NULL,NULL));
#endif
  p4est->data_size = cellSize;
  PetscFunctionReturn(0);
}

/***************************************
 * CELL DATA
 **************************************/

typedef struct _p_DM_BF_SetCellDataIterCtx {
  /* DM-specifc info (required) */
  DM                dm;
  DM_BF_Cell        *cells;
  size_t            cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt    *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt          nValsPerElemRead, nValsPerElemReadWrite;
  const PetscScalar **vecViewRead, **vecViewReadWrite;
} DM_BF_SetCellDataIterCtx;

static void _p_iterSetCellData(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetCellDataIterCtx *iterCtx = ctx;
  DM_BF_Cell               *cell    = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,info->treeid,info->quadid,0/*!ghost*/);
  PetscScalar              *data;
  PetscInt                 i, j, di;

  /* set cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (PetscScalar*)_p_getCellDataPtr(cell,iterCtx->cellOffsetDataRead);
    di   = 0;
    for (i=0; i<iterCtx->nValsPerElemRead; i++) {
      for (j=0; j<iterCtx->valsPerElemRead[i]; j++) {
        data[di] = iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
  /* set cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (PetscScalar*)_p_getCellDataPtr(cell,iterCtx->cellOffsetDataReadWrite);
    di   = 0;
    for (i=0; i<iterCtx->nValsPerElemReadWrite; i++) {
      for (j=0; j<iterCtx->valsPerElemReadWrite[i]; j++) {
        data[di] = iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
}

PetscErrorCode DMBF_XD_IterateSetCellData(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite,
                                          const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead,
                                          const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite,
                                          Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF_SetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t                  *p4est;
  p4est_ghost_t            *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nValsPerElemRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<nValsPerElemRead; i++) {
      ierr = VecGetArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  else {
    iterCtx.vecViewRead = PETSC_NULL;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nValsPerElemReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<nValsPerElemReadWrite; i++) {
      ierr = VecGetArrayRead(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  else {
    iterCtx.vecViewReadWrite = PETSC_NULL;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
#if defined(P4_TO_P8)
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterSetCellData,NULL,NULL,NULL));
#else
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterSetCellData,NULL,NULL));
#endif
  /* clear iterator context */
  if (vecRead) {
    for (i=0; i<nValsPerElemRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i=0; i<nValsPerElemReadWrite; i++) {
      ierr = VecRestoreArrayRead(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_GetCellDataIterCtx {
  /* DM-specifc info (required) */
  DM                dm;
  DM_BF_Cell        *cells;
  size_t            cellSize, cellOffsetDataRead, cellOffsetDataReadWrite;
  /* iterator-specific info */
  const PetscInt    *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt          nValsPerElemRead, nValsPerElemReadWrite;
  PetscScalar       **vecViewRead, **vecViewReadWrite;
} DM_BF_GetCellDataIterCtx;

static void _p_iterGetCellDate(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_GetCellDataIterCtx *iterCtx = ctx;
  DM_BF_Cell               *cell    = _p_getCellPtr(iterCtx->cells,iterCtx->cellSize,info->p4est,info->treeid,info->quadid,0/*!ghost*/);
  const PetscScalar        *data;
  PetscInt                 i, j, di;

  /* get cell data for reading */
  if (iterCtx->vecViewRead) {
    data = (const PetscScalar*)_p_getCellDataPtr(cell,iterCtx->cellOffsetDataRead);
    di   = 0;
    for (i=0; i<iterCtx->nValsPerElemRead; i++) {
      for (j=0; j<iterCtx->valsPerElemRead[i]; j++) {
        iterCtx->vecViewRead[i][iterCtx->valsPerElemRead[i]*cell->indexLocal+j] = data[di];
        di++;
      }
    }
  }
  /* get cell data for reading & writing */
  if (iterCtx->vecViewReadWrite) {
    data = (const PetscScalar*)_p_getCellDataPtr(cell,iterCtx->cellOffsetDataReadWrite);
    di   = 0;
    for (i=0; i<iterCtx->nValsPerElemReadWrite; i++) {
      for (j=0; j<iterCtx->valsPerElemReadWrite[i]; j++) {
        iterCtx->vecViewReadWrite[i][iterCtx->valsPerElemReadWrite[i]*cell->indexLocal+j] = data[di];
        di++;
      }
    }
  }
}

PetscErrorCode DMBF_XD_IterateGetCellData(DM dm, DM_BF_Cell *cells, size_t cellSize, size_t cellOffsetDataRead, size_t cellOffsetDataReadWrite,
                                          const PetscInt *valsPerElemRead, PetscInt nValsPerElemRead,
                                          const PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite,
                                          Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF_GetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t                  *p4est;
  p4est_ghost_t            *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set iterator context */
  iterCtx.dm                      = dm;
  iterCtx.cells                   = cells;
  iterCtx.cellSize                = cellSize;
  iterCtx.cellOffsetDataRead      = cellOffsetDataRead;
  iterCtx.cellOffsetDataReadWrite = cellOffsetDataReadWrite;
  iterCtx.valsPerElemRead         = valsPerElemRead;
  iterCtx.nValsPerElemRead        = nValsPerElemRead;
  iterCtx.valsPerElemReadWrite    = valsPerElemReadWrite;
  iterCtx.nValsPerElemReadWrite   = nValsPerElemReadWrite;
  if (vecRead) {
    ierr = PetscMalloc1(nValsPerElemRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<nValsPerElemRead; i++) {
      ierr = VecGetArray(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  else {
    iterCtx.vecViewRead = PETSC_NULL;
  }
  if (vecReadWrite) {
    ierr = PetscMalloc1(nValsPerElemReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<nValsPerElemReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  else {
    iterCtx.vecViewReadWrite = PETSC_NULL;
  }
  /* run iterator */
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
#if defined(P4_TO_P8)
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterGetCellDate,NULL,NULL,NULL));
#else
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterGetCellDate,NULL,NULL));
#endif
  /* clear iterator context */
  if (vecRead) {
    for (i=0; i<nValsPerElemRead; i++) {
      ierr = VecRestoreArray(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
  }
  if (vecReadWrite) {
    for (i=0; i<nValsPerElemReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/***************************************
 * GHOST CELLS
 **************************************/

PetscErrorCode DMBF_XD_IterateGhostExchange(DM dm, DM_BF_Cell *cells, size_t cellSize)
{
  DM_BF_Cell     *ghostCells;
  PetscErrorCode ierr;
  p4est_t        *p4est;
  p4est_ghost_t  *ghost;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr);
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr);
  ghostCells = _p_getCellPtr(cells,cellSize,p4est,-1/*no tree id*/,0 /*quadid*/,1/*ghost*/);
  PetscStackCallP4est(p4est_ghost_exchange_data,(p4est,ghost,ghostCells));
  PetscFunctionReturn(0);
}

/***************************************
 * CELL ITERATORS
 **************************************/

typedef struct _p_DM_BF_CellIterCtx {
  /* DM-specifc info (required) */
  DM                dm;
  DM_BF_Cell        *cells;
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

PetscErrorCode DMBF_XD_IterateOverCellsVectors(DM dm, DM_BF_Cell *cells, size_t cellSize,
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
#if defined(P4_TO_P8)
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterVolume,NULL,NULL,NULL));
#else
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,_p_iterVolume,NULL,NULL));
#endif
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
  DM_BF_Cell        *cells;
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

PetscErrorCode DMBF_XD_IterateOverFaces(DM dm, DM_BF_Cell *cells, size_t cellSize,
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
#if defined(P4_TO_P8)
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,NULL,_p_iterFace,NULL,NULL));
#else
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,NULL,_p_iterFace,NULL));
#endif
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
