#include <petscdmbf.h>
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "bf_2d_topology.h"
#include "bf_3d_topology.h"
#include "bf_2d_cells.h"
#include "bf_3d_cells.h"

#if defined(PETSC_HAVE_P4EST)

//TODO deprecated
#include "petsc_p4est_package.h"
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#include <p4est_vtk.h>
#include <p4est_plex.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>

/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/

typedef struct _p_DM_BF {
  /* forest-of-tree objects sequence: topology -> cells -> nodes */
  void       *ftTopology;
  void       *ftCells;
  void       *ftNodes;
  /* DMBF cells */
  DM_BF_Cell *cells;
  PetscBool  ownedCellsSetUpCalled;
  PetscBool  ghostCellsSetUpCalled;
  /* [option] blocks within a cell */
  PetscInt   blockSize[3];
  /* [option] settings for cell data */
  PetscInt   *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt   nValsPerElemRead, nValsPerElemReadWrite;
  PetscInt   valsPerElemReadTotal, valsPerElemReadWriteTotal;
} DM_BF;

/******************************************************************************
 * PRIVATE FUNCTIONS WITHOUT ERROR CHECKING
 *****************************************************************************/

static inline DM_BF *_p_getBF(DM dm)
{
  return (DM_BF*) ((DM_Forest*) dm->data)->data;
}

#define _p_bytesAlign(a) (((a)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

static inline size_t _p_cellSizeOfInfo()
{
  return _p_bytesAlign(sizeof(DM_BF_Cell));
}

static inline size_t _p_cellSizeOfDataRead(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadTotal));
}

static inline size_t _p_cellSizeOfDataReadWrite(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadWriteTotal));
}

static inline size_t _p_cellSizeOfData(DM_BF *bf)
{
  return _p_cellSizeOfDataRead(bf) + _p_cellSizeOfDataReadWrite(bf);
}

static inline size_t _p_cellSize(DM_BF *bf)
{
  return _p_cellSizeOfInfo() + _p_cellSizeOfData(bf);
}

static inline DM_BF_Cell *_p_cellGetPtrIndex(DM_BF *bf, PetscInt index)
{
  return (DM_BF_Cell*)(((char*)bf->cells) + _p_cellSize(bf) * ((size_t)index));
}

static inline DM_BF_Cell *_p_cellGetPtrQuadId(DM_BF *bf, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost)
{
  p4est_t        *p4est; //TODO deprecated
  DMBF_2D_CellsGetP4est((DM_BF_2D_Cells*)bf->ftCells,&p4est); //TODO deprecated

  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est->trees,treeid);

    return (DM_BF_Cell*)(((char*)bf->cells) + _p_cellSize(bf) * ((size_t)(tree->quadrants_offset + quadid)));
  } else {
    return (DM_BF_Cell*)(((char*)bf->cells) + _p_cellSize(bf) * ((size_t)(p4est->local_num_quadrants + quadid)));
  }
}

static inline PetscScalar *_p_cellGetDataRead(DM_BF_Cell *cell)
{
  return (PetscScalar*)(((char*)cell) + _p_cellSizeOfInfo());
}

static inline PetscScalar *_p_cellGetDataReadWrite(DM_BF_Cell *cell, DM_BF *bf)
{
  return (PetscScalar*)(((char*)cell) + _p_cellSizeOfInfo() + _p_cellSizeOfDataRead(bf));
}

/******************************************************************************
 * PRIVATE FUNCTION DEFINITIONS
 *****************************************************************************/

static PetscErrorCode DMBFSetUpOwnedCells(DM);
static PetscErrorCode DMBFSetUpGhostCells(DM);
static PetscErrorCode DMForestDestroy_BF(DM);
static PetscErrorCode DMClone_BF(DM,DM*);
static PetscErrorCode DMCreateLocalVector_BF(DM,Vec*);
static PetscErrorCode DMCreateGlobalVector_BF(DM,Vec*);
static PetscErrorCode DMCreateMatrix_BF(DM,Mat*);
static PetscErrorCode DMCoarsen_BF(DM,MPI_Comm,DM*);
static PetscErrorCode DMRefine_BF(DM,MPI_Comm,DM*);

/******************************************************************************
 * PRIVATE & PUBLIC FUNCTIONS
 *****************************************************************************/

/***************************************
 * SETUP
 **************************************/

static void _p_cellGetInfo(/*IN */ p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost,
                           /*OUT*/ DM_BF_Cell *cell)
{
  const p4est_qcoord_t qlength = P4EST_QUADRANT_LEN(quad->level);
  double               vertex1[3], vertex2[3];

  /* get vertex coordinates of opposite corners */
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x,quad->y,vertex1);
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x+qlength,quad->y+qlength,vertex2);
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

static void _p_cellGetVecView(/*IN    */ const PetscScalar **vecViewRead, PetscInt nVecsRead,
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

static void p4est_iter_set_cell_info(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF      *bf   = ctx;
  DM_BF_Cell *cell = _p_cellGetPtrQuadId(bf,info->treeid,info->quadid,0);

  /* get cell info */
  _p_cellGetInfo(info->p4est,info->quad,info->treeid,info->quadid,0,cell);
  cell->dataRead      = (const PetscScalar*)_p_cellGetDataRead(cell);
  cell->dataReadWrite = _p_cellGetDataReadWrite(cell,bf);
  /* assign cell to forest quadrant */
  info->quad->p.user_data = cell;
}

static PetscErrorCode DMBFSetUpOwnedCells(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* run iterator */
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,bf,p4est_iter_set_cell_info,NULL,NULL));
  p4est->data_size = _p_cellSize(bf);
  bf->ownedCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFSetUpGhostCells(DM dm)
{
  DM_BF      *bf;
  DM_BF_Cell *cell;
  PetscInt   offset_cells, ng_cells, i;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* set data pointers of all ghost cells */
  offset_cells = (PetscInt)p4est->local_num_quadrants;
  ng_cells     = (PetscInt)ghost->ghosts.elem_count;
  for (i=offset_cells; i<(offset_cells+ng_cells); i++) {
    cell                = _p_cellGetPtrIndex(bf,i);
    cell->dataRead      = (const PetscScalar*)_p_cellGetDataRead(cell);
    cell->dataReadWrite = _p_cellGetDataReadWrite(cell,bf);
  }
  bf->ghostCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsCreate(DM dm)
{
  DM_BF          *bf;
  size_t         n_cells, ng_cells, cell_size;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  /* get number of cells and their size */
  bf = _p_getBF(dm);
  n_cells   = (size_t)p4est->local_num_quadrants;
  ng_cells  = (size_t)ghost->ghosts.elem_count;
  cell_size = _p_cellSize(bf);
  /* create DMBF cells */
  ierr = PetscMalloc((n_cells+ng_cells)*cell_size,&bf->cells);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)dm,(n_cells+ng_cells)*cell_size);CHKERRQ(ierr);
  /* setup cells */
  ierr = DMBFSetUpOwnedCells(dm);CHKERRQ(ierr);
  ierr = DMBFSetUpGhostCells(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsDestroy(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bf = _p_getBF(dm);
  if (!bf->cells) {
    PetscFunctionReturn(0);
  }
  ierr = PetscFree(bf->cells);CHKERRQ(ierr);
  bf->cells = PETSC_NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  bf   = _p_getBF(dm);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set before setup");
  if (dim < 2 || 3 < dim)     SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DM does not support %d dimensional domains",dim);
  if (bf->ftTopology)         SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topology exists already");
  if (bf->ftCells)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells exist already");
  if (bf->ftNodes)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Nodes exist already");
  /* create forest-of-tree topology */
  if (2 == dim) {
    ierr = DMBF_2D_TopologyCreate(dm,(DM_BF_2D_Topology**)&bf->ftTopology);CHKERRQ(ierr);
  } else if (3 == dim) {
    ierr = DMBF_3D_TopologyCreate(dm,(DM_BF_3D_Topology**)&bf->ftTopology);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  if (!bf->ftTopology) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topology does not exist");
  /* create forest-of-tree cells */
  if (2 == dim) {
    ierr = DMBF_2D_CellsCreate(dm,(DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Cells**)&bf->ftCells);CHKERRQ(ierr);
  } else if (3 == dim) {
    ierr = DMBF_3D_CellsCreate(dm,(DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Cells**)&bf->ftCells);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* create DMBF cells */
  ierr = DMBF_CellsCreate(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  bf   = _p_getBF(dm);
  /* destroy forest-of-tree objects (in reverse order of creation) */
  if (2 == dim) {
    //TODO destroy nodes
    if (bf->ftCells)    { ierr = DMBF_2D_CellsDestroy(dm,(DM_BF_2D_Cells*)bf->ftCells);CHKERRQ(ierr); }
    if (bf->ftTopology) { ierr = DMBF_2D_TopologyDestroy(dm,(DM_BF_2D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
  } else if (3 == dim) {
    //TODO destroy nodes
    if (bf->ftCells)    { ierr = DMBF_3D_CellsDestroy(dm,(DM_BF_3D_Cells*)bf->ftCells);CHKERRQ(ierr); }
    if (bf->ftTopology) { ierr = DMBF_3D_TopologyDestroy(dm,(DM_BF_3D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unreachable code");
  }
  bf->ftNodes    = PETSC_NULL;
  bf->ftCells    = PETSC_NULL;
  bf->ftTopology = PETSC_NULL;
  /* destroy DMBF cells */
  ierr = DMBF_CellsDestroy(dm);CHKERRQ(ierr);
  /* destroy cell options */
  if (!bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);CHKERRQ(ierr);
  }
  if (!bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  PetscFunctionReturn(0);
}

/***************************************
 * OPTIONS
 **************************************/

/*@
  DMBFSetBlockSize - During the pre-setup phase, set the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object
- blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: DMBFGetBlockSize(), DMGetDimension()
@*/
PetscErrorCode DMBFSetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF          *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(blockSize,2);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the block refinement after setup");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set block refinement before topological dimension");
  bf = _p_getBF(dm);
  for (i=0; i<dim; i++) {
    bf->blockSize[i] = (1 <= blockSize[i] ? blockSize[i] : 1);
  }
  PetscFunctionReturn(0);
}

/*@
  DMBFGetBlockSize - Get the levels of uniform block refinement of each cell in each dimension.

  Logically collective on dm

  Input Parameters:
+ dm        - the DMBF object
- blockSize - levels of uniform block refinement of each cell in each dimension

  Level: intermediate

.seealso: DMBFSetBlockSize(), DMGetDimension()
@*/
PetscErrorCode DMBFGetBlockSize(DM dm, PetscInt *blockSize)
{
  DM_BF          *bf;
  PetscInt       dim, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(blockSize,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set for block refinement");
  bf = _p_getBF(dm);
  for (i=0; i<dim; i++) {
    blockSize[i] = bf->blockSize[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFSetCellDataSize(DM dm, PetscInt *valsPerElemRead, PetscInt nValsPerElemRead, PetscInt *valsPerElemReadWrite, PetscInt nValsPerElemReadWrite)
{
  DM_BF          *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(valsPerElemRead,2);
  PetscValidIntPointer(valsPerElemReadWrite,4);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change cell data after setup");
  bf = _p_getBF(dm);
  /* reset exising settings */
  if (!bf->valsPerElemRead) {
    ierr = PetscFree(bf->valsPerElemRead);CHKERRQ(ierr);
  }
  if (!bf->valsPerElemReadWrite) {
    ierr = PetscFree(bf->valsPerElemReadWrite);CHKERRQ(ierr);
  }
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;
  /* set new settings */
  if (0 < nValsPerElemRead) {
    bf->nValsPerElemRead = nValsPerElemRead;
    ierr = PetscMalloc1(bf->nValsPerElemRead,&bf->valsPerElemRead);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemRead; i++) {
      bf->valsPerElemRead[i]    = valsPerElemRead[i];
      bf->valsPerElemReadTotal += valsPerElemRead[i];
    }
  }
  if (0 < nValsPerElemReadWrite) {
    bf->nValsPerElemReadWrite = nValsPerElemReadWrite;
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite,&bf->valsPerElemReadWrite);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      bf->valsPerElemReadWrite[i]    = valsPerElemReadWrite[i];
      bf->valsPerElemReadWriteTotal += valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetCellDataSize(DM dm, PetscInt **valsPerElemRead, PetscInt *nValsPerElemRead, PetscInt **valsPerElemReadWrite, PetscInt *nValsPerElemReadWrite)
{
  DM_BF          *bf;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (0 < bf->nValsPerElemRead) {
    ierr = PetscMalloc1(bf->nValsPerElemRead,valsPerElemRead);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemRead; i++) {
      (*valsPerElemRead)[i] = bf->valsPerElemRead[i];
    }
  }
  if (0 < bf->nValsPerElemReadWrite) {
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite,valsPerElemReadWrite);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      (*valsPerElemReadWrite)[i] = bf->valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_BF(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscInt          blockSize[3], nBlockDim=3;
  PetscBool         set;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMSetFromOptions_Forest(PetscOptionsObject,dm);CHKERRQ(ierr);
  /* block_size */
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray(
      "-dm_bf_block_size","set uniform refinement inside each cell in each dimension x,y,z","DMBFSetBlockSize",
      blockSize,&nBlockDim,&set);CHKERRQ(ierr);
  if (set) {
    //TODO if (nBlockDim != dim)
    ierr = DMBFSetBlockSize(dm,blockSize);CHKERRQ(ierr);
  }
//TODO
//char              stringBuffer[256];
//ierr = PetscOptionsHead(PetscOptionsObject,"DM" P4EST_STRING " options");CHKERRQ(ierr);
//ierr = PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL);CHKERRQ(ierr);
//ierr = PetscOptionsString("-dm_p4est_ghost_label_name","the name of the ghost label when converting from a DMPlex",NULL,NULL,stringBuffer,sizeof(stringBuffer),&flg);CHKERRQ(ierr);
//ierr = PetscOptionsTail();CHKERRQ(ierr);
//if (flg) {
//  ierr = PetscFree(pforest->ghostName);CHKERRQ(ierr);
//  ierr = PetscStrallocpy(stringBuffer,&pforest->ghostName);CHKERRQ(ierr);
//}
  PetscFunctionReturn(0);
}

/***************************************
 * CREATE/DESTROY
 **************************************/

static PetscErrorCode DMInitialize_BF(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup              = DMSetUp_BF;
  dm->ops->setfromoptions     = DMSetFromOptions_BF;
  dm->ops->clone              = DMClone_BF;
  dm->ops->createlocalvector  = DMCreateLocalVector_BF;
  dm->ops->createglobalvector = DMCreateGlobalVector_BF;
  dm->ops->creatematrix       = DMCreateMatrix_BF;
  dm->ops->coarsen            = DMCoarsen_BF;
  dm->ops->refine             = DMRefine_BF;
  //TODO
  //dm->ops->createsubdm    = DMCreateSubDM_Forest;
  //dm->ops->adaptlabel     = DMAdaptLabel_Forest;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreate_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  ierr = DMInitialize_BF(dm);CHKERRQ(ierr);

  /* set default parameters of Forest object */
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,P4EST_QMAXLEVEL);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  /* create BF */
  ierr = PetscNewLog(dm,&bf);CHKERRQ(ierr);
  bf->ftTopology                = PETSC_NULL;
  bf->ftCells                   = PETSC_NULL;
  bf->ftNodes                   = PETSC_NULL;
  bf->cells                     = PETSC_NULL;
  bf->ownedCellsSetUpCalled     = PETSC_FALSE;
  bf->ghostCellsSetUpCalled     = PETSC_FALSE;
  bf->blockSize[0]              = 1;
  bf->blockSize[1]              = 1;
  bf->blockSize[2]              = 1;
  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;

  /* set data & functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest*) dm->data;

    forest->data    = bf;
    forest->destroy = DMForestDestroy_BF;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMForestDestroy_BF(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy contents of BF */
  ierr = DMBFClear(dm);CHKERRQ(ierr);
  /* destroy BF object */
  ierr = PetscFree(((DM_Forest*)dm->data)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_BF(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone_Forest(dm,newdm);CHKERRQ(ierr);
  ierr = DMInitialize_BF(*newdm);CHKERRQ(ierr);
  //TODO this is likely incomplete
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetP4est(DM dm, void *p4est)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMBF_2D_CellsGetP4est((DM_BF_2D_Cells*)bf->ftCells,p4est);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetGhost(DM dm, void *ghost)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMBF_2D_CellsGetGhost((DM_BF_2D_Cells*)bf->ftCells,ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_BF(DM dm, Vec *vec)
{
  DM_BF          *bf;
  PetscInt       n, ng;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  bf = _p_getBF(dm);
  n  = (PetscInt)p4est->local_num_quadrants;
  ng = (PetscInt)ghost->ghosts.elem_count;
  /* create vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n+ng,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_BF(DM dm, Vec *vec)
{
  DM_BF          *bf;
  PetscInt       n, N;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  bf = _p_getBF(dm);
  n  = (PetscInt)p4est->local_num_quadrants;
  N  = (PetscInt)p4est->global_num_quadrants;
  /* create vector */
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),n,N,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  //TODO
  //ierr = VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void))VecView_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  //ierr = VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void))VecLoad_pforest_Native);CHKERRQ(ierr);
  //ierr = VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_BF(DM dm, Mat *mat)
{
  DM_BF          *bf;
  void           *appctx;
  PetscInt       n, N;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(mat,2);
  /* get number of rows/cols */
  bf = _p_getBF(dm);
  n  = (PetscInt)p4est->local_num_quadrants;
  N  = (PetscInt)p4est->global_num_quadrants;
  /* create matrix */
  ierr = DMGetApplicationContext(dm,&appctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),n,n,N,N,appctx,mat);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  //TODO set null space?
  PetscFunctionReturn(0);
}

#if 0
typedef struct _p_DM_BF_VecGhostData {
  PetscScalar value;
} DM_BF_VecGhostData;

static PetscErrorCode DMGlobalToLocalBegin_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF              *bf;
  const PetscScalar  *glodata;
  PetscScalar        *locdata;
  DM_BF_VecGhostData *ghostdata;
  PetscInt           n, ng, i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,2);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,4);
  /* get number of entries */
  bf = _p_getBF(dm);
  if (!bf->ghost) {
    ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  }
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  ng = (PetscInt)bf->ghost->ghosts.elem_count;
  /* */
//{
//  p4est_tree_t     *tree;
//  p4est_quadrant_t *quad;
//  sc_array_t       *tquadrants;
//  p4est_topidx_t   ti;
//  size_t           tqi;

//  bf->p4est->data_size      = sizeof(DM_BF_VecGhostData);
//  bf->p4est->user_data_pool = sc_mempool_new(sizeof(DM_BF_VecGhostData));
//  for (ti=bf->p4est->first_local_tree; ti<=bf->p4est->last_local_tree; ++ti) {
//    tree       = p4est_tree_array_index(p4est->trees,ti);
//    tquadrants = &tree->quadrants;
//    for (tqi=0; tqi<tquadrants->elem_count; ++tqi) {
//      quad = p4est_quadrant_array_index(tquadrants,tqi);
//      quad->p.user_data = sc_mempool_alloc(bf->p4est->user_data_pool);
//    }
//  }

//  sc_mempool_destroy(bf->p4est->user_data_pool);
//  bf->p4est->data_size      = p4est_data_size;
//  bf->p4est->user_data_pool = p4est_data_pool;
//  ierr = PetscMalloc1(ng,&ghostdata);CHKERRQ(ierr);
//  ierr = PetscFree(ghostdata);CHKERRQ(ierr);
//}
  /* copy local entries */
  ierr = VecGetArrayRead(glo,&glodata);CHKERRQ(ierr);
  switch (mode) {
    case INSERT_VALUES:
      ierr = VecGetArrayWrite(loc,&locdata);CHKERRQ(ierr);
      ierr = PetscArraycpy(locdata,glodata,n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(loc,&locdata);CHKERRQ(ierr);
      break;
    case ADD_VALUES:
      ierr = VecGetArray(loc,&locdata);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        locdata[i] += glodata[i];
      }
      ierr = VecRestoreArray(loc,&locdata);CHKERRQ(ierr);
      break;
    default:
       SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Insert mode %d is not supported",mode);
  }
  ierr = VecRestoreArrayRead(glo,&glodata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/***************************************
 * AMR
 **************************************/

typedef struct _p_DM_BF_AmrCtx {
  PetscInt  minLevel;
  PetscInt  maxLevel;
} DM_BF_AmrCtx;

static int p4est_coarsen_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  DM_BF_AmrCtx   *amrCtx = p4est->user_pointer;
  const PetscInt minLevel = amrCtx->minLevel;
  const PetscInt l = quadrants[0]->level;

  return (minLevel < l);
}

static int p4est_refine_uniformly(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  DM_BF_AmrCtx   *amrCtx = p4est->user_pointer;
  const PetscInt maxLevel = amrCtx->maxLevel;
  const PetscInt l = quadrant->level;

  return (l < maxLevel);
}

/*@
  DMBFCoarsenInPlace - Coarsens the mesh uniformly.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nCycles - number of coarsening cycles

  Level: intermediate

.seealso: DMBFRefineInPlace()
@*/
PetscErrorCode DMBFCoarsenInPlace(DM dm, PetscInt nCycles)
{
  DM_BF_AmrCtx   amrCtx;
  DM_BF          *bf;
  void           *p4est_user_pointer;
  PetscInt       i;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMinimumRefinement(dm,&amrCtx.minLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer  = p4est->user_pointer;
  p4est->user_pointer = (void*) &amrCtx;
  /* coarsen & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_coarsen,(p4est,0,p4est_coarsen_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  p4est->user_pointer = p4est_user_pointer;
  //TODO need to update bf cells
  PetscFunctionReturn(0);
}

/*@
  DMBFRefineInPlace - Refines the mesh uniformly.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nCycles - number of refinement cycles

  Level: intermediate

.seealso: DMBFCoarsenInPlace()
@*/
PetscErrorCode DMBFRefineInPlace(DM dm, PetscInt nCycles)
{
  DM_BF_AmrCtx   amrCtx;
  DM_BF          *bf;
  void           *p4est_user_pointer;
  PetscInt       i;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMaximumRefinement(dm,&amrCtx.maxLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer  = p4est->user_pointer;
  p4est->user_pointer = (void*) &amrCtx;
  /* refine & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_refine,(p4est,0,p4est_refine_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  p4est->user_pointer = p4est_user_pointer;
  //TODO need to update bf cells
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsen_BF(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* coarsen input DM */
  if (!dmc) {
    ierr = DMBFCoarsenInPlace(dm,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* coarsen output DM */
  if (dm != *dmc) { /* if new fine DM */
    //TODO need to clone
  }
  ierr = DMBFCoarsenInPlace(*dmc,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_BF(DM dm, MPI_Comm comm, DM *dmf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* refine input DM */
  if (!dmf) {
    ierr = DMBFRefineInPlace(dm,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* refine output DM */
  if (dm != *dmf) { /* if new fine DM */
    //TODO need to clone
  }
  ierr = DMBFRefineInPlace(*dmf,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * ITERATORS
 **************************************/

typedef struct _p_DM_BF_CellIterCtx {
  DM_BF             *bf;
  PetscErrorCode    (*iterCell)(DM_BF_Cell*,void*);
  void              *userIterCtx;
  const PetscScalar **vecViewRead, **cellVecViewRead;
  PetscScalar       **vecViewReadWrite, **cellVecViewReadWrite;
  PetscInt          nVecsRead, nVecsReadWrite;
} DM_BF_CellIterCtx;

static void p4est_iter_volume(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_CellIterCtx *iterCtx = ctx;
  DM_BF             *bf      = iterCtx->bf;
  DM_BF_Cell        *cell    = _p_cellGetPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscErrorCode    ierr;

  /* assign vector view to cell */
  cell->vecViewRead      = iterCtx->cellVecViewRead;
  cell->vecViewReadWrite = iterCtx->cellVecViewReadWrite;
  /* get vector view */
  _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
  /* call cell function */
  ierr = iterCtx->iterCell(cell,iterCtx->userIterCtx);CHKERRV(ierr);
  /* remove vector view from cell */
  cell->vecViewRead      = PETSC_NULL;
  cell->vecViewReadWrite = PETSC_NULL;
}

PetscErrorCode DMBFIterateOverCellsVectors(DM dm, PetscErrorCode (*iterCell)(DM_BF_Cell*,void*), void *userIterCtx,
                                           Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF             *bf;
  DM_BF_CellIterCtx iterCtx;
  PetscInt          i;
  PetscErrorCode    ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterCell,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  /* set iterator context */
  iterCtx.bf             = bf;
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
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,p4est_iter_volume,NULL,NULL));
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

PetscErrorCode DMBFIterateOverCells(DM dm, PetscErrorCode (*iterCell)(DM_BF_Cell*,void*), void *userIterCtx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFIterateOverCellsVectors(dm,iterCell,userIterCtx,PETSC_NULL,0,PETSC_NULL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
typedef struct _p_DM_BF_FaceIterCtx {
  DM_BF             *bf;
  PetscErrorCode    (*iterFace)(DM_BF_Face*,void*);
  void              *userIterCtx;
  const PetscScalar **vecViewRead;
  PetscScalar       **vecViewReadWrite;
  PetscInt          nVecsRead, nVecsReadWrite;
  DM_BF_Face        face;
//DM_BF_Cell        cell[3]; //TODO only for 2D
} DM_BF_FaceIterCtx;

static void p4est_iter_face(p4est_iter_face_info_t *info, void *ctx)
{
  p4est_t              *p4est     = info->p4est;
  DM_BF_FaceIterCtx    *iterCtx   = ctx;
  DM_BF_Face           *face      = &iterCtx->face;
  DM_BF_Cell           *cell      = iterCtx->cell;
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

    _p_cellGetInfo(p4est,side->is.full.quad,side->treeid,side->is.full.quadid,0,cell);
    _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
    face->nCellsL = 1;
    face->nCellsR = 0;
    face->cellL[0] = cell;
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
        _p_cellGetInfo(p4est,sideL->is.hanging.quad[i],sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i],cell);
        _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
        face->cellL[i] = cell;
        cell++;
      }
    } else {
      _p_cellGetInfo(p4est,sideL->is.full.quad,sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost,cell);
      _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
      face->cellL[0] = cell;
      cell++;
    }
    if (sideR->is_hanging) {
      for (i=0; i<face->nCellsR; i++) {
        _p_cellGetInfo(p4est,sideR->is.hanging.quad[i],sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i],cell);
        _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
        face->cellR[i] = cell;
        cell++;
      }
    } else {
      _p_cellGetInfo(p4est,sideR->is.full.quad,sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost,cell);
      _p_cellGetVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
      face->cellR[0] = cell;
      cell++;
    }
  }

  /* call face function */
  ierr = iterCtx->iterFace(face,iterCtx->userIterCtx);CHKERRV(ierr);

  /* set vector data */
//cell = iterCtx->cell;
//for (i=0; i<(face->nCellsL + face->nCellsR); i++) {
//  _p_set_vec_data(cell,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite);
//  cell++;
//}
}
PetscErrorCode DMBFIterateOverFacesVectors(DM dm, PetscErrorCode (*iterFace)(DM_BF_Face*,void*), void *userIterCtx,
                                           Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF             *bf;
  DM_BF_FaceIterCtx iterCtx;
  PetscInt          i;
  PetscErrorCode    ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  /* set iterator context */
  iterCtx.iterFace       = iterFace;
  iterCtx.userIterCtx    = userIterCtx;
  iterCtx.nVecsRead      = nVecsRead;
  iterCtx.nVecsReadWrite = nVecsReadWrite;
  if (0 < nVecsRead) {
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscMalloc1(nVecsRead,&iterCtx.cell[i].vecViewRead);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nVecsRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<nVecsRead; i++) {
      ierr = VecGetArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  if (0 < nVecsReadWrite) {
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscMalloc1(nVecsReadWrite,&iterCtx.cell[i].vecViewReadWrite);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nVecsReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<nVecsReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  bf = _p_getBF(dm);
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,NULL,p4est_iter_face,NULL));
  /* clear iterator context */
  if (0 < nVecsRead) {
    for (i=0; i<nVecsRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscFree(iterCtx.cell[i].vecViewRead);CHKERRQ(ierr);
    }
  }
  if (0 < nVecsReadWrite) {
    for (i=0; i<nVecsReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscFree(iterCtx.cell[i].vecViewReadWrite);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM_BF_Face*,void*), void *userIterCtx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFIterateOverFacesVectors(dm,iterFace,userIterCtx,PETSC_NULL,0,PETSC_NULL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
typedef struct _p_DM_BF_FaceIterCtx {
  DM_BF             *bf;
  PetscErrorCode    (*iterFace)(DM_BF_Face*,void*);
  void              *userIterCtx;
  DM_BF_Face        face;
} DM_BF_FaceIterCtx;

static void p4est_iter_face(p4est_iter_face_info_t *info, void *ctx)
{
  DM_BF_FaceIterCtx    *iterCtx   = ctx;
  DM_BF                *bf        = iterCtx->bf;
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
    face->cellL[0] = _p_cellGetPtrQuadId(bf,side->treeid,side->is.full.quadid,0);
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
        face->cellL[i] = _p_cellGetPtrQuadId(bf,sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellL[0] = _p_cellGetPtrQuadId(bf,sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost);
    }
    if (sideR->is_hanging) {
      for (i=0; i<face->nCellsR; i++) {
        face->cellR[i] = _p_cellGetPtrQuadId(bf,sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellR[0] = _p_cellGetPtrQuadId(bf,sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost);
    }
  }
  /* call face function */
  ierr = iterCtx->iterFace(face,iterCtx->userIterCtx);CHKERRV(ierr);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM_BF_Face*,void*), void *userIterCtx)
{
  DM_BF             *bf;
  DM_BF_FaceIterCtx iterCtx;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (!bf->ghostCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost cells not set up");
  /* set iterator context */
  iterCtx.bf          = bf;
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  /* run iterator */
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,NULL,p4est_iter_face,NULL));
  PetscFunctionReturn(0);
}
#endif

/***************************************
 * CELL DATA
 **************************************/

typedef struct _p_DM_BF_SetCellDataIterCtx {
  DM_BF             *bf;
  const PetscScalar **vecViewRead, **vecViewReadWrite;
  PetscInt          nVecsRead, nVecsReadWrite;
} DM_BF_SetCellDataIterCtx;

static void p4est_iter_set_cell_data(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_SetCellDataIterCtx *iterCtx = ctx;
  DM_BF                    *bf      = iterCtx->bf;
  DM_BF_Cell               *cell    = _p_cellGetPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscScalar              *data;
  PetscInt                 i, j, di;

  /* set cell data for reading */
  if (0 < iterCtx->nVecsRead) {
    data = _p_cellGetDataRead(cell);
    di   = 0;
    for (i=0; i<bf->nValsPerElemRead; i++) {
      for (j=0; j<bf->valsPerElemRead[i]; j++) {
        if (bf->valsPerElemReadTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read index is above max");
        data[di] = iterCtx->vecViewRead[i][bf->valsPerElemRead[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
  /* set cell data for reading & writing */
  if (0 < iterCtx->nVecsReadWrite) {
    data = _p_cellGetDataReadWrite(cell,bf);
    di   = 0;
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      for (j=0; j<bf->valsPerElemReadWrite[i]; j++) {
        if (bf->valsPerElemReadWriteTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read-write index is above max");
        data[di] = iterCtx->vecViewReadWrite[i][bf->valsPerElemReadWrite[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
}

PetscErrorCode DMBFSetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF                    *bf;
  DM_BF_SetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  /* set iterator context */
  iterCtx.nVecsRead      = (vecRead      ? bf->nValsPerElemRead      : 0);
  iterCtx.nVecsReadWrite = (vecReadWrite ? bf->nValsPerElemReadWrite : 0);
  iterCtx.bf             = bf;
  if (0 < iterCtx.nVecsRead) {
    ierr = PetscMalloc1(iterCtx.nVecsRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecGetArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  if (0 < iterCtx.nVecsReadWrite) {
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecGetArrayRead(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,p4est_iter_set_cell_data,NULL,NULL));
  /* clear iterator context */
  if (0 < iterCtx.nVecsRead) {
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecRestoreArrayRead(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
  }
  if (0 < iterCtx.nVecsReadWrite) {
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecRestoreArrayRead(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_GetCellDataIterCtx {
  DM_BF       *bf;
  PetscScalar **vecViewRead, **vecViewReadWrite;
  PetscInt    nVecsRead, nVecsReadWrite;
} DM_BF_GetCellDataIterCtx;

static void p4est_iter_get_cell_data(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_GetCellDataIterCtx *iterCtx = ctx;
  DM_BF                    *bf      = iterCtx->bf;
  DM_BF_Cell               *cell    = _p_cellGetPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscScalar              *data;
  PetscInt                 i, j, di=0;

  /* get cell data for reading */
  if (0 < iterCtx->nVecsRead) {
    data = _p_cellGetDataRead(cell);
    di   = 0;
    for (i=0; i<bf->nValsPerElemRead; i++) {
      for (j=0; j<bf->valsPerElemRead[i]; j++) {
        if (bf->valsPerElemReadTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read index is above max");
        iterCtx->vecViewRead[i][bf->valsPerElemRead[i]*cell->indexLocal+j] = data[di];
        di++;
      }
    }
  }
  /* get cell data for reading & writing */
  if (0 < iterCtx->nVecsReadWrite) {
    data = _p_cellGetDataReadWrite(cell,bf);
    di   = 0;
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      for (j=0; j<bf->valsPerElemReadWrite[i]; j++) {
        if (bf->valsPerElemReadWriteTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read-write index is above max");
        iterCtx->vecViewReadWrite[i][bf->valsPerElemReadWrite[i]*cell->indexLocal+j] = data[di];
        di++;
      }
    }
  }
}

PetscErrorCode DMBFGetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF                    *bf;
  DM_BF_GetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  /* set iterator context */
  iterCtx.nVecsRead      = (vecRead      ? bf->nValsPerElemRead      : 0);
  iterCtx.nVecsReadWrite = (vecReadWrite ? bf->nValsPerElemReadWrite : 0);
  iterCtx.bf             = bf;
  if (0 < iterCtx.nVecsRead) {
    ierr = PetscMalloc1(iterCtx.nVecsRead,&iterCtx.vecViewRead);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecGetArray(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
  }
  if (0 < iterCtx.nVecsReadWrite) {
    ierr = PetscMalloc1(iterCtx.nVecsReadWrite,&iterCtx.vecViewReadWrite);CHKERRQ(ierr);
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecGetArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  PetscStackCallP4est(p4est_iterate,(p4est,ghost,&iterCtx,p4est_iter_get_cell_data,NULL,NULL));
  /* clear iterator context */
  if (0 < iterCtx.nVecsRead) {
    for (i=0; i<iterCtx.nVecsRead; i++) {
      ierr = VecRestoreArray(vecRead[i],&iterCtx.vecViewRead[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewRead);CHKERRQ(ierr);
  }
  if (0 < iterCtx.nVecsReadWrite) {
    for (i=0; i<iterCtx.nVecsReadWrite; i++) {
      ierr = VecRestoreArray(vecReadWrite[i],&iterCtx.vecViewReadWrite[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecViewReadWrite);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFCommunicateGhostCells(DM dm)
{
  DM_BF          *bf;
  DM_BF_Cell     *ghostCells;
  PetscErrorCode ierr;
  p4est_t        *p4est; //TODO deprecated
  p4est_ghost_t  *ghost; //TODO deprecated

  PetscFunctionBegin;
  ierr = DMBFGetP4est(dm,&p4est);CHKERRQ(ierr); //TODO deprecated
  ierr = DMBFGetGhost(dm,&ghost);CHKERRQ(ierr); //TODO deprecated
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  /* run ghost exchange */
  ghostCells = _p_cellGetPtrIndex(bf,p4est->local_num_quadrants);
  PetscStackCallP4est(p4est_ghost_exchange_data,(p4est,ghost,ghostCells));
  bf->ghostCellsSetUpCalled = PETSC_FALSE;
  ierr = DMBFSetUpGhostCells(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
