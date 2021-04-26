#include <petscdmbf.h>                  /*I "petscdmbf.h" I*/
#include <petsc/private/dmbfimpl.h>
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "bf_2d_topology.h"
#include "bf_3d_topology.h"
#include "bf_2d_cells.h"
#include "bf_3d_cells.h"
#include "bf_2d_iterate.h"
#include "bf_3d_iterate.h"
#if defined(PETSC_HAVE_P4EST)
#include "petsc_p4est_package.h"
#endif

/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/

typedef struct _p_DM_BF {
  /* forest-of-tree objects sequence: topology -> cells -> nodes */
  void         *ftTopology;
  void         *ftCells;
  void         *ftNodes;
  /* DMBF cells */
  DM_BF_Cell   *cells;
  PetscBool    ownedCellsSetUpCalled;
  PetscBool    ghostCellsSetUpCalled;
  /* [option] blocks within a cell */
  PetscInt     blockSize[3];
  /* [option] settings for cell data */
  PetscInt     *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt     nValsPerElemRead, nValsPerElemReadWrite;
  PetscInt     valsPerElemReadTotal, valsPerElemReadWriteTotal;
  VecScatter   ltog; /* local to global scatter object, for parallel communication of data between local and global vectors */
  /* AMR callback functions */
  DM_BF_AmrOps *amrOps;
} DM_BF;

/******************************************************************************
 * PRIVATE FUNCTIONS WITHOUT ERROR CHECKING
 *****************************************************************************/

static inline DM_BF *_p_getBF(DM dm)
{
  return (DM_BF*) ((DM_Forest*) dm->data)->data;
}

static PetscInt _p_dim(DM dm)
{
  PetscInt dim;

  CHKERRQ( DMGetDimension(dm,&dim) );
  return dim;
}

#define _p_comm(dm) PetscObjectComm((PetscObject)(dm))

#define _p_SETERRQ_UNREACHABLE(dm) SETERRQ(PetscObjectComm((PetscObject)(dm)),PETSC_ERR_SUP,"Unreachable code")

/***************************************
 * CELL SIZES
 **************************************/

#define _p_bytesAlign(a) (((a)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

static inline size_t _p_cellSizeOfInfo()
{
  return _p_bytesAlign(sizeof(DM_BF_Cell));
}

#define _p_cellOffsetDataRead _p_cellSizeOfInfo

static inline size_t _p_cellSizeOfDataRead(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadTotal));
}

#define _p_cellOffsetDataReadWrite(bf) (_p_cellSizeOfInfo() + _p_cellSizeOfDataRead(bf))

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

/***************************************
 * CELL POINTERS
 **************************************/

static inline DM_BF_Cell *_p_cellGetPtrIndex(DM_BF *bf, PetscInt index)
{
  return (DM_BF_Cell*)(((char*)bf->cells) + _p_cellSize(bf) * ((size_t)index));
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

static PetscErrorCode DMForestDestroy_BF(DM);
static PetscErrorCode DMClone_BF(DM,DM*);

static PetscErrorCode DMCreateLocalVector_BF(DM,Vec*);
static PetscErrorCode DMCreateGlobalVector_BF(DM,Vec*);
static PetscErrorCode DMLocalToGlobalBegin_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMLocalToGlobalEnd_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMGlobalToLocalBegin_BF(DM,Vec,InsertMode,Vec);
static PetscErrorCode DMGlobalToLocalEnd_BF(DM,Vec,InsertMode,Vec);

static PetscErrorCode DMCreateMatrix_BF(DM,Mat*);

static PetscErrorCode DMCoarsen_BF(DM,MPI_Comm,DM*);
static PetscErrorCode DMRefine_BF(DM,MPI_Comm,DM*);
static PetscErrorCode DMView_BF(DM,PetscViewer);
static PetscErrorCode VecView_BF(Vec,PetscViewer);

/******************************************************************************
 * PRIVATE & PUBLIC FUNCTIONS
 *****************************************************************************/

/***************************************
 * CHECKING
 **************************************/

static PetscErrorCode DMBFCheck(DM dm)
{
  PetscBool      isCorrectDM;
  DM_BF          *bf;

  PetscFunctionBegin;
  /* check type of DM */
  CHKERRQ( PetscObjectTypeCompare((PetscObject)dm,DMBF,&isCorrectDM) );
  if (!isCorrectDM) SETERRQ2(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Type of DM is %s, but has to be %s",((PetscObject)dm)->type_name,DMBF);
  /* check cells */
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (!bf->ghostCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost cells not set up");
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#define DMBFCheckDebug(dm) DMBFCheck((dm))
#else
#define DMBFCheckDebug(dm) ((void) (0))
#endif

/***************************************
 * SETUP
 **************************************/

static PetscErrorCode DMBF_CellsCreate(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim, n, ng;
  size_t         n_cells, ng_cells, cell_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* get number of cells and their size */
  bf   = _p_getBF(dm);
  ierr = DMBFGetInfo(dm,&dim,&n,PETSC_NULL,&ng);CHKERRQ(ierr);
  n_cells   = (size_t)n;
  ng_cells  = (size_t)ng;
  cell_size = _p_cellSize(bf);
  /* create DMBF cells */
  ierr = PetscMalloc((n_cells+ng_cells)*cell_size,&bf->cells);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)dm,(n_cells+ng_cells)*cell_size);CHKERRQ(ierr);
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
  //ierr = PetscLogObjectMemory((PetscObject)dm,-(n_cells+ng_cells)*cell_size);CHKERRQ(ierr); //TODO need to "unlog" memeory?
  bf->cells = PETSC_NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsSetUpOwned(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  switch (_p_dim(dm)) {
    case 2: ierr = DMBF_2D_IterateSetUpCells(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateSetUpCells(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  bf->ownedCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsSetUpGhost(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim, offset_cells, ng_cells, i;
  DM_BF_Cell     *cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* set data pointers of all ghost cells */
  ierr = DMBFGetInfo(dm,&dim,&offset_cells,PETSC_NULL,&ng_cells);CHKERRQ(ierr);
  for (i=offset_cells; i<(offset_cells+ng_cells); i++) {
    cell                = _p_cellGetPtrIndex(bf,i);
    cell->dataRead      = (const PetscScalar*)_p_cellGetDataRead(cell);
    cell->dataReadWrite = _p_cellGetDataReadWrite(cell,bf);
  }
  bf->ghostCellsSetUpCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_LocalToGlobalScatterCreate(DM dm, VecScatter *ltog)
{
  DM_BF          *bf;
  PetscInt       dim, n, ng, locDof, bs, blockSize[3] = {1, 1, 1};
  PetscInt       *fromIdx, *toIdx;
  IS             fromIS, toIS;
  Vec            vloc, vglo;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* create local-to-global indices */
  bf = _p_getBF(dm);
  ierr = DMBFGetInfo(dm,&dim,&n,PETSC_NULL,&ng);CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  bs   = blockSize[0]*blockSize[1]*blockSize[2];
  locDof = (n + ng)*bs;
  ierr = PetscMalloc1(locDof,&fromIdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(locDof,&toIdx);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_GetLocalToGlobalIndices(dm,(DM_BF_2D_Cells*)bf->ftCells,fromIdx,toIdx);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_GetLocalToGlobalIndices(dm,(DM_BF_3D_Cells*)bf->ftCells,fromIdx,toIdx);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  /* create IS */
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,locDof,fromIdx,PETSC_COPY_VALUES,&fromIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,locDof,toIdx,PETSC_COPY_VALUES,&toIS);CHKERRQ(ierr);
  ierr = PetscFree(fromIdx);CHKERRQ(ierr);
  ierr = PetscFree(toIdx);CHKERRQ(ierr);
  /* create vectors */
  ierr = DMCreateLocalVector(dm,&vloc);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vglo);CHKERRQ(ierr);
  /* create vec scatter object */
  ierr = VecScatterCreate(vloc,fromIS,vglo,toIS,ltog);CHKERRQ(ierr);
  /* destroy */
  ierr = ISDestroy(&fromIS);CHKERRQ(ierr);
  ierr = ISDestroy(&toIS);CHKERRQ(ierr);
  ierr = VecDestroy(&vloc);CHKERRQ(ierr);
  ierr = VecDestroy(&vglo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_LocalToGlobalScatterDestroy(DM dm, VecScatter *ltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  ierr = VecScatterDestroy(ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set before setup");
  if (dim < 2 || 3 < dim)     SETERRQ1(_p_comm(dm),PETSC_ERR_SUP,"DM does not support %d dimensional domains",dim);
  if (bf->ftTopology)         SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Topology exists already");
  if (bf->ftCells)            SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells exist already");
  if (bf->ftNodes)            SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Nodes exist already");
  /* create forest-of-tree topology */
  switch (dim) {
    case 2: ierr = DMBF_2D_TopologyCreate(dm,(DM_BF_2D_Topology**)&bf->ftTopology);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_TopologyCreate(dm,(DM_BF_3D_Topology**)&bf->ftTopology);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  if (!bf->ftTopology) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Topology does not exist");
  /* create forest-of-tree cells */
  switch (dim) {
    case 2: ierr = DMBF_2D_CellsCreate(dm,(DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Cells**)&bf->ftCells);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_CellsCreate(dm,(DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Cells**)&bf->ftCells);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  if (!bf->ftCells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* create and setup DMBF cells */
  ierr = DMBF_CellsCreate(dm);CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpOwned(dm);CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(dm);CHKERRQ(ierr);
  /* create local-to-global vector scattering info */
  ierr = DMBF_LocalToGlobalScatterCreate(dm,&bf->ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf   = _p_getBF(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  /* destroy forest-of-tree objects (in reverse order of creation) */
  switch (dim) {
    case 2:
      //TODO destroy nodes
      if (bf->ftCells)    { ierr = DMBF_2D_CellsDestroy(dm,(DM_BF_2D_Cells*)bf->ftCells);CHKERRQ(ierr); }
      if (bf->ftTopology) { ierr = DMBF_2D_TopologyDestroy(dm,(DM_BF_2D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
      break;
    case 3:
      //TODO destroy nodes
      if (bf->ftCells)    { ierr = DMBF_3D_CellsDestroy(dm,(DM_BF_3D_Cells*)bf->ftCells);CHKERRQ(ierr); }
      if (bf->ftTopology) { ierr = DMBF_3D_TopologyDestroy(dm,(DM_BF_3D_Topology*)bf->ftTopology);CHKERRQ(ierr); }
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
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
  /* destroy local-to-global vector scattering info */
  ierr = DMBF_LocalToGlobalScatterDestroy(dm,&bf->ltog);CHKERRQ(ierr);
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

  Output Parameters:
+ blockSize - levels of uniform block refinement of each cell in each dimension

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
  if (dm->setupcalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the block refinement after setup");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == PETSC_DETERMINE) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set block refinement before topological dimension");
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
  if (dim == PETSC_DETERMINE) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Topological dimension has to be set for block refinement");
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
  if (dm->setupcalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change cell data after setup");
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
  /* PetscValidIntPointer(valsPerElemRead,3); */
  /* PetscValidIntPointer(valsPerElemReadWrite,5); */
  bf = _p_getBF(dm);
  if(nValsPerElemRead) { *nValsPerElemRead = bf->nValsPerElemRead; }
  if (0 < bf->nValsPerElemRead && valsPerElemRead) {
    ierr = PetscMalloc1(bf->nValsPerElemRead,valsPerElemRead);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemRead; i++) {
      (*valsPerElemRead)[i] = bf->valsPerElemRead[i];
    }
  }
  if(nValsPerElemReadWrite) { *nValsPerElemReadWrite = bf->nValsPerElemReadWrite; }
  if (0 < bf->nValsPerElemReadWrite && valsPerElemReadWrite) {
    ierr = PetscMalloc1(bf->nValsPerElemReadWrite,valsPerElemReadWrite);CHKERRQ(ierr);
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      (*valsPerElemReadWrite)[i] = bf->valsPerElemReadWrite[i];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFSetDefaultOptions(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);

  ierr = DMSetDimension(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,18);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  ierr = DMSetVecType(dm,VECSTANDARD);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,MATSHELL);CHKERRQ(ierr);

  bf = _p_getBF(dm);

  bf->ftTopology = PETSC_NULL;
  bf->ftCells    = PETSC_NULL;
  bf->ftNodes    = PETSC_NULL;

  bf->cells                 = PETSC_NULL;
  bf->ownedCellsSetUpCalled = PETSC_FALSE;
  bf->ghostCellsSetUpCalled = PETSC_FALSE;

  bf->blockSize[0] = 1;
  bf->blockSize[1] = 1;
  bf->blockSize[2] = 1;

  bf->valsPerElemRead           = PETSC_NULL;
  bf->valsPerElemReadWrite      = PETSC_NULL;
  bf->nValsPerElemRead          = 0;
  bf->nValsPerElemReadWrite     = 0;
  bf->valsPerElemReadTotal      = 0;
  bf->valsPerElemReadWriteTotal = 0;

  bf->amrOps = PETSC_NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFCopyOptions(DM srcdm, DM trgdm)
{
  DM_BF          *srcbf, *trgbf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(srcdm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecificType(trgdm,DM_CLASSID,2,DMBF);

  srcbf = _p_getBF(srcdm);
  trgbf = _p_getBF(trgdm);

  trgbf->ftTopology = PETSC_NULL;
  trgbf->ftCells    = PETSC_NULL;
  trgbf->ftNodes    = PETSC_NULL;

  trgbf->cells                 = PETSC_NULL;
  trgbf->ownedCellsSetUpCalled = PETSC_FALSE;
  trgbf->ghostCellsSetUpCalled = PETSC_FALSE;

  trgbf->blockSize[0] = srcbf->blockSize[0];
  trgbf->blockSize[1] = srcbf->blockSize[1];
  trgbf->blockSize[2] = srcbf->blockSize[2];

  if (0 < srcbf->nValsPerElemRead) {
    ierr = PetscMalloc1(srcbf->nValsPerElemRead,&trgbf->valsPerElemRead);CHKERRQ(ierr);
    ierr = PetscArraycpy(trgbf->valsPerElemRead,srcbf->valsPerElemRead,srcbf->nValsPerElemRead);CHKERRQ(ierr);
  }
  if (0 < srcbf->nValsPerElemReadWrite) {
    ierr = PetscMalloc1(srcbf->nValsPerElemReadWrite,&trgbf->valsPerElemReadWrite);CHKERRQ(ierr);
    ierr = PetscArraycpy(trgbf->valsPerElemReadWrite,srcbf->valsPerElemReadWrite,srcbf->nValsPerElemReadWrite);CHKERRQ(ierr);
  }
  trgbf->nValsPerElemRead          = srcbf->nValsPerElemRead;
  trgbf->nValsPerElemReadWrite     = srcbf->nValsPerElemReadWrite;
  trgbf->valsPerElemReadTotal      = srcbf->valsPerElemReadTotal;
  trgbf->valsPerElemReadWriteTotal = srcbf->valsPerElemReadWriteTotal;

  trgbf->amrOps = srcbf->amrOps;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_BF(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscInt          blockSize[3], nBlockDim=3;
  PetscBool         set;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,2,DMBF);
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
  dm->ops->view               = DMView_BF;

  dm->ops->createlocalvector  = DMCreateLocalVector_BF;
  dm->ops->createglobalvector = DMCreateGlobalVector_BF;
  dm->ops->creatematrix       = DMCreateMatrix_BF;

  dm->ops->coarsen            = DMCoarsen_BF;
  dm->ops->refine             = DMRefine_BF;

  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_BF;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_BF;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_BF;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_BF;
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreate_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  /* create Forest object */
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  /* create BF object */
  ierr = PetscNewLog(dm,&bf);CHKERRQ(ierr);
  /* set data and functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest*) dm->data;

    forest->data    = bf;
    forest->destroy = DMForestDestroy_BF;
  }
  /* set operators */
  ierr = DMInitialize_BF(dm);CHKERRQ(ierr);
  /* set default options */
  ierr = DMBFSetDefaultOptions(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMForestDestroy_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy contents of BF */
  ierr = DMBFClear(dm);CHKERRQ(ierr);
  /* destroy BF object */
  bf   = _p_getBF(dm);
  ierr = PetscFree(bf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFCloneInit(DM dm, DM *newdm)
{
  DM_BF          *bf, *newbf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* clone Forest object */
  ierr = DMForestTemplate(dm,_p_comm(dm),newdm);CHKERRQ(ierr);
  ierr = DMInitialize_BF(*newdm);CHKERRQ(ierr);
  /* create BF object */
  ierr = PetscNewLog(*newdm,&newbf);CHKERRQ(ierr);
  /* set data and functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest*) (*newdm)->data;

    forest->data    = newbf;
    forest->destroy = DMForestDestroy_BF;
  }
  /* copy operators */
  bf   = _p_getBF(dm);
  ierr = PetscMemcpy((*newdm)->ops,dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);
  /* copy options */
  ierr = DMBFCopyOptions(dm,*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFCloneForestOfTrees(DM dm, DM newdm)
{
  DM_BF          *bf, *newbf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bf    = _p_getBF(dm);
  newbf = _p_getBF(newdm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2:
      ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Topology**)&newbf->ftTopology,newdm);CHKERRQ(ierr);
      ierr = DMBF_2D_CellsClone((DM_BF_2D_Cells*)bf->ftCells,(DM_BF_2D_Cells**)&newbf->ftCells,newdm);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    case 3:
      ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Topology**)&newbf->ftTopology,newdm);CHKERRQ(ierr);
      ierr = DMBF_3D_CellsClone((DM_BF_3D_Cells*)bf->ftCells,(DM_BF_3D_Cells**)&newbf->ftCells,newdm);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFCloneFinalize(DM newdm)
{
  DM_BF          *newbf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create and setup DMBF cells */
  ierr = DMBF_CellsCreate(newdm);CHKERRQ(ierr); //TODO create a clone fnc instead?
  ierr = DMBF_CellsSetUpOwned(newdm);CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(newdm);CHKERRQ(ierr);
  /* create local-to-global vector scattering info */
  newbf = _p_getBF(newdm);
  ierr = DMBF_LocalToGlobalScatterCreate(newdm,&newbf->ltog);CHKERRQ(ierr); //TODO create a clone fnc instead?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_BF(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  ierr = DMBFCloneInit(dm,newdm);CHKERRQ(ierr);
  ierr = DMBFCloneForestOfTrees(dm,*newdm);CHKERRQ(ierr);
  ierr = DMBFCloneFinalize(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetP4est(DM dm, void *p4est)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_CellsGetP4est((DM_BF_2D_Cells*)bf->ftCells,p4est);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_CellsGetP4est((DM_BF_3D_Cells*)bf->ftCells,p4est);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetGhost(DM dm, void *ghost)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_CellsGetGhost((DM_BF_2D_Cells*)bf->ftCells,ghost);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_CellsGetGhost((DM_BF_3D_Cells*)bf->ftCells,ghost);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

//TODO need to use DMGetVecType and; add DMSetVecType -> STANDARD to be default for the DM
static PetscErrorCode DMCreateLocalVector_BF(DM dm, Vec *vec)
{
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, ng;
  PetscInt       locDof, cellDof = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm,&dim,&n,PETSC_NULL,&ng);CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  for(PetscInt i = 0; i < dim; i++) {
    cellDof *= blockSize[i];
  }
  locDof    = cellDof*(n + ng);
  /* create vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,locDof,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO need to use DMGetVecType and; add DMSetVecType -> STANDARD to be default for the DM
static PetscErrorCode DMCreateGlobalVector_BF(DM dm, Vec *vec)
{
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, N;
  PetscInt       locDof, gloDof, cellDof = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  ierr = DMBFGetInfo(dm,&dim,&n,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  for(PetscInt i = 0; i < dim; i++) {
    cellDof *= blockSize[i];
  }
  locDof = cellDof*n;
  gloDof = cellDof*N;
  /* create vector */
  ierr = VecCreateMPI(_p_comm(dm),locDof,gloDof,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_VIEW,(void (*)(void))VecView_BF);CHKERRQ(ierr);
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
  PetscInt       blockSize[3] = {1, 1, 1};
  PetscInt       dim, n, N;
  PetscInt       cellDof = 1;
  PetscInt       locDof,gloDof;
  MatType        mattype;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(mat,2);
  /* get number of rows/cols */
  ierr = DMBFGetInfo(dm,&dim,&n,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
  for(PetscInt i = 0; i < dim; i++) {
    cellDof *= blockSize[i];
  }
  locDof = cellDof*n;
  gloDof = cellDof*N;
  /* create matrix */
  ierr = MatCreate(_p_comm(dm),mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,locDof,locDof,gloDof,gloDof);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*mat,1/*blocksize*/);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  /* set type */
  ierr = DMGetMatType(dm,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(*mat,mattype);CHKERRQ(ierr);
  /* set context */
  PetscStrcmp(mattype,MATSHELL,&match);
  if (match) {
    void *appctx;
    ierr = DMGetApplicationContext(dm,&appctx);CHKERRQ(ierr);
    ierr = MatShellSetContext(*mat,appctx);CHKERRQ(ierr);
    ierr = MatSetUp(*mat);CHKERRQ(ierr);
  }
  //TODO set null space?
  PetscFunctionReturn(0);
}

/* take global vector and return local version */
static PetscErrorCode DMGlobalToLocalBegin_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,2);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog,glo,loc,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalEnd_BF(DM dm, Vec glo, InsertMode mode, Vec loc)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,2);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog,glo,loc,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode DMLocalToGlobalBegin_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,2);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterBegin(bf->ltog,loc,glo,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalEnd_BF(DM dm, Vec loc, InsertMode mode, Vec glo)
{
  DM_BF              *bf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidHeaderSpecific(loc,VEC_CLASSID,2);
  PetscValidHeaderSpecific(glo,VEC_CLASSID,4);

  bf = _p_getBF(dm);
  ierr = VecScatterEnd(bf->ltog,loc,glo,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * MESH
 **************************************/

/*@
  DMBFGetInfo - Gets information about the DM.

  Not Collective

  Input Parameter:
+ dm      - the DMBF object

  Output Parameters:
+ dim     - spatial dimension (2 or 3)
. nLocal  - number of local cells
. nGlobal - number of global cells
. nGhost  - number of ghost cells

  Level: beginner

.seealso: DMGetDimension(), DMBFGetLocalSize(), DMBFGetGlobalSize(), DMBFGetGhostSize()
@*/
PetscErrorCode DMBFGetInfo(DM dm, PetscInt *dim, PetscInt *nLocal, PetscInt *nGlobal, PetscInt *nGhost)
{
  DM_BF          *bf;
  PetscInt       n, ng, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidIntPointer(dim,2);
  bf = _p_getBF(dm);
  if (!bf->ftCells) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  ierr = DMGetDimension(dm,dim);CHKERRQ(ierr);
  switch (*dim) {
    case 2: ierr = DMBF_2D_GetSizes(dm,(DM_BF_2D_Cells*)bf->ftCells,&n,&N,&ng);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_GetSizes(dm,(DM_BF_3D_Cells*)bf->ftCells,&n,&N,&ng);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  if (nLocal)  *nLocal  = n;
  if (nGlobal) *nGlobal = N;
  if (nGhost)  *nGhost  = ng;
  PetscFunctionReturn(0);
}

/*@
  DMBFGetLocalSize - Gets local number of quadrants in the forest.

  Not Collective

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nLocal  - number of local cells (does not count ghost cells)

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetGlobalSize(), DMBFGetGhostSize()
@*/
PetscErrorCode DMBFGetLocalSize(DM dm, PetscInt *nLocal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,nLocal,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMBFGetGlobalSize - Gets global number of quadrants in the forest.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nGlobal - number of global cells

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetLocalSize(), DMBFGetGhostSize()
@*/
PetscErrorCode DMBFGetGlobalSize(DM dm, PetscInt *nGlobal)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,PETSC_NULL,nGlobal,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMBFGetGhostSize - Gets number of quadrants in the ghost layer.

  Not Collective

  Input Parameters:
+ dm      - the DMBF object

  Output Parameters:
+ nGhost  - number of ghost cells

  Level: beginner

.seealso: DMBFGetInfo(), DMBFGetLocalSize(), DMBFGetGlobalSize()
@*/
PetscErrorCode DMBFGetGhostSize(DM dm, PetscInt *nGhost)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFGetInfo(dm,&dim,PETSC_NULL,PETSC_NULL,nGhost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************************************
 * AMR
 **************************************/

static PetscErrorCode DMCoarsen_BF(DM dm, MPI_Comm comm, DM *coarseDm)
{
  DM_BF          *bf, *coarsebf;
  PetscInt       minLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  CHKERRQ( DMBFCheck(dm) );
  {
    PetscMPIInt mpiComparison;
    MPI_Comm dmcomm = _p_comm(dm);

    ierr = MPI_Comm_compare(comm,dmcomm,&mpiComparison);CHKERRQ(ierr);
    if (mpiComparison != MPI_IDENT && mpiComparison != MPI_CONGRUENT) SETERRQ(dmcomm,PETSC_ERR_SUP,"No support for different communicators");
  }
  ierr = DMBFCloneInit(dm,coarseDm);CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(*coarseDm,&minLevel);CHKERRQ(ierr);
  bf       = _p_getBF(dm);
  coarsebf = _p_getBF(*coarseDm);
  switch (_p_dim(dm)) {
    case 2:
      ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Topology**)&coarsebf->ftTopology,*coarseDm);CHKERRQ(ierr);
      ierr = DMBF_2D_CellsCoarsen((DM_BF_2D_Cells*)bf->ftCells,(DM_BF_2D_Cells**)&coarsebf->ftCells,*coarseDm,minLevel);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    case 3:
      ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Topology**)&coarsebf->ftTopology,*coarseDm);CHKERRQ(ierr);
      ierr = DMBF_3D_CellsCoarsen((DM_BF_3D_Cells*)bf->ftCells,(DM_BF_3D_Cells**)&coarsebf->ftCells,*coarseDm,minLevel);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  ierr = DMBFCloneFinalize(*coarseDm);CHKERRQ(ierr);
  CHKERRQ( DMBFCheck(*coarseDm) );
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefine_BF(DM dm, MPI_Comm comm, DM *fineDm)
{
  DM_BF          *bf, *finebf;
  PetscInt       maxLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  CHKERRQ( DMBFCheck(dm) );
  {
    PetscMPIInt mpiComparison;
    MPI_Comm dmcomm = _p_comm(dm);

    ierr = MPI_Comm_compare(comm,dmcomm,&mpiComparison);CHKERRQ(ierr);
    if (mpiComparison != MPI_IDENT && mpiComparison != MPI_CONGRUENT) SETERRQ(dmcomm,PETSC_ERR_SUP,"No support for different communicators");
  }
  ierr = DMBFCloneInit(dm,fineDm);CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(*fineDm,&maxLevel);CHKERRQ(ierr);
  bf     = _p_getBF(dm);
  finebf = _p_getBF(*fineDm);
  switch (_p_dim(dm)) {
    case 2:
      ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Topology**)&finebf->ftTopology,*fineDm);CHKERRQ(ierr);
      ierr = DMBF_2D_CellsRefine((DM_BF_2D_Cells*)bf->ftCells,(DM_BF_2D_Cells**)&finebf->ftCells,*fineDm,maxLevel);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    case 3:
      ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Topology**)&finebf->ftTopology,*fineDm);CHKERRQ(ierr);
      ierr = DMBF_3D_CellsRefine((DM_BF_3D_Cells*)bf->ftCells,(DM_BF_3D_Cells**)&finebf->ftCells,*fineDm,maxLevel);CHKERRQ(ierr);
      //TODO clone nodes
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  ierr = DMBFCloneFinalize(*fineDm);CHKERRQ(ierr);
  CHKERRQ( DMBFCheck(*fineDm) );
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFAMRSetOperators(DM dm, DM_BF_AmrOps *amrOps)
{
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  bf->amrOps = amrOps;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFAMRFlag(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim, n_cells, i;
  DM_BF_Cell     *cell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  CHKERRQ( DMBFCheck(dm) );
  bf = _p_getBF(dm);
  if (!bf->amrOps) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"AMR operators do not exist");
  /* set data pointers of all ghost cells */
  ierr = DMBFGetInfo(dm,&dim,&n_cells,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<n_cells; i++) {
    cell = _p_cellGetPtrIndex(bf,i);
    ierr = bf->amrOps->setAmrFlag(dm,cell,bf->amrOps->setAmrFlagCtx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFAMRAdapt(DM dm, DM *adaptedDm)
{
  DM_BF          *bf, *adaptedbf;
  const PetscInt dim = _p_dim(dm);
  PetscInt       minLevel, maxLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  CHKERRQ( DMBFCheck(dm) );
  ierr = DMBFCloneInit(dm,adaptedDm);CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(*adaptedDm,&minLevel);CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(*adaptedDm,&maxLevel);CHKERRQ(ierr);
  bf        = _p_getBF(dm);
  adaptedbf = _p_getBF(*adaptedDm);
  /* adapt and partition forest-of-tree cells */
  if (!bf->amrOps) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"AMR operators do not exist");
  switch (dim) {
    case 2:
      ierr = DMBF_2D_TopologyClone((DM_BF_2D_Topology*)bf->ftTopology,(DM_BF_2D_Topology**)&adaptedbf->ftTopology,*adaptedDm);CHKERRQ(ierr);
      ierr = DMBF_2D_CellsAmrAdapt((DM_BF_2D_Cells*)bf->ftCells,(DM_BF_2D_Cells**)&adaptedbf->ftCells,*adaptedDm,bf->amrOps,
                                   minLevel,maxLevel,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr);
      ierr = DMBF_2D_CellsAmrPartition((DM_BF_2D_Cells*)adaptedbf->ftCells);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMBF_3D_TopologyClone((DM_BF_3D_Topology*)bf->ftTopology,(DM_BF_3D_Topology**)&adaptedbf->ftTopology,*adaptedDm);CHKERRQ(ierr);
      ierr = DMBF_3D_CellsAmrAdapt((DM_BF_3D_Cells*)bf->ftCells,(DM_BF_3D_Cells**)&adaptedbf->ftCells,*adaptedDm,bf->amrOps,
                                   minLevel,maxLevel,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf));CHKERRQ(ierr);
      ierr = DMBF_3D_CellsAmrPartition((DM_BF_3D_Cells*)adaptedbf->ftCells);CHKERRQ(ierr);
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  /* create DMBF cells */
  ierr = DMBF_CellsCreate(*adaptedDm);CHKERRQ(ierr);
  /* copy data of DMBF cells from p4est */
  switch (dim) {
    case 2:
      ierr = DMBF_2D_CellsAmrFinalize(*adaptedDm,(DM_BF_2D_Cells*)adaptedbf->ftCells,adaptedbf->cells,_p_cellSize(adaptedbf));CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMBF_3D_CellsAmrFinalize(*adaptedDm,(DM_BF_3D_Cells*)adaptedbf->ftCells,adaptedbf->cells,_p_cellSize(adaptedbf));CHKERRQ(ierr);
      break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  /* setup DMBF cells */
  ierr = DMBF_CellsSetUpOwned(*adaptedDm);CHKERRQ(ierr);
  ierr = DMBF_CellsSetUpGhost(*adaptedDm);CHKERRQ(ierr);
  ierr = DMBF_LocalToGlobalScatterCreate(*adaptedDm,&adaptedbf->ltog);CHKERRQ(ierr);
  /* create forest-of-tree nodes */
  //TODO create nodes
  /* check resulting DM */
  CHKERRQ( DMBFCheck(*adaptedDm) );
  PetscFunctionReturn(0);
}

/***************************************
 * ITERATORS
 **************************************/

PetscErrorCode DMBFIterateOverCellsVectors(DM dm, PetscErrorCode (*iterCell)(DM,DM_BF_Cell*,void*), void *userIterCtx,
                                           Vec *vecRead, PetscInt nVecsRead, Vec *vecReadWrite, PetscInt nVecsReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterCell,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateOverCellsVectors(dm,bf->cells,_p_cellSize(bf),iterCell,userIterCtx,
                                                   vecRead,nVecsRead,vecReadWrite,nVecsReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateOverCellsVectors(dm,bf->cells,_p_cellSize(bf),iterCell,userIterCtx,
                                                   vecRead,nVecsRead,vecReadWrite,nVecsReadWrite);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverCells(DM dm, PetscErrorCode (*iterCell)(DM,DM_BF_Cell*,void*), void *userIterCtx)
{
  PetscFunctionBegin;
  CHKERRQ( DMBFIterateOverCellsVectors(dm,iterCell,userIterCtx,PETSC_NULL,0,PETSC_NULL,0) );
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM,DM_BF_Face*,void*), void *userIterCtx)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (!bf->ghostCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost cells not set up");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateOverFaces(dm,bf->cells,_p_cellSize(bf),iterFace,userIterCtx);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateOverFaces(dm,bf->cells,_p_cellSize(bf),iterFace,userIterCtx);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFSetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateSetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateSetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFSetCellFields(DM dm, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateSetCellFields(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite,nFieldsRead,fieldsRead,nFieldsReadWrite,fieldsReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateSetCellFields(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite,nFieldsRead,fieldsRead,nFieldsReadWrite,fieldsReadWrite);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateGetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateGetCellData(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                              bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                              vecRead,vecReadWrite);CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetCellFields(DM dm, Vec *vecRead, Vec *vecReadWrite, PetscInt nFieldsRead, PetscInt *fieldsRead, PetscInt nFieldsReadWrite, PetscInt *fieldsReadWrite)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  if (vecRead      && bf->nValsPerElemRead)      PetscValidPointer(vecRead,2);
  if (vecReadWrite && bf->nValsPerElemReadWrite) PetscValidPointer(vecReadWrite,3);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
  case 2: ierr = DMBF_2D_IterateGetCellFields(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                            bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                            vecRead,vecReadWrite,nFieldsRead,fieldsRead,nFieldsReadWrite,fieldsReadWrite);CHKERRQ(ierr); break;
  case 3: ierr = DMBF_3D_IterateGetCellFields(dm,bf->cells,_p_cellSize(bf),_p_cellOffsetDataRead(),_p_cellOffsetDataReadWrite(bf),
                                            bf->valsPerElemRead,bf->nValsPerElemRead,bf->valsPerElemReadWrite,bf->nValsPerElemReadWrite,
                                            vecRead,vecReadWrite,nFieldsRead,fieldsRead,nFieldsReadWrite,fieldsReadWrite);CHKERRQ(ierr); break;   default: _p_SETERRQ_UNREACHABLE(dm);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFCommunicateGhostCells(DM dm)
{
  DM_BF          *bf;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells)                 SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
  if (!bf->ownedCellsSetUpCalled) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONGSTATE,"Owned cells not set up");
  /* run ghost exchange */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 2: ierr = DMBF_2D_IterateGhostExchange(dm,bf->cells,_p_cellSize(bf));CHKERRQ(ierr); break;
    case 3: ierr = DMBF_3D_IterateGhostExchange(dm,bf->cells,_p_cellSize(bf));CHKERRQ(ierr); break;
    default: _p_SETERRQ_UNREACHABLE(dm);
  }
  /* setup ghost cells */
  bf->ghostCellsSetUpCalled = PETSC_FALSE;
  ierr = DMBF_CellsSetUpGhost(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/***************
 * VIEWER
 ***************/

PetscErrorCode DMView_BF(DM dm, PetscViewer viewer)
{

  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!dm) SETERRQ(_p_comm(dm),PETSC_ERR_ARG_WRONG,"No DM provided to view");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    ierr = DMBFVTKWriteAll((PetscObject)dm,viewer);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(_p_comm(dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_BF(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;
  PetscInt       vsize,locDof,ng;
  PetscInt       bs, blockSize[3] = {1, 1, 1};
  PetscViewerVTKFieldType ft;
  Vec            locv;
  const char     *name;


  PetscFunctionBegin;
  ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(_p_comm(v),PETSC_ERR_ARG_WRONG,"Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    /* create a copy and store it in the viewer */
    ierr = DMCreateLocalVector(dm,&locv);CHKERRQ(ierr);                /* we store local vectors in the viewer. done't know why, since we don't need ghost values */
    ierr = DMGlobalToLocal(dm,v,INSERT_VALUES,locv);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
    ierr = DMBFGetLocalSize(dm,&locDof);CHKERRQ(ierr);
    ierr = DMBFGetGhostSize(dm,&ng);CHKERRQ(ierr);
    ierr = VecGetSize(locv,&vsize);CHKERRQ(ierr);
    ierr = DMBFGetBlockSize(dm,blockSize);CHKERRQ(ierr);
    bs = blockSize[0]*blockSize[1]*blockSize[2];
    locDof = (locDof + ng)*bs;
    //if(vsize == P4EST_DIM*size)                { ft = PETSC_VTK_CELL_VECTOR_FIELD; } /* right now this is not actually supported (dm local to global is only for cell fields) */
    //else if(vsize == size)                     { ft = PETSC_VTK_CELL_FIELD;        }
    if(vsize == locDof)                    { ft = PETSC_VTK_CELL_FIELD;        } /* if it's a local vector field, there will be an error before this in the dmlocaltoglobal */
    else  SETERRQ(_p_comm(locv), PETSC_ERR_SUP, "Only scalar cell fields currently supported");

    ierr = PetscViewerVTKAddField(viewer,(PetscObject)dm,DMBFVTKWriteAll,PETSC_DEFAULT,ft,PETSC_TRUE,(PetscObject)locv);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(_p_comm(dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(0);
}
