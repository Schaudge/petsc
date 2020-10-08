#include <petscdmbf.h>
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "petsc_p4est_package.h"


#if defined(PETSC_HAVE_P4EST)

//TODO the way it's implemented now, only 2d domains are supported
#if !defined(P4_TO_P8)
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#include <p4est_vtk.h>
#include <p4est_plex.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_geometry.h>
#include <p8est_ghost.h>
#include <p8est_lnodes.h>
#include <p8est_vtk.h>
#include <p8est_plex.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#endif /* !defined(P4_TO_P8) */

/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/

typedef struct _p_DM_BF {
  /* forest-of-tree topology */
  p4est_connectivity_t  *connectivity;
  p4est_geometry_t      *geometry;//TODO need this?
  /* forest-of-tree cells */
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  /* forest-of-tree nodes */
  p4est_lnodes_t        *lnodes;
  /* DMBF cells */
  DM_BF_Cell            *cells;
  /* [option] blocks within a cell */
  PetscInt              blockSize[3];
  /* [option] settings for cell data */
  PetscInt              *valsPerElemRead, *valsPerElemReadWrite;
  PetscInt              nValsPerElemRead, nValsPerElemReadWrite;
  PetscInt              valsPerElemReadTotal, valsPerElemReadWriteTotal;
} DM_BF;

typedef struct _p_DM_BF_MatCtx {
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  void                  *userMatCtx;
} DM_BF_MatCtx;

/******************************************************************************
 * PRIVATE FUNCTIONS WITHOUT ERROR CHECKING
 *****************************************************************************/

static inline DM_BF *_p_getBF(DM dm)
{
  return (DM_BF*) ((DM_Forest*) dm->data)->data;
}

#define _p_bytesAlign(a) (((a)+(PETSC_MEMALIGN-1)) & ~(PETSC_MEMALIGN-1))

static inline size_t _p_sizeOfCellInfo()
{
  return _p_bytesAlign(sizeof(DM_BF_Cell));
}

static inline size_t _p_sizeOfCellDataRead(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadTotal));
}

/* unused
static inline size_t _p_sizeOfCellDataReadWrite(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadWriteTotal));
}
*/

static inline size_t _p_sizeOfCellData(DM_BF *bf)
{
  return _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadTotal)) +
         _p_bytesAlign((size_t)(sizeof(PetscScalar)*bf->blockSize[0]*bf->blockSize[1]*bf->blockSize[2]*bf->valsPerElemReadWriteTotal));
}

static inline size_t _p_sizeOfCell(DM_BF *bf)
{
  return _p_sizeOfCellInfo() + _p_sizeOfCellData(bf);
}

static inline DM_BF_Cell *_p_getCellPtrIndex(DM_BF *bf, PetscInt index)
{
  return (DM_BF_Cell*)(((char*)bf->cells) + _p_sizeOfCell(bf) * ((size_t)index));
}

static inline DM_BF_Cell *_p_getCellPtrQuadId(DM_BF *bf, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost)
{
  if (!is_ghost) {
    p4est_tree_t *tree = p4est_tree_array_index(bf->p4est->trees,treeid);

    return (DM_BF_Cell*)(((char*)bf->cells) + _p_sizeOfCell(bf) * ((size_t)(tree->quadrants_offset + quadid)));
  } else {
    return (DM_BF_Cell*)(((char*)bf->cells) + _p_sizeOfCell(bf) * ((size_t)(bf->p4est->global_first_quadrant[bf->p4est->mpirank+1] + quadid)));
  }
}

static inline PetscScalar *_p_getCellDataRead(DM_BF_Cell *cell)
{
  return (PetscScalar*)(((char*)cell) + _p_sizeOfCellInfo());
}

static inline PetscScalar *_p_getCellDataReadWrite(DM_BF_Cell *cell, DM_BF *bf)
{
  return (PetscScalar*)(((char*)cell) + _p_sizeOfCellInfo() + _p_sizeOfCellDataRead(bf));
}

/******************************************************************************
 * PRIVATE FUNCTION DEFINITIONS
 *****************************************************************************/

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

static PetscErrorCode DMBF_ConnectivityCreate(DM dm, p4est_connectivity_t **connectivity)
{
  const char        *prefix;
  DMForestTopology  topologyName;
  PetscBool         isBrick;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);

  /* get topology name */
  ierr = DMForestGetTopology(dm,&topologyName);CHKERRQ(ierr);
  if (!topologyName) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DMBF needs a topology");
  ierr = PetscStrcmp((const char*) topologyName,"brick",&isBrick);CHKERRQ(ierr);

  if (isBrick && dm->setfromoptionscalled) { /* if brick topology with given uptions */
    PetscBool flgN, flgP, flgB, periodic=PETSC_FALSE;
    PetscInt  N[3]={2,2,2}, P[3]={0,0,0}, nretN=P4EST_DIM, nretP=P4EST_DIM, nretB=2*P4EST_DIM, i, j;
    PetscReal B[6]={0.0,1.0,0.0,1.0,0.0,1.0};

    /* get brick options */
    ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_size",N,&nretN,&flgN);CHKERRQ(ierr);
    ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_periodicity",P,&nretP,&flgP);CHKERRQ(ierr);
    ierr = PetscOptionsGetRealArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_bounds",B,&nretB,&flgB);CHKERRQ(ierr);
    if (flgN && nretN != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d sizes in -dm_p4est_brick_size, gave %d",P4EST_DIM,nretN);
    if (flgP && nretP != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d periodicities in -dm_p4est_brick_periodicity, gave %d",P4EST_DIM,nretP);
    if (flgB && nretB != 2 * P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d bounds in -dm_p4est_brick_bounds, gave %d",P4EST_DIM,nretP);

    /* update periodicity */
    for (i=0; i<P4EST_DIM; i++) {
      P[i] = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
      periodic = (PetscBool)(P[i] || periodic);
      if (!flgB) B[2*i+1] = N[i];
    }
    ierr = DMSetPeriodicity(dm,periodic,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* create connectivity */
    PetscStackCallP4estReturn(
        *connectivity,p4est_connectivity_new_brick,
        ((int) N[0], (int) N[1], (P[0] == DM_BOUNDARY_PERIODIC), (P[1] == DM_BOUNDARY_PERIODIC)) );

    { /* scale to bounds */
      double *vertices = (*connectivity)->vertices;

      for (i=0; i<3*(*connectivity)->num_vertices; i++) {
        j = i % 3;
        vertices[i] = B[2*j] + (vertices[i]/N[j]) * (B[2*j+1] - B[2*j]);
      }
    }
  } else { /* otherwise call generic function */
    /* create connectivity */
    PetscStackCallP4estReturn(*connectivity,p4est_connectivity_new_byname,((const char*) topologyName));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_ConnectivityDestroy(DM dm, p4est_connectivity_t *connectivity)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_connectivity_destroy,(connectivity));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_P4estCreate(DM dm, p4est_connectivity_t *connectivity, p4est_t **p4est)
{
  PetscInt       initLevel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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

static PetscErrorCode DMBF_P4estDestroy(DM dm, p4est_t *p4est)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_destroy,(p4est));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_GhostCreate(DM dm, p4est_t *p4est, p4est_ghost_t **ghost)
{
  PetscFunctionBegin;
  if (!p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est does not exist");
  PetscStackCallP4estReturn(*ghost,p4est_ghost_new,(p4est,P4EST_CONNECT_FULL));
  //TODO which connect flag, P4EST_CONNECT_FULL, P4EST_CONNECT_FACE, ...?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_GhostDestroy(DM dm, p4est_ghost_t *ghost)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_ghost_destroy,(ghost));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsCreate(DM dm, DM_BF_Cell **cells)
{
  DM_BF          *bf;
  size_t         n_cells, ng_cells, cell_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get number of cells and their size */
  bf = _p_getBF(dm);
  if (!bf->ghost) {
    ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  }
  n_cells   = (size_t)bf->p4est->local_num_quadrants;
  ng_cells  = (size_t)bf->ghost->ghosts.elem_count;
  cell_size = _p_sizeOfCell(bf);
  /* create DMBF cells */
  ierr = PetscMalloc((n_cells+ng_cells)*cell_size,cells);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)dm,(n_cells+ng_cells)*cell_size);CHKERRQ(ierr);
  /* set cell info */
  ierr = DMBFSetCellData(dm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBF_CellsDestroy(DM dm, DM_BF_Cell *cells)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  /* create topology */
  if (bf->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity exists already");
  ierr = DMBF_ConnectivityCreate(dm,&bf->connectivity);CHKERRQ(ierr);
  /* create forest of trees */
  if (!bf->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity does not exist");
  if (bf->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est exists already");
  ierr = DMBF_P4estCreate(dm, bf->connectivity, &bf->p4est);CHKERRQ(ierr);
  /* create ghost */
//if (bf->ghost) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost exists already");
//ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  /* create nodes */
  //TODO
  /* create DMBF cells */
  ierr = DMBF_CellsCreate(dm,&bf->cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  /* destroy forest-of-tree objects */
//if (bf->lnodes)       { PetscStackCallP4est(p4est_lnodes_destroy,(bf->lnodes)); }
  if (bf->ghost)        { ierr = DMBF_GhostDestroy(dm,bf->ghost);CHKERRQ(ierr); }
  if (bf->p4est)        { ierr = DMBF_P4estDestroy(dm,bf->p4est);CHKERRQ(ierr); }
//if (bf->geometry)     { PetscStackCallP4est(p4est_geometry_destroy,(bf->geometry)); }
  if (bf->connectivity) { ierr = DMBF_ConnectivityDestroy(dm,bf->connectivity);CHKERRQ(ierr); }
  bf->lnodes            = PETSC_NULL;
  bf->ghost             = PETSC_NULL;
  bf->p4est             = PETSC_NULL;
  bf->geometry          = PETSC_NULL;
  bf->connectivity      = PETSC_NULL;
  /* destroy DMBF cells */
  if (bf->cells) { ierr = DMBF_CellsDestroy(dm,bf->cells);CHKERRQ(ierr); }
  bf->cells      = PETSC_NULL;
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

PetscErrorCode DMView_BF(DM dm, PetscViewer viewer)
{

  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;
  PetscInt       gsize;
  PetscInt       vsize;
  PetscViewerVTKFieldType ft;

  PetscFunctionBegin;

  if (!dm) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"No DM provided to view");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    ierr = DMBFVTKWriteAll((PetscObject) dm,viewer);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
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
  dm->ops->view               = DMView_BF;
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
  ierr = DMSetDimension(dm,P4EST_DIM);CHKERRQ(ierr);

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
  bf->connectivity              = PETSC_NULL;
  bf->geometry                  = PETSC_NULL;
  bf->p4est                     = PETSC_NULL;
  bf->ghost                     = PETSC_NULL;
  bf->lnodes                    = PETSC_NULL;
  bf->cells                     = PETSC_NULL;
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
  DM_BF *bf;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est does not exist");
  *(void**)p4est = bf->p4est;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFGetGhost(DM dm, void *ghost)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->ghost) {
    ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  }
  *(void**)ghost = bf->ghost;
  PetscFunctionReturn(0);
}


/*************************
 * VEC
 *************************/

PetscErrorCode VecView_BF(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isvtk, ishdf5, isdraw, isglvis;
  PetscErrorCode ierr;
  PetscInt       gsize;
  PetscInt       vsize;
  PetscViewerVTKFieldType ft;

  PetscFunctionBegin;
  ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_ARG_WRONG,"Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if(isvtk) {
    ierr = DMBFGetGlobalSize(dm, &gsize);CHKERRQ(ierr);
    ierr = VecGetSize(v, &vsize);CHKERRQ(ierr);
    if(vsize == 3*gsize)                { ft = PETSC_VTK_CELL_VECTOR_FIELD; }
    else if(vsize == gsize)             { ft = PETSC_VTK_CELL_FIELD;        }
    else  SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Only vector and scalar cell fields currently supported");    
    ierr = PetscViewerVTKAddField(viewer,(PetscObject)dm,DMBFVTKWriteAll,PETSC_DEFAULT,ft,PETSC_TRUE,(PetscObject)v);CHKERRQ(ierr);
  } else if(ishdf5 || isdraw || isglvis) {
    SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"non-VTK viewer currently not supported by BF");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_BF(DM dm, Vec *vec)
{
  DM_BF          *bf;
  PetscInt       n, ng;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  bf = _p_getBF(dm);
  if (!bf->ghost) {
    ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  }
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  ng = (PetscInt)bf->ghost->ghosts.elem_count;
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(vec,2);
  /* get number of entries */
  bf = _p_getBF(dm);
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  N  = (PetscInt)bf->p4est->global_num_quadrants;
  /* create vector */
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),n,N,vec);CHKERRQ(ierr);
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
  DM_BF          *bf;
  void           *appctx;
  PetscInt       n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(mat,2);
  /* get number of rows/cols */
  bf = _p_getBF(dm);
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  N  = (PetscInt)bf->p4est->global_num_quadrants;
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

/* take global vector and return local version */
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

//{ ghostupdatebegin
// ghostupdateend on local vec
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
 * MESH
 **************************************/

/*@
  DMBFGetLocalSize - Gets number of local quadrants.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nQuads  - number of local quadrants (does not count ghosts)

  Level: beginner

.seealso: DMBFGetGlobalSize(), DMBFGetBlockSize()
@*/

PetscErrorCode DMBFGetLocalSize(DM dm, PetscInt *nQuads) {
  
  DM_BF   *bf;
  
  PetscFunctionBegin;
  bf = _p_getBF(dm);
  *nQuads  = (PetscInt)bf->p4est->local_num_quadrants;
  PetscFunctionReturn(0);
}

/*@
  DMBFGetGlobalSize - Gets number of global quadrants.

  Logically collective on DM

  Input Parameters:
+ dm      - the DMBF object
- nQuads  - number of global quadrants

  Level: beginner

.seealso: DMBFGetGlobalSize(), DMBFGetBlockSize()
@*/

PetscErrorCode DMBFGetGlobalSize(DM dm, PetscInt *nQuads) {
  
  DM_BF   *bf;
  
  PetscFunctionBegin;
  bf = _p_getBF(dm);
  *nQuads  = (PetscInt)bf->p4est->global_num_quadrants;
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMinimumRefinement(dm,&amrCtx.minLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer      = bf->p4est->user_pointer;
  bf->p4est->user_pointer = (void*) &amrCtx;
  /* coarsen & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_coarsen,(bf->p4est,0,p4est_coarsen_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(bf->p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  bf->p4est->user_pointer = p4est_user_pointer;
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set AMR parameters */
  ierr = DMForestGetMaximumRefinement(dm,&amrCtx.maxLevel);CHKERRQ(ierr);
  /* prepare p4est for AMR */
  bf = _p_getBF(dm);
  p4est_user_pointer      = bf->p4est->user_pointer;
  bf->p4est->user_pointer = (void*) &amrCtx;
  /* refine & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_refine,(bf->p4est,0,p4est_refine_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(bf->p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  bf->p4est->user_pointer = p4est_user_pointer;
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

static void _p_getCellInfo(/*IN */ p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost,
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

static void _p_getCellVecView(/*IN    */ const PetscScalar **vecViewRead, PetscInt nVecsRead,
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
  DM_BF_CellIterCtx *iterCtx  = ctx;
  DM_BF             *bf       = iterCtx->bf;
  DM_BF_Cell        *cell     = _p_getCellPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscErrorCode    ierr;

  /* assign vector view to cell */
  cell->vecViewRead      = iterCtx->cellVecViewRead;
  cell->vecViewReadWrite = iterCtx->cellVecViewReadWrite;
  /* get vector view */
  _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterCell,2);
  if (nVecsRead)      PetscValidPointer(vecRead,4);
  if (nVecsReadWrite) PetscValidPointer(vecReadWrite,6);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
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
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,p4est_iter_volume,NULL,NULL));
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

    _p_getCellInfo(p4est,side->is.full.quad,side->treeid,side->is.full.quadid,0,cell);
    _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
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
        _p_getCellInfo(p4est,sideL->is.hanging.quad[i],sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i],cell);
        _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
        face->cellL[i] = cell;
        cell++;
      }
    } else {
      _p_getCellInfo(p4est,sideL->is.full.quad,sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost,cell);
      _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
      face->cellL[0] = cell;
      cell++;
    }
    if (sideR->is_hanging) {
      for (i=0; i<face->nCellsR; i++) {
        _p_getCellInfo(p4est,sideR->is.hanging.quad[i],sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i],cell);
        _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
        face->cellR[i] = cell;
        cell++;
      }
    } else {
      _p_getCellInfo(p4est,sideR->is.full.quad,sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost,cell);
      _p_getCellVecView(iterCtx->vecViewRead,iterCtx->nVecsRead,iterCtx->vecViewReadWrite,iterCtx->nVecsReadWrite,cell);
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

  PetscFunctionBegin;
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
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,NULL,p4est_iter_face,NULL));
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
    face->cellL[0] = _p_getCellPtrQuadId(bf,side->treeid,side->is.full.quadid,0);
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
        face->cellL[i] = _p_getCellPtrQuadId(bf,sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellL[0] = _p_getCellPtrQuadId(bf,sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost);
    }
    if (sideR->is_hanging) {
      for (i=0; i<face->nCellsR; i++) {
        face->cellR[i] = _p_getCellPtrQuadId(bf,sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i]);
      }
    } else {
      face->cellR[0] = _p_getCellPtrQuadId(bf,sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost);
    }
  }
  /* call face function */
  ierr = iterCtx->iterFace(face,iterCtx->userIterCtx);CHKERRV(ierr);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM_BF_Face*,void*), void *userIterCtx)
{
  DM_BF             *bf;
  DM_BF_FaceIterCtx iterCtx;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidFunction(iterFace,2);
  bf = _p_getBF(dm);
  /* set iterator context */
  iterCtx.bf          = bf;
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  /* run iterator */
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,NULL,p4est_iter_face,NULL));
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
  DM_BF_SetCellDataIterCtx *iterCtx  = ctx;
  DM_BF                    *bf       = iterCtx->bf;
  DM_BF_Cell               *cell     = _p_getCellPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscScalar              *data;
  PetscInt                 i, j, di;

  /* get cell info */
  _p_getCellInfo(info->p4est,info->quad,info->treeid,info->quadid,0,cell);
  /* set cell data */
  if (0 < iterCtx->nVecsRead) {
    data = _p_getCellDataRead(cell);
    cell->dataRead = (const PetscScalar*)data;
    di = 0;
    for (i=0; i<bf->nValsPerElemRead; i++) {
      for (j=0; j<bf->valsPerElemRead[i]; j++) {
        if (bf->valsPerElemReadTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read index is above max");
        data[di] = iterCtx->vecViewRead[i][bf->valsPerElemRead[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
  if (0 < iterCtx->nVecsReadWrite) {
    cell->dataReadWrite = data = _p_getCellDataReadWrite(cell,bf);
    di = 0;
    for (i=0; i<bf->nValsPerElemReadWrite; i++) {
      for (j=0; j<bf->valsPerElemReadWrite[i]; j++) {
        if (bf->valsPerElemReadWriteTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read-write index is above max");
        data[di] = iterCtx->vecViewReadWrite[i][bf->valsPerElemReadWrite[i]*cell->indexLocal+j];
        di++;
      }
    }
  }
  /* assign cell to forest quadrant */
  info->quad->p.user_data = cell;
}

PetscErrorCode DMBFSetCellData(DM dm, Vec *vecRead, Vec *vecReadWrite)
{
  DM_BF                    *bf;
  DM_BF_SetCellDataIterCtx iterCtx;
  PetscInt                 i;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
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
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,p4est_iter_set_cell_data,NULL,NULL));
  bf->p4est->data_size = _p_sizeOfCell(bf);
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
  DM_BF_GetCellDataIterCtx *iterCtx  = ctx;
  DM_BF                    *bf       = iterCtx->bf;
  DM_BF_Cell               *cell     = _p_getCellPtrQuadId(bf,info->treeid,info->quadid,0);
  PetscScalar              *data;
  PetscInt                 i, j, di=0;

  /* get cell data */
  if (0 < iterCtx->nVecsRead) {
    data = _p_getCellDataRead(cell);
    di   = 0;
    for (i=0; i<bf->nValsPerElemRead; i++) {
      for (j=0; j<bf->valsPerElemRead[i]; j++) {
        if (bf->valsPerElemReadTotal <= di) SETERRABORT(PetscObjectComm((PetscObject)bf),PETSC_ERR_PLIB,"Cell data read index is above max");
        iterCtx->vecViewRead[i][bf->valsPerElemRead[i]*cell->indexLocal+j] = data[di];
        di++;
      }
    }
  }
  if (0 < iterCtx->nVecsReadWrite) {
    data = _p_getCellDataReadWrite(cell,bf);
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  if (!bf->cells) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cells do not exist");
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
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,p4est_iter_get_cell_data,NULL,NULL));
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

PetscErrorCode DMBFExchangeGhostCells(DM dm)
{
  DM_BF      *bf;
  DM_BF_Cell *ghostCells;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_getBF(dm);
  ghostCells = _p_getCellPtrIndex(bf,bf->p4est->local_num_quadrants);
  PetscStackCallP4est(p4est_ghost_exchange_data,(bf->p4est,bf->ghost,ghostCells));
  PetscFunctionReturn(0);
}


#endif /* defined(PETSC_HAVE_P4EST) */
