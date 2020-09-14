#include <petscdmbf.h>
#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include "petsc_p4est_package.h"

#if defined(PETSC_HAVE_P4EST)

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
  /* topology */
  p4est_connectivity_t  *connectivity;
  p4est_geometry_t      *geometry;//TODO need this?
  /* cells */
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  /* nodes */
  p4est_lnodes_t        *lnodes;
  /* block */
  PetscInt              blockSize[3];
} DM_BF;

typedef struct _p_DM_BF_MatCtx {
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  void                  *userMatCtx;
} DM_BF_MatCtx;

/******************************************************************************
 * SECRET FUNCTIONS (W/O ERROR CHECKING)
 *****************************************************************************/

static inline DM_BF *_p_GetBF(DM dm)
{
  return (DM_BF*) ((DM_Forest*) dm->data)->data;
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
    ierr = DMSetPeriodicity(dm,periodic,NULL,NULL,NULL);CHKERRQ(ierr);

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

static PetscErrorCode DMSetUp_BF(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_GetBF(dm);

  /* create topology */
  if (bf->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity exists already");
  ierr = DMBF_ConnectivityCreate(dm,&bf->connectivity);CHKERRQ(ierr);

  /* create forest of trees */
  if (!bf->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity does not exist");
  if (bf->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est exists already");
  ierr = DMBF_P4estCreate(dm, bf->connectivity, &bf->p4est);CHKERRQ(ierr);

  /* create ghost */
  //if (bf->ghost)  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost exists already");
  //ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);

  /* create nodes */
  //TODO

  PetscFunctionReturn(0);
}

static PetscErrorCode DMBFClear(DM dm)
{
  DM_BF          *bf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  bf = _p_GetBF(dm);
//if (bf->lnodes)       { PetscStackCallP4est(p4est_lnodes_destroy,(bf->lnodes)); }
  if (bf->ghost)        { ierr = DMBF_GhostDestroy(dm,bf->ghost);CHKERRQ(ierr); }
  if (bf->p4est)        { ierr = DMBF_P4estDestroy(dm,bf->p4est);CHKERRQ(ierr); }
//if (bf->geometry)     { PetscStackCallP4est(p4est_geometry_destroy,(bf->geometry)); }
  if (bf->connectivity) { ierr = DMBF_ConnectivityDestroy(dm,bf->connectivity);CHKERRQ(ierr); }
  bf->lnodes       = NULL;
  bf->ghost        = NULL;
  bf->p4est        = NULL;
  bf->geometry     = NULL;
  bf->connectivity = NULL;
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
  bf = _p_GetBF(dm);
  for (i=0; i<dim; i++) {
    bf->blockSize[i] = blockSize[i];
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
  bf = _p_GetBF(dm);
  for (i=0; i<dim; i++) {
    blockSize[i] = bf->blockSize[i];
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
  bf->connectivity = NULL;
  bf->geometry     = NULL;
  bf->p4est        = NULL;
  bf->ghost        = NULL;
  bf->lnodes       = NULL;
  bf->blockSize[0] = PETSC_DEFAULT;
  bf->blockSize[1] = PETSC_DEFAULT;
  bf->blockSize[2] = PETSC_DEFAULT;

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
  bf = _p_GetBF(dm);
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
  bf = _p_GetBF(dm);
  if (!bf->ghost) {
    ierr = DMBF_GhostCreate(dm,bf->p4est,&bf->ghost);CHKERRQ(ierr);
  }
  *(void**)ghost = bf->ghost;
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
  bf = _p_GetBF(dm);
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
  bf = _p_GetBF(dm);
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  N  = (PetscInt)bf->p4est->global_num_quadrants;
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

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  PetscValidPointer(mat,2);
  /* get number of rows/cols */
  bf = _p_GetBF(dm);
  n  = (PetscInt)bf->p4est->local_num_quadrants;
  N  = (PetscInt)bf->p4est->global_num_quadrants;
  /* create matrix */
  ierr = DMGetApplicationContext(dm,&appctx);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),n,n,N,N,appctx,mat);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  //TODO set null space?
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
  bf = _p_GetBF(dm);
  p4est_user_pointer      = bf->p4est->user_pointer;
  bf->p4est->user_pointer = (void*) &amrCtx;
  /* coarsen & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_coarsen,(bf->p4est,0,p4est_coarsen_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(bf->p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  bf->p4est->user_pointer = p4est_user_pointer;
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
  bf = _p_GetBF(dm);
  p4est_user_pointer      = bf->p4est->user_pointer;
  bf->p4est->user_pointer = (void*) &amrCtx;
  /* refine & balance */
  for (i=0; i<nCycles; i++) {
    PetscStackCallP4est(p4est_refine,(bf->p4est,0,p4est_refine_uniformly,NULL));
  }
  PetscStackCallP4est(p4est_balance,(bf->p4est,P4EST_CONNECT_FULL,NULL));
  /* finalize p4est after AMR */
  bf->p4est->user_pointer = p4est_user_pointer;
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

static void _p_get_cell_data(/*IN */ p4est_t *p4est, p4est_quadrant_t *quad, p4est_topidx_t treeid, p4est_locidx_t quadid, int8_t is_ghost,
                             /*OUT*/ DM_BF_CellData *cellData)
{
  const p4est_qcoord_t qlength = P4EST_QUADRANT_LEN(quad->level);
  p4est_tree_t         *tree = p4est_tree_array_index(p4est->trees,treeid);
  double               vertex1[3], vertex2[3];

  /* get vertex coordinates of opposite corners */
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x,quad->y,vertex1);
  p4est_qcoord_to_vertex(p4est->connectivity,treeid,quad->x+qlength,quad->y+qlength,vertex2);
  /* set cell data */
  if (!is_ghost) {
    cellData->index_local  = (PetscInt)(tree->quadrants_offset + quadid);
    cellData->index_global = cellData->index_local + (PetscInt)p4est->global_first_quadrant[p4est->mpirank];
  } else {
    cellData->index_local  = (PetscInt)(p4est->global_first_quadrant[p4est->mpirank+1] + quadid);
    cellData->index_global = -1;
  }
  cellData->level     = (PetscInt)quad->level;
  cellData->corner[0] = (PetscReal)vertex1[0];
  cellData->corner[1] = (PetscReal)vertex1[1];
  cellData->corner[2] = (PetscReal)vertex1[2];
  cellData->length[0] = (PetscReal)(vertex2[0] - vertex1[0]);
  cellData->length[1] = (PetscReal)(vertex2[1] - vertex1[1]);
  cellData->length[2] = (PetscReal)(vertex2[2] - vertex1[2]);
}

static void _p_get_vec_data(/*IN    */ const PetscScalar **vecdata_in, PetscInt nVecsIn, PetscScalar **vecdata_out, PetscInt nVecsOut,
                            /*IN/OUT*/ DM_BF_CellData *cellData)
{
  PetscInt i;

  for (i=0; i<nVecsIn; i++) {
    cellData->vecdata_in[i] = vecdata_in[i][cellData->index_local];
  }
  for (i=0; i<nVecsOut; i++) {
    cellData->vecdata_out[i] = vecdata_out[i][cellData->index_local];
  }
}

static void _p_set_vec_data(/*IN */ DM_BF_CellData *cellData,
                            /*OUT*/ PetscScalar **vecdata_out, PetscInt nVecsOut)
{
  PetscInt i;

  for (i=0; i<nVecsOut; i++) {
    vecdata_out[i][cellData->index_local] = cellData->vecdata_out[i];
  }
}

typedef struct _p_DM_BF_CellIterCtx {
  PetscErrorCode    (*iterCell)(DM_BF_CellData*,void*);
  void              *userIterCtx;
  PetscInt          nVecsIn, nVecsOut;
  const PetscScalar **vecdata_in;
  PetscScalar       **vecdata_out;
  DM_BF_CellData    cellData;
} DM_BF_CellIterCtx;

static void p4est_iter_volume(p4est_iter_volume_info_t *info, void *ctx)
{
  DM_BF_CellIterCtx *iterCtx = ctx;
  DM_BF_CellData    *cellData = &iterCtx->cellData;
  PetscErrorCode    ierr;

  /* get cell and vector data */
  _p_get_cell_data(info->p4est,info->quad,info->treeid,info->quadid,0,cellData);
  _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
  /* call cell function */
  ierr = iterCtx->iterCell(cellData,iterCtx->userIterCtx);CHKERRV(ierr);
  /* set vector data */
  _p_set_vec_data(cellData,iterCtx->vecdata_out,iterCtx->nVecsOut);
}

PetscErrorCode DMBFIterateOverCellsVectors(DM dm, PetscErrorCode (*iterCell)(DM_BF_CellData*,void*), void *userIterCtx,
                                           Vec *in, PetscInt nVecsIn, Vec *out, PetscInt nVecsOut)
{
  DM_BF             *bf;
  DM_BF_CellIterCtx iterCtx;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set iterator context */
  iterCtx.iterCell    = iterCell;
  iterCtx.userIterCtx = userIterCtx;
  iterCtx.nVecsIn     = nVecsIn;
  iterCtx.nVecsOut    = nVecsOut;
  if (0 < nVecsIn) {
    ierr = PetscMalloc1(nVecsIn,&iterCtx.cellData.vecdata_in);CHKERRQ(ierr);
    ierr = PetscMalloc1(nVecsIn,&iterCtx.vecdata_in);CHKERRQ(ierr);
    for (i=0; i<nVecsIn; i++) {
      ierr = VecGetArrayRead(in[i],&iterCtx.vecdata_in[i]);CHKERRQ(ierr);
    }
  }
  if (0 < nVecsOut) {
    ierr = PetscMalloc1(nVecsOut,&iterCtx.cellData.vecdata_out);CHKERRQ(ierr);
    ierr = PetscMalloc1(nVecsOut,&iterCtx.vecdata_out);CHKERRQ(ierr);
    for (i=0; i<nVecsOut; i++) {
      ierr = VecGetArray(out[i],&iterCtx.vecdata_out[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  bf = _p_GetBF(dm);
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,p4est_iter_volume,NULL,NULL));
  /* clear iterator context */
  if (0 < nVecsIn) {
    for (i=0; i<nVecsIn; i++) {
      ierr = VecRestoreArrayRead(in[i],&iterCtx.vecdata_in[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecdata_in);CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellData.vecdata_in);CHKERRQ(ierr);
  }
  if (0 < nVecsOut) {
    for (i=0; i<nVecsOut; i++) {
      ierr = VecRestoreArray(out[i],&iterCtx.vecdata_out[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecdata_out);CHKERRQ(ierr);
    ierr = PetscFree(iterCtx.cellData.vecdata_out);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverCells(DM dm, PetscErrorCode (*iterCell)(DM_BF_CellData*,void*), void *userIterCtx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFIterateOverCellsVectors(dm,iterCell,userIterCtx,PETSC_NULL,0,PETSC_NULL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct _p_DM_BF_FaceIterCtx {
  PetscErrorCode    (*iterFace)(DM_BF_FaceData*,void*);
  void              *userIterCtx;
  PetscInt          nVecsIn, nVecsOut;
  const PetscScalar **vecdata_in;
  PetscScalar       **vecdata_out;
  DM_BF_FaceData    faceData;
  DM_BF_CellData    cellData[3]; //TODO only for 2D
} DM_BF_FaceIterCtx;

static void p4est_iter_face(p4est_iter_face_info_t *info, void *ctx)
{
  p4est_t              *p4est = info->p4est;
  DM_BF_FaceIterCtx    *iterCtx = ctx;
  DM_BF_FaceData       *faceData = &iterCtx->faceData;
  DM_BF_CellData       *cellData = iterCtx->cellData;
  const PetscBool      is_boundary = (1 == info->sides.elem_count);
  PetscInt             i;
  PetscErrorCode       ierr;

  //TODO if debug
  faceData->cellDataL[0] = PETSC_NULL;
  faceData->cellDataL[1] = PETSC_NULL;
  faceData->cellDataL[2] = PETSC_NULL;
  faceData->cellDataL[3] = PETSC_NULL;
  faceData->cellDataR[0] = PETSC_NULL;
  faceData->cellDataR[1] = PETSC_NULL;
  faceData->cellDataR[2] = PETSC_NULL;
  faceData->cellDataR[3] = PETSC_NULL;

  /* get cell and vector data */
  if (is_boundary) {
    p4est_iter_face_side_t *side = p4est_iter_fside_array_index_int(&info->sides,0);

    _p_get_cell_data(p4est,side->is.full.quad,side->treeid,side->is.full.quadid,0,cellData);
    _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
    faceData->nCellsL = 1;
    faceData->nCellsR = 0;
    faceData->cellDataL[0] = cellData;
  } else { /* !is_boundary */
    p4est_iter_face_side_t *sideL = p4est_iter_fside_array_index_int(&info->sides,0);
    p4est_iter_face_side_t *sideR = p4est_iter_fside_array_index_int(&info->sides,1);

    faceData->nCellsL = (sideL->is_hanging ? 2 : 1); //TODO only 2D
    faceData->nCellsR = (sideR->is_hanging ? 2 : 1); //TODO only 2D
    if ( !(1 <= faceData->nCellsL && 1 <= faceData->nCellsR && (faceData->nCellsL + faceData->nCellsR) <= 3) ) { //TODO only 2D
      //TODO error
    }
    if (sideL->is_hanging) {
      for (i=0; i<faceData->nCellsL; i++) {
        _p_get_cell_data(p4est,sideL->is.hanging.quad[i],sideL->treeid,sideL->is.hanging.quadid[i],sideL->is.hanging.is_ghost[i],cellData);
        _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
        faceData->cellDataL[i] = cellData;
        cellData++;
      }
    } else {
      _p_get_cell_data(p4est,sideL->is.full.quad,sideL->treeid,sideL->is.full.quadid,sideL->is.full.is_ghost,cellData);
      _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
      faceData->cellDataL[0] = cellData;
      cellData++;
    }
    if (sideR->is_hanging) {
      for (i=0; i<faceData->nCellsR; i++) {
        _p_get_cell_data(p4est,sideR->is.hanging.quad[i],sideR->treeid,sideR->is.hanging.quadid[i],sideR->is.hanging.is_ghost[i],cellData);
        _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
        faceData->cellDataR[i] = cellData;
        cellData++;
      }
    } else {
      _p_get_cell_data(p4est,sideR->is.full.quad,sideR->treeid,sideR->is.full.quadid,sideR->is.full.is_ghost,cellData);
      _p_get_vec_data(iterCtx->vecdata_in,iterCtx->nVecsIn,iterCtx->vecdata_out,iterCtx->nVecsOut,cellData);
      faceData->cellDataR[0] = cellData;
      cellData++;
    }
  }

  /* call face function */
  ierr = iterCtx->iterFace(faceData,iterCtx->userIterCtx);CHKERRV(ierr);

  /* set vector data */
  cellData = iterCtx->cellData;
  for (i=0; i<(faceData->nCellsL + faceData->nCellsR); i++) {
    _p_set_vec_data(cellData,iterCtx->vecdata_out,iterCtx->nVecsOut);
    cellData++;
  }
}

PetscErrorCode DMBFIterateOverFacesVectors(DM dm, PetscErrorCode (*iterFace)(DM_BF_FaceData*,void*), void *userIterCtx,
                                           Vec *in, PetscInt nVecsIn, Vec *out, PetscInt nVecsOut)
{
  DM_BF             *bf;
  DM_BF_FaceIterCtx iterCtx;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMBF);
  /* set iterator context */
  iterCtx.iterFace    = iterFace;
  iterCtx.userIterCtx = userIterCtx;
  iterCtx.nVecsIn     = nVecsIn;
  iterCtx.nVecsOut    = nVecsOut;
  if (0 < nVecsIn) {
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscMalloc1(nVecsIn,&iterCtx.cellData[i].vecdata_in);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nVecsIn,&iterCtx.vecdata_in);CHKERRQ(ierr);
    for (i=0; i<nVecsIn; i++) {
      ierr = VecGetArrayRead(in[i],&iterCtx.vecdata_in[i]);CHKERRQ(ierr);
    }
  }
  if (0 < nVecsOut) {
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscMalloc1(nVecsOut,&iterCtx.cellData[i].vecdata_out);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nVecsOut,&iterCtx.vecdata_out);CHKERRQ(ierr);
    for (i=0; i<nVecsOut; i++) {
      ierr = VecGetArray(out[i],&iterCtx.vecdata_out[i]);CHKERRQ(ierr);
    }
  }
  /* run iterator */
  bf = _p_GetBF(dm);
  PetscStackCallP4est(p4est_iterate,(bf->p4est,bf->ghost,&iterCtx,NULL,p4est_iter_face,NULL));
  /* clear iterator context */
  if (0 < nVecsIn) {
    for (i=0; i<nVecsIn; i++) {
      ierr = VecRestoreArrayRead(in[i],&iterCtx.vecdata_in[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecdata_in);CHKERRQ(ierr);
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscFree(iterCtx.cellData[i].vecdata_in);CHKERRQ(ierr);
    }
  }
  if (0 < nVecsOut) {
    for (i=0; i<nVecsOut; i++) {
      ierr = VecRestoreArray(out[i],&iterCtx.vecdata_out[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iterCtx.vecdata_out);CHKERRQ(ierr);
    for (i=0; i<3; i++) {//TODO only 2D
      ierr = PetscFree(iterCtx.cellData[i].vecdata_out);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFIterateOverFaces(DM dm, PetscErrorCode (*iterFace)(DM_BF_FaceData*,void*), void *userIterCtx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMBFIterateOverFacesVectors(dm,iterFace,userIterCtx,PETSC_NULL,0,PETSC_NULL,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
