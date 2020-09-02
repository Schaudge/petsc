#include <petscdmp4estfv.h>
#include <petscdmforest.h>
#include <petsc/private/dmforestimpl.h>
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

typedef struct {
  /* topology */
  p4est_connectivity_t  *connectivity;
  p4est_geometry_t      *geometry;//TODO need this?
  /* cells */
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  /* nodes */
  p4est_lnodes_t        *lnodes;
} DM_P4estFV;

typedef struct {
  p4est_t               *p4est;
  p4est_ghost_t         *ghost;
  void                  *userMatCtx;
} DM_P4estFV_MatCtx;

/*
 * Private Functions
 */

static DM_P4estFV *_p_GetP4estFV(DM dm)
{
  return (DM_P4estFV*) ((DM_Forest*) dm->data)->data;
}

/*
 * Function Definitions
 */

static PetscErrorCode DMForestDestroy_P4estFV(DM);
static PetscErrorCode DMClone_P4estFV(DM,DM*);
static PetscErrorCode DMCreateGlobalVector_P4estFV(DM,Vec*);
static PetscErrorCode DMCreateMatrix_P4estFV(DM,Mat*);

/*
 * Public Functions
 */

static PetscErrorCode DMP4estFV_ConnectivityCreate(DM dm, p4est_connectivity_t **connectivity)
{
  const char        *prefix;
  DMForestTopology  topologyName;
  PetscBool         isBrick;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);

  /* get topology name */
  ierr = DMForestGetTopology(dm,&topologyName);CHKERRQ(ierr);
  if (!topologyName) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4estFV needs a topology");
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

static PetscErrorCode DMP4estFV_ConnectivityDestroy(DM dm, p4est_connectivity_t *connectivity)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_connectivity_destroy,(connectivity));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMP4estFV_P4estCreate(DM dm, p4est_connectivity_t *connectivity, p4est_t **p4est)
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

static PetscErrorCode DMP4estFV_P4estDestroy(DM dm, p4est_t *p4est)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_destroy,(p4est));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMP4estFV_GhostCreate(DM dm, p4est_t *p4est, p4est_ghost_t **ghost)
{
  PetscFunctionBegin;
  PetscStackCallP4estReturn(*ghost,p4est_ghost_new,(p4est,P4EST_CONNECT_FULL));
  //TODO which connect flag, P4EST_CONNECT_FULL, P4EST_CONNECT_FACE, ...?
  PetscFunctionReturn(0);
}

static PetscErrorCode DMP4estFV_GhostDestroy(DM dm, p4est_ghost_t *ghost)
{
  PetscFunctionBegin;
  PetscStackCallP4est(p4est_ghost_destroy,(ghost));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_P4estFV(DM dm)
{
  DM_P4estFV     *p4estfv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  p4estfv = _p_GetP4estFV(dm);

  /* create topology */
  if (p4estfv->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity exists already");
  ierr = DMP4estFV_ConnectivityCreate(dm,&p4estfv->connectivity);CHKERRQ(ierr);

  /* create forest of trees */
  if (!p4estfv->connectivity) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Connectivity does not exist");
  if (p4estfv->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est exists already");
  ierr = DMP4estFV_P4estCreate(dm, p4estfv->connectivity, &p4estfv->p4est);CHKERRQ(ierr);

  /* create ghost */
  //if (p4estfv->ghost)  SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Ghost exists already");
  //ierr = DMP4estFV_GhostCreate(dm,p4estfv->p4est,&p4estfv->ghost);CHKERRQ(ierr);

  /* create nodes */
  //TODO

  PetscFunctionReturn(0);
}

static PetscErrorCode DMP4estFVClear(DM dm)
{
  DM_P4estFV     *p4estfv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  p4estfv = _p_GetP4estFV(dm);
//if (p4estfv->lnodes)       { PetscStackCallP4est(p4est_lnodes_destroy,(p4estfv->lnodes)); }
  if (p4estfv->ghost)        { ierr = DMP4estFV_GhostDestroy(dm,p4estfv->ghost);CHKERRQ(ierr); }
  if (p4estfv->p4est)        { ierr = DMP4estFV_P4estDestroy(dm,p4estfv->p4est);CHKERRQ(ierr); }
//if (p4estfv->geometry)     { PetscStackCallP4est(p4est_geometry_destroy,(p4estfv->geometry)); }
  if (p4estfv->connectivity) { ierr = DMP4estFV_ConnectivityDestroy(dm,p4estfv->connectivity);CHKERRQ(ierr); }
  p4estfv->lnodes       = NULL;
  p4estfv->ghost        = NULL;
  p4estfv->p4est        = NULL;
  p4estfv->geometry     = NULL;
  p4estfv->connectivity = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_P4estFV(PetscOptionItems *PetscOptionsObject,DM dm)
{
//DM_Forest_pforest *pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
//char              stringBuffer[256];
//PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMSetFromOptions_Forest(PetscOptionsObject,dm);CHKERRQ(ierr);
  //TODO
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
static PetscErrorCode DMInitialize_P4estFV(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup              = DMSetUp_P4estFV;
  dm->ops->setfromoptions     = DMSetFromOptions_P4estFV;
  dm->ops->clone              = DMClone_P4estFV;
  dm->ops->createglobalvector = DMCreateGlobalVector_P4estFV;
  dm->ops->creatematrix       = DMCreateMatrix_P4estFV;
  //TODO
  //dm->ops->createsubdm    = DMCreateSubDM_Forest;
  //dm->ops->refine         = DMRefine_Forest;
  //dm->ops->coarsen        = DMCoarsen_Forest;
  //dm->ops->adaptlabel     = DMAdaptLabel_Forest;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreate_P4estFV(DM dm)
{
  DM_P4estFV     *p4estfv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  ierr = DMInitialize_P4estFV(dm);CHKERRQ(ierr);
  ierr = DMSetDimension(dm,P4EST_DIM);CHKERRQ(ierr);

  /* set default parameters of Forest object */
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,P4EST_QMAXLEVEL);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  /* create P4estFV */
  ierr = PetscNewLog(dm,&p4estfv);CHKERRQ(ierr);
  p4estfv->connectivity = NULL;
  p4estfv->geometry     = NULL;
  p4estfv->p4est        = NULL;
  p4estfv->ghost        = NULL;
  p4estfv->lnodes       = NULL;

  /* set data & functions of Forest object */
  {
    DM_Forest *forest = (DM_Forest*) dm->data;

    forest->data    = p4estfv;
    forest->destroy = DMForestDestroy_P4estFV;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMForestDestroy_P4estFV(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy contents of P4estFV */
  ierr = DMP4estFVClear(dm);CHKERRQ(ierr);
  /* destroy P4estFV object */
  ierr = PetscFree(((DM_Forest*)dm->data)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_P4estFV(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone_Forest(dm,newdm);CHKERRQ(ierr);
  ierr = DMInitialize_P4estFV(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMP4estFVGetP4est(DM dm, void *p4est)
{
  DM_P4estFV *p4estfv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  p4estfv = _p_GetP4estFV(dm);
  if (!p4estfv->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est does not exist");
  *(void**)p4est = p4estfv->p4est;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMP4estFVGetGhost(DM dm, void *ghost)
{
  DM_P4estFV     *p4estfv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  p4estfv = _p_GetP4estFV(dm);
  if (!p4estfv->ghost) {
    if (!p4estfv->p4est) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"P4est does not exist");
    ierr = DMP4estFV_GhostCreate(dm,p4estfv->p4est,&p4estfv->ghost);CHKERRQ(ierr);
  }
  *(void**)ghost = p4estfv->ghost;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_P4estFV(DM dm, Vec *vec)
{
  DM_P4estFV     *p4estfv;
  PetscInt       n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  PetscValidPointer(vec,2);
  p4estfv = _p_GetP4estFV(dm);
  /* set number of local entries */
  n = (PetscInt) p4estfv->p4est->local_num_quadrants;
  /* set number of global entries */
  N = (PetscInt) p4estfv->p4est->global_num_quadrants;
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

static PetscErrorCode DMCreateMatrix_P4estFV(DM dm, Mat *mat)
{
  DM_P4estFV     *p4estfv;
  void           *appctx;
  PetscInt       n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMP4ESTFV);
  PetscValidPointer(mat,2);
  p4estfv = _p_GetP4estFV(dm);
  ierr = DMGetApplicationContext(dm,&appctx);CHKERRQ(ierr);
  /* set number of local rows/cols */
  n = (PetscInt) p4estfv->p4est->local_num_quadrants;
  /* set number of global rows/cols */
  N = (PetscInt) p4estfv->p4est->global_num_quadrants;
  /* create matrix */
  ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),n,n,N,N,appctx,mat);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  //TODO set null space?
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
