#include "bf_2d_topology.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_ConnectivityCreate      DMBF_2D_ConnectivityCreate
#define DMBF_XD_ConnectivityDestroy     DMBF_2D_ConnectivityDestroy

#define DM_BF_XD_Topology               DM_BF_2D_Topology
#define _p_DM_BF_XD_Topology            _p_DM_BF_2D_Topology

#define DMBF_XD_TopologyCreate          DMBF_2D_TopologyCreate
#define DMBF_XD_TopologyDestroy         DMBF_2D_TopologyDestroy
#define DMBF_XD_TopologyClone           DMBF_2D_TopologyClone
#define DMBF_XD_TopologyGetConnectivity DMBF_2D_TopologyGetConnectivity

/* include generic functions */
#include "bf_xd_topology.c"

static PetscErrorCode DMBF_2D_ConnectivityCreate(DM dm, p4est_connectivity_t **connectivity)
{
  const char       *prefix;
  DMForestTopology topologyName;
  PetscBool        isBrick;
  PetscErrorCode   ierr;

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

#endif /* defined(PETSC_HAVE_P4EST) */
